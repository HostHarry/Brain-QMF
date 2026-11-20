"""Real-data correctness and timing checks for three-stream model parallelism."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

import utils
import main_finetuning_qmf as train
from datasets import BrainSignalDataset_multimodal, DataTransform
from models.transformer import Router
from qmf.core import LossHistory, fusion_weights, fuse
from qmf.model_parallel_candidate import QMFBrainOF, update_router_bias


def sync():
    for i in range(3):
        torch.cuda.synchronize(i)


def main():
    args, cfg = train.get_parse_option()
    utils.init_distributed_mode(cfg)
    utils.set_seed(3407)
    torch.set_num_threads(4)
    router=Router(8,4,2).to('cuda:0')
    router.counts.copy_(torch.tensor([1,2,4,9],device='cuda:0'))
    router.update_bias(0.001)
    expected=router.probs_correction_bias.clone()
    router.probs_correction_bias.zero_()
    update_router_bias(router,0.001)
    torch.testing.assert_close(router.probs_correction_bias,expected,rtol=0,atol=0)
    del router,expected
    print('PASS: local router update exactly matches the original world-size-one update',flush=True)
    transform = DataTransform(cfg.dataset.signal_size, cfg.model.patch_size, cfg.model.max_seq_length)
    dataset = train.Indexed(BrainSignalDataset_multimodal(cfg.root, 'train', cfg.dataset.name, transform))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    batch = train.to_device(next(iter(loader)), 'cuda:0')
    model = QMFBrainOF(cfg, ('m1','m2','fused'))
    model.load_pretrained(cfg.model.pretrained_model)
    model.to('cuda:0').eval()
    small = tuple(x[:2] for x in batch)
    def objective():
        logits = model(small[:2], small[2:4])
        fused = fuse(logits, fusion_weights(logits))
        loss = F.cross_entropy(fused, small[4]) + sum(F.cross_entropy(v, small[4]) for v in logits.values())
        return logits, loss
    before, loss = objective()
    loss.backward()
    reference = {s:v.detach().cpu() for s,v in before.items()}
    grads = {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    del before, loss
    model.parallelize(['cuda:1','cuda:2','cuda:0'])
    after, loss = objective()
    loss.backward()
    for s in reference:
        torch.testing.assert_close(after[s].cpu(),reference[s],rtol=2e-4,atol=2e-5)
    for name, parameter in model.named_parameters():
        assert (parameter.grad is not None) == (name in grads), name
        if name in grads:
            torch.testing.assert_close(parameter.grad.cpu(),grads[name],rtol=2e-3,atol=2e-5,msg=name)
    with torch.no_grad(), torch.amp.autocast('cuda',dtype=torch.bfloat16):
        values = model(small[:2],small[2:4])
    assert all(not v.requires_grad and v.dtype == torch.bfloat16 for v in values.values())
    print('PASS: real-data logits and all parameter gradients match; bf16 and no_grad preserved across threads',flush=True)
    del grads, reference, after, loss, values
    model.zero_grad(set_to_none=True)
    model._stream_pool.shutdown(wait=True)
    del model
    gc.collect()
    for i in range(3):
        with torch.cuda.device(i):torch.cuda.empty_cache()
    report = {'correctness':'passed','batch_size':16,'accumulation_steps':32,'measurements':[]}
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.criterion.label_smoothing)
    for mode in ['single','parallel']:
        utils.set_seed(3407)
        model = QMFBrainOF(cfg, ('m1','m2','fused')).to('cuda:0')
        if mode == 'parallel':model.parallelize(['cuda:1','cuda:2','cuda:0'])
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(),lr=4e-5)
        scaler = utils.NativeScalerWithGradNormCount()
        histories = {s:LossHistory(len(dataset),'cuda:0') for s in model.streams}
        durations=[]
        for step in range(72):
            sync(); start=time.perf_counter()
            loss,reg,*_ = train.forward_loss(model,*batch,args,cfg,criterion,histories,True,True)
            update = (step+1)%32==0
            scaler(loss/32,optimizer,clip_grad=5.0,parameters=model.parameters(),update_grad=update,model=model)
            if mode=='parallel':update_router_bias(model,0.001)
            else:
                for module in model.modules():
                    if isinstance(module,Router):module.update_bias(0.001)
            if update:optimizer.zero_grad(set_to_none=True)
            sync(); elapsed=time.perf_counter()-start
            if step>=8:durations.append(elapsed)
            if step%16==0:print(json.dumps({'mode':mode,'step':step,'seconds':elapsed,'loss':float(loss.detach()),'reg':float(reg.detach())}),flush=True)
        row={'mode':mode,'mean_step_seconds':sum(durations)/len(durations),'samples_per_second':16*len(durations)/sum(durations),'parameter_devices':{s:str(next(n.parameters()).device) for s,n in model.nets.items()}}
        model.eval(); sync(); start=time.perf_counter()
        with torch.no_grad():
            for _ in range(20):train.forward_loss(model,*batch,args,cfg,criterion,None,False,False)
        sync();row['evaluation_batch_seconds']=(time.perf_counter()-start)/20
        report['measurements'].append(row)
        print(json.dumps(row),flush=True)
        if hasattr(model,'_stream_pool'):model._stream_pool.shutdown(wait=True)
        del model,optimizer,scaler,histories,loss,reg,_
        gc.collect()
        for i in range(3):
            with torch.cuda.device(i):torch.cuda.empty_cache()
    report['speedup']=report['measurements'][0]['mean_step_seconds']/report['measurements'][1]['mean_step_seconds']
    out = Path(cfg.out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'three-gpu-benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
