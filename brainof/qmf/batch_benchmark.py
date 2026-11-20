"""Measure full QMF training updates at fixed effective batch 512 on real LEMON inputs."""
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

import utils
import main_finetuning_qmf as training
from datasets import BrainSignalDataset_multimodal, DataTransform
from models.transformer import Router
from qmf.core import LossHistory
from qmf.model import QMFBrainOF, STREAMS


def measure(args,cfg):
    batch_size=cfg.train.batch_size
    accum=cfg.train.accumulation_steps
    assert batch_size*accum==512
    utils.set_seed(3407)
    transform=DataTransform(cfg.dataset.signal_size,cfg.model.patch_size,cfg.model.max_seq_length)
    dataset=training.Indexed(BrainSignalDataset_multimodal(cfg.root,'train',cfg.dataset.name,transform))
    indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(817))[:128].tolist()
    loader=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,indices),batch_size=batch_size,num_workers=0)
    batches=[training.to_device(b,'cuda:0') for b in loader]
    model=QMFBrainOF(cfg,('m1','m2','fused')).to('cuda:0').train()
    model.load_pretrained(cfg.model.pretrained_model)
    multipliers={'m1':1.875,'m2':0.125,'fused':1.0}
    groups=[]
    for stream,net in model.nets.items():
        values=list(cfg.train.optimizer.layer_decay**i for i in reversed(range(net.num_layers+2)))
        assigner=utils.LayerDecayValueAssigner(values,'DDP')
        part=utils.get_finetune_param_groups_ddp(net,4e-5,cfg.train.optimizer.weight_decay,
              assigner.get_layer_id,assigner.get_scale,
              skip_keywords=['gamma','norm','lambda_qk','patch_embed','router.gate','latents'])
        for group in part:group['lr_scale']*=multipliers[stream]
        groups+=part
    optimizer=torch.optim.AdamW(groups,lr=4e-5,betas=cfg.train.optimizer.betas,eps=cfg.train.optimizer.eps)
    scaler=utils.NativeScalerWithGradNormCount()
    histories={s:LossHistory(len(dataset),'cuda:0') for s in model.streams}
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.criterion.label_smoothing)
    durations=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(accum*4):
        torch.cuda.synchronize();start=time.perf_counter()
        loss,reg,*outputs=training.forward_loss(model,*batches[step%len(batches)],args,cfg,criterion,histories,True,True)
        assert torch.isfinite(loss).item()
        update=(step+1)%accum==0
        scaler(loss/accum,optimizer,clip_grad=cfg.train.clip_grad,parameters=model.parameters(),update_grad=update,model=model)
        for module in model.modules():
            if isinstance(module,Router):module.update_bias(cfg.train.aux_loss_free_alpha)
        if update:optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if step>=accum:durations.append(elapsed)
        if update:print(json.dumps({'batch':batch_size,'completed_optimizer_steps':(step+1)//accum,'loss':float(loss.detach()),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}),flush=True)
    return {'status':'ok','batch_size':batch_size,'accumulation_steps':accum,'effective_batch':512,
            'measured_samples':batch_size*len(durations),'seconds':sum(durations),
            'seconds_per_microbatch':sum(durations)/len(durations),
            'samples_per_second':batch_size*len(durations)/sum(durations),
            'seconds_per_512_samples':512*sum(durations)/(batch_size*len(durations)),
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
            'input_tokens':[(b[0].shape[1],b[1].shape[1]) for b in batches]}


def main():
    args,cfg=training.get_parse_option()
    utils.init_distributed_mode(cfg)
    result={'batch_size':cfg.train.batch_size,'accumulation_steps':cfg.train.accumulation_steps}
    try:
        result=measure(args,cfg)
    except torch.cuda.OutOfMemoryError as error:
        result.update(status='oom',error=str(error))
    finally:
        out = Path(cfg.out_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / f'batch-benchmark-{cfg.train.batch_size}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        dist.destroy_process_group()


if __name__=='__main__':main()
