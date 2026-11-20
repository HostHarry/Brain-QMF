"""Time a complete real-data epoch, validation, test and checkpoint at batch 64."""
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from timm.scheduler.cosine_lr import CosineLRScheduler

import utils
import main_finetuning_qmf as training
from datasets import BrainSignalDataset_multimodal, DataTransform
from qmf.core import LossHistory
from qmf.model import QMFBrainOF


def main():
    args,cfg=training.get_parse_option();utils.init_distributed_mode(cfg);utils.set_seed(3407)
    assert cfg.train.batch_size==64 and cfg.train.accumulation_steps==8
    logger=SimpleNamespace(info=lambda message:print(message,flush=True))
    transform=DataTransform(cfg.dataset.signal_size,cfg.model.patch_size,cfg.model.max_seq_length)
    sets={s:training.Indexed(BrainSignalDataset_multimodal(cfg.root,s,cfg.dataset.name,transform)) for s in ['train','val','test']}
    common=dict(batch_size=64,num_workers=cfg.num_workers,pin_memory=cfg.pin_memory,multiprocessing_context='fork')
    loaders={s:torch.utils.data.DataLoader(ds,shuffle=s=='train',drop_last=s=='train',**common) for s,ds in sets.items()}
    streams=tuple(args.streams.split(','));model=QMFBrainOF(cfg,streams)
    model.load_pretrained(cfg.model.pretrained_model);model.to('cuda:0')
    lr=cfg.train.optimizer.base_lr;groups=[];multipliers={'m1':1.875,'m2':0.125,'fused':1.0}
    for stream,net in model.nets.items():
        values=list(cfg.train.optimizer.layer_decay**i for i in reversed(range(net.num_layers+2)))
        assigner=utils.LayerDecayValueAssigner(values,'DDP')
        part=utils.get_finetune_param_groups_ddp(net,lr,cfg.train.optimizer.weight_decay,assigner.get_layer_id,assigner.get_scale,skip_keywords=['gamma','norm','lambda_qk','patch_embed','router.gate','latents'])
        for group in part:group['lr_scale']*=multipliers[stream]
        groups+=part
    optimizer=torch.optim.AdamW(groups,lr=lr,betas=cfg.train.optimizer.betas,eps=cfg.train.optimizer.eps)
    per_epoch=math.ceil(len(loaders['train'])/8);warmup=int(cfg.train.lr_scheduler.warmup_epochs*per_epoch)
    scheduler=CosineLRScheduler(optimizer,t_initial=cfg.train.epochs*per_epoch-(warmup if cfg.train.lr_scheduler.warmup_prefix else 0),cycle_mul=1.,lr_min=cfg.train.lr_scheduler.min_lr,warmup_lr_init=cfg.train.lr_scheduler.warmup_lr,warmup_t=warmup,cycle_limit=1,warmup_prefix=cfg.train.lr_scheduler.warmup_prefix)
    scaler=utils.NativeScalerWithGradNormCount();criterion=torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.criterion.label_smoothing)
    histories={s:LossHistory(len(sets['train']),'cuda:0') for s in streams}
    for history in histories.values():
        history.cnt.fill_(1);history.sum.copy_(torch.linspace(0.1,1.1,len(sets['train']),device='cuda:0'))
    result={'tag':args.tag,'batch_size':64,'accumulation_steps':8,'streams':streams,'training_batches':len(loaders['train']),'seconds':{},'notes':'Real DataLoaders, synthetic initialized loss histories for timing; not an accuracy experiment'}
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    training.train_one_epoch(model,loaders['train'],optimizer,scheduler,scaler,cfg.train.aux_loss_free_alpha,1,'cuda:0',cfg,args,criterion,histories,logger)
    torch.cuda.synchronize();result['seconds']['train']=time.perf_counter()-start
    for split in ['val','test']:
        start=time.perf_counter()
        if split=='val':metrics=training.evaluate(model,loaders[split],'cuda:0',cfg,args,criterion)
        else:metrics,outputs=training.evaluate(model,loaders[split],'cuda:0',cfg,args,criterion,return_outputs=True)
        torch.cuda.synchronize();result['seconds'][split]=time.perf_counter()-start
        print(json.dumps({'finished':split,'seconds':result['seconds'][split]}),flush=True)
    start=time.perf_counter()
    out = Path(cfg.out_root)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='batch-timing-',dir=out) as temporary:
        np.savez_compressed(Path(temporary)/'outputs.npz',**outputs)
        if not args.no_save:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'model':state,'args':vars(args)},Path(temporary)/'checkpoint.pth')
    result['seconds']['save']=time.perf_counter()-start
    result['seconds']['full_epoch_with_test']=sum(result['seconds'].values())
    result['seconds']['epoch_without_test']=result['seconds']['train']+result['seconds']['val']
    result['peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
    (out / f'batch64-epoch-{args.tag}.json').write_text(json.dumps(result,indent=2))
    print('FULL_EPOCH_RESULT '+json.dumps(result),flush=True);dist.destroy_process_group()


if __name__=='__main__':main()
