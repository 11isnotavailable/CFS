"""Frozen DeiT-style training using timm augmentation, optimizer, scheduler and EMA.

Epoch-boundary atomic resume (failed partial epochs are replayed). torchrun DDP
and single-GPU accumulation have the same effective batch and sample budget.
Benchmarks use fresh disposable models, never production checkpoints.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from timm.data import create_transform, FastCollateMixup
from timm.data.loader import PrefetchLoader, fast_collate
from timm.data.distributed_sampler import RepeatAugSampler
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
from src.controlled_vit_imagenet import SPECS, PROTOCOL as P, MEAN, STD, make_model, config, protocol_hash
from src.indexed_image_dataset import IndexedImages


def seed(value):
    random.seed(value); np.random.seed(value % (2**32)); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def worker_seed(worker):
    s=torch.initial_seed() % (2**32); random.seed(s); np.random.seed(s)


def atomic_torch(value,path):
    temp=path.with_suffix('.tmp'); torch.save(value,temp); temp.replace(path)


def atomic_json(value,path):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)); temp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def make_loader(root,training,batch,workers,rank,world,device,benchmark=False,packed=None,benchmark_full=False):
    transform=create_transform(input_size=(3,224,224),is_training=training,use_prefetcher=True,
        interpolation='bicubic',mean=MEAN,std=STD,crop_pct=P['crop_pct'],
        scale=P['random_crop_scale'],ratio=P['random_crop_ratio'],hflip=P['hflip'],
        color_jitter=P['color_jitter'],auto_augment=P['auto_augment'] if training else None)
    storage=json.loads((Path(packed)/'storage.json').read_text()) if packed and (Path(packed)/'storage.json').exists() else {}
    split='train' if training and (not benchmark or benchmark_full) else 'benchmark' if training and storage else 'val'
    if not training and storage.get('val')=='chw_uint8': transform=None
    dataset=IndexedImages(packed,split,transform) if packed else ImageFolder(root,transform=transform)
    expected=json.loads((Path(packed).parent/'DATA_READY.json').read_text())['training_images_after_overlap_exclusion'] if training and packed and not benchmark else (1281167 if training else 50000)
    if not benchmark and len(dataset)!=expected:
        raise ValueError(f'Unexpected dataset size: {root}: {len(dataset)}')
    classes=dataset.class_to_idx
    if training:
        sampler=RepeatAugSampler(dataset,num_replicas=world,rank=rank,num_repeats=3)
        collate=FastCollateMixup(mixup_alpha=.8,cutmix_alpha=1.,prob=1.,switch_prob=.5,
                               mode='batch',label_smoothing=.1,num_classes=1000)
    else:
        dataset=Subset(dataset,list(range(rank,len(dataset),world)))
        sampler=None; collate=fast_collate
    raw=DataLoader(dataset,batch_size=batch,sampler=sampler,shuffle=False,num_workers=workers,
        pin_memory=True,drop_last=training,collate_fn=collate,worker_init_fn=worker_seed,
        persistent_workers=False,**(dict(prefetch_factor=2) if workers else {}))
    loader=PrefetchLoader(raw,mean=MEAN,std=STD,device=device,img_dtype=torch.float32,
        re_prob=P['re_prob'] if training else 0.,re_mode=P['re_mode'],re_count=1)
    return loader,sampler,classes


@torch.no_grad()
def evaluate(model,loader,device,world,rank,probe_indices):
    model.eval(); totals=torch.zeros(6,device=device,dtype=torch.float64); offset=0
    excluded=torch.zeros(50000,device=device,dtype=torch.bool); excluded[probe_indices]=True
    for x,y in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(x)
        ce=torch.nn.functional.cross_entropy(logits.float(),y,reduction='none')
        if not torch.isfinite(ce).all(): raise FloatingPointError('Nonfinite validation CE')
        top=logits.topk(5,dim=-1).indices
        ids=(torch.arange(len(y),device=device)+offset)*world+rank; offset+=len(y)
        monitor=~excluded[ids]
        totals+=torch.stack([ce.sum(),(top[:,0]==y).sum(),(top==y[:,None]).any(-1).sum(),y.new_tensor(y.numel()),
                             ce[monitor].sum(),monitor.sum()])
    if world>1: dist.all_reduce(totals)
    ce,acc,top5,n,monitor_ce,monitor_n=totals.cpu().tolist()
    assert int(n)==50000 and int(monitor_n)==49488
    return dict(val_ce=ce/n,top1=acc/n,top5=top5/n,val_samples=int(n),monitor_ce=monitor_ce/monitor_n,monitor_samples=int(monitor_n))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=SPECS,required=True); p.add_argument('--train-root',type=Path,required=True)
    p.add_argument('--val-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--micro-batch',type=int,default=256); p.add_argument('--workers',type=int,default=6)
    p.add_argument('--resume',action='store_true'); p.add_argument('--benchmark-steps',type=int,default=0)
    p.add_argument('--benchmark-full-data',action='store_true',help='Pilot on the complete cleaned training index')
    p.add_argument('--packed',type=Path,help='Shared indexed image packs; storage.json identifies preprocessing')
    p.add_argument('--probe-metadata',type=Path,default=Path('/data/cfs_rebuttal_20261114/outputs/vit_small/metadata.json'))
    p.add_argument('--stop-after-epoch',type=int,help='Operational pause; never changes the common120-epoch schedule')
    a=p.parse_args(); world=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0')); torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.benchmark=True
    assert P['global_batch']%(a.micro_batch*world)==0
    accum=P['global_batch']//(a.micro_batch*world); seed(P['seed'])
    a.output.mkdir(parents=True,exist_ok=True)
    if not a.benchmark_steps:
        ready=a.train_root.parent/'DATA_READY.json'
        if not ready.exists(): raise RuntimeError('Complete verified training data required')
    model=make_model(a.model).to(device)
    initial_hash=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    optimizer=create_optimizer_v2(model,opt='adamw',lr=P['learning_rate'],weight_decay=P['weight_decay'],
        eps=P['eps'],betas=tuple(P['betas']),filter_bias_and_bn=True)
    ema=ModelEmaV2(model,decay=P['ema_decay'])
    train,sampler,classes=make_loader(a.train_root,True,a.micro_batch,a.workers,rank,world,device,bool(a.benchmark_steps),a.packed,a.benchmark_full_data)
    steps=len(train)//accum
    if steps<1: raise RuntimeError('Not enough benchmark/training images for one effective batch')
    val=None
    if not a.benchmark_steps:
        val,_,vc=make_loader(a.val_root,False,a.micro_batch,a.workers,rank,world,device,packed=a.packed)
        assert classes==vc and len(classes)==1000
    scheduler=CosineLRScheduler(optimizer,t_initial=P['schedule_epochs']*steps,lr_min=P['min_lr'],
        warmup_t=P['warmup_epochs']*steps,warmup_lr_init=P['warmup_lr'],t_in_epochs=False)
    start=0; history=[]; gpu_hours=0.
    probe_indices=json.loads(a.probe_metadata.read_text())['sample_indices']; assert len(set(probe_indices))==512
    checkpoint=a.output/'last.pt'
    if checkpoint.exists():
        if not a.resume: raise RuntimeError('Existing checkpoint: explicitly request resume')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['protocol_hash']==protocol_hash() and state['model_id']==a.model
        assert state['world_size']==world and state['micro_batch']==a.micro_batch
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        ema.module.load_state_dict(state['ema']); scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'][rank]); start=state['epoch']; history=state['history']; gpu_hours=state['gpu_hours']
    elif a.resume and (a.output/'run.json').exists():
        print('REPLAY_FROM_INITIALIZATION: no completed epoch checkpoint',flush=True)
    if rank==0:
        atomic_json(dict(model_id=a.model,model_config=config(a.model),norm_eps=1e-6,protocol=P,
            protocol_hash=protocol_hash(),initial_state_sha256=initial_hash,from_scratch=True,
            pretrained_loaded=False,world_size=world,micro_batch=a.micro_batch,accumulation=accum,
            updates_per_epoch=steps,sample_visits_per_epoch=steps*1024,
            scheduled_sample_visits=steps*1024*P['epochs'],
            benchmark=bool(a.benchmark_steps)),a.output/'run.json')
    ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    def ensure_monitor(epoch):
        if a.benchmark_steps or epoch not in P['light_cfs']['epochs']: return
        if rank==0:
            torch.cuda.empty_cache()
            out=a.output/'light_cfs'/f'epoch_{epoch:03d}'; out.mkdir(parents=True,exist_ok=True)
            export_path=a.output/f'epoch_{epoch:03d}.pt'
            if not export_path.exists() and epoch==start:
                atomic_torch(dict(model_config=config(a.model),state_dict=model.state_dict(),class_to_idx=classes,
                    model_id=a.model,epoch=epoch,protocol_hash=protocol_hash()),export_path)
            with (out/'monitor.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve().with_name('monitor_controlled_vit_imagenet.py')),
                    '--model',a.model,'--checkpoint',str(a.output/f'epoch_{epoch:03d}.pt'),'--data-root',str(a.val_root),
                    '--probe-metadata',str(a.probe_metadata),'--output',str(out),'--epoch',str(epoch)],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        if world>1: dist.barrier()
    # A failure after epoch checkpointing cannot silently skip its CFS monitor.
    for e in P['light_cfs']['epochs']:
        if e<=start: ensure_monitor(e)
    criterion=SoftTargetCrossEntropy(); stop=min(P['epochs'],a.stop_after_epoch or P['epochs'])
    for epoch in range(start,stop):
        # Epoch seed resets augmentation workers and model stochasticity reproducibly.
        seed(P['seed']+100000*epoch+rank); sampler.set_epoch(P['seed']+epoch)
        ddp.train(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); began=time.perf_counter()
        total_loss=0.; count=0; durations=[]; step_started=began
        for i,(x,y) in enumerate(train):
            if i>=steps*accum: break
            update=(i+1)%accum==0
            sync=nullcontext() if update or world==1 else ddp.no_sync()
            with sync:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=criterion(ddp(x),y)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, microstep {i}')
                (loss/accum).backward()
            total_loss+=loss.detach().item()*x.shape[0]; count+=x.shape[0]
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); ema.update(model)
                completed=(i+1)//accum
                scheduler.step_update(epoch*steps+completed)
                if a.benchmark_steps:
                    torch.cuda.synchronize(); now=time.perf_counter(); durations.append(now-step_started); step_started=now
                if rank==0 and completed%(5 if a.benchmark_steps else 25)==0:
                    print(json.dumps(dict(model=a.model,epoch=epoch+1,update=completed,total_updates=steps,
                        loss=total_loss/count,elapsed_seconds=time.perf_counter()-began)),flush=True)
                if a.benchmark_steps and completed>=a.benchmark_steps: break
        torch.cuda.synchronize(); train_sec=time.perf_counter()-began
        if a.benchmark_steps:
            discard=min(10,max(0,len(durations)//3)); warm=durations[discard:]
            result=dict(model=a.model,steps=len(durations),warmup_discarded=discard,
                full_training_index=a.benchmark_full_data,
                step_seconds=durations,median_seconds_per_update=float(np.median(warm)),
                mean_seconds_per_update=float(np.mean(warm)),
                images_per_second=1024/float(np.mean(warm)),peak_memory_bytes=torch.cuda.max_memory_allocated(),
                train_seconds=train_sec,world_size=world,micro_batch=a.micro_batch,
                initial_budget_epochs=P['epochs'],estimated_train_gpu_hours=P['epochs']*1250*float(np.mean(warm))*world/3600,
                scope='Real decode/augmentation/forward/backward/AdamW/EMA; excludes full validation and checkpoint IO')
            if rank==0: atomic_json(result,a.output/'benchmark.json'); print(json.dumps(result),flush=True)
            break
        totals=torch.tensor([total_loss,count],device=device,dtype=torch.float64)
        if world>1: dist.all_reduce(totals)
        validation_started=time.perf_counter(); metrics=evaluate(model,val,device,world,rank,probe_indices)
        torch.cuda.synchronize(); val_sec=time.perf_counter()-validation_started
        gpu_hours+=(time.perf_counter()-began)*world/3600
        row=dict(epoch=epoch+1,train_loss=float(totals[0]/totals[1]),train_seconds=train_sec,
            validation_seconds=val_sec,lr=optimizer.param_groups[0]['lr'],gpu_hours=gpu_hours,**metrics)
        history.append(row)
        local_rng=rng_state(); states=[None]*world
        if world>1: dist.all_gather_object(states,local_rng)
        else: states[0]=local_rng
        if rank==0:
            state=dict(model_id=a.model,epoch=epoch+1,protocol_hash=protocol_hash(),model=model.state_dict(),
                optimizer=optimizer.state_dict(),ema=ema.module.state_dict(),scheduler=scheduler.state_dict(),
                rng=states,world_size=world,micro_batch=a.micro_batch,history=history,gpu_hours=gpu_hours,
                amp_dtype='bfloat16',grad_scaler=None,sampler_epoch=P['seed']+epoch)
            atomic_torch(state,checkpoint)
            atomic_json(history,a.output/'history.json'); print(json.dumps(row),flush=True)
            if epoch+1 in P['checkpoint_epochs']:
                # Existing model loader accepts this schema; LayerNorm default is eps=1e-6.
                export=dict(model_config=config(a.model),state_dict=model.state_dict(),class_to_idx=classes,
                            model_id=a.model,epoch=epoch+1,protocol_hash=protocol_hash())
                atomic_torch(export,a.output/f'epoch_{epoch+1:03d}.pt')
        if world>1: dist.barrier()
        ensure_monitor(epoch+1)
    if not a.benchmark_steps and history and history[-1]['epoch']==P['epochs'] and rank==0:
        atomic_json(dict(model_id=a.model,checkpoint='epoch_120.pt',selected_epoch=120,stopped_epoch=120,
            stop_reason='common_initial_budget_complete',sample_visits=steps*1024*120,
            checkpoint_selection=P['checkpoint_selection'],gpu_hours=gpu_hours,
            **{k:v for k,v in history[-1].items() if k not in ['epoch','gpu_hours']}),a.output/'COMPLETE.json')
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
