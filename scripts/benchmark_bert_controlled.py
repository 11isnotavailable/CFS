"""Finite benchmark only: 50 warmup +100 measured updates; no checkpoints/CFS."""
import argparse,gc,json,os,subprocess,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import transformers
from src.bert_throughput import SPECS,make_model,mask_tokens,sparse_mlm_loss

class Tokens(Dataset):
    def __init__(self,path): self.rows=np.load(path,mmap_mode='r')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return torch.from_numpy(self.rows[i].astype(np.int64))

def batches(path,batch):
    loader=DataLoader(Tokens(path),batch_size=batch,shuffle=True,drop_last=True,
        num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2,
        generator=torch.Generator().manual_seed(42))
    if not len(loader): raise ValueError('Real cache too small for batch')
    while True:
        for ids in loader: yield ids.to('cuda',non_blocking=True)

class Telemetry:
    def __init__(self,gpu): self.gpu=gpu; self.stop=threading.Event(); self.rows=[]
    def sample(self):
        while not self.stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','-i',str(self.gpu),
                    '--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=4)
                self.rows.append(dict(unix=time.time(),**dict(zip(['gpu_util_percent','memory_mib','power_w'],map(float,raw.strip().split(','))))))
            except Exception as e: self.rows.append(dict(error=str(e)))
            self.stop.wait(.5)
    def start(self): self.thread=threading.Thread(target=self.sample,daemon=True); self.thread.start()
    def finish(self): self.stop.set(); self.thread.join(timeout=5)

def verify():
    # TF32's shape-dependent GEMM rounding is not an algebraic objective check.
    previous_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42); model=make_model('B1',dropout=0).cuda().eval()
    ids=torch.randint(104,30522,(2,128),device='cuda'); ids[:,0]=101; ids[:,-1]=102
    ids,labels=mask_tokens(ids)
    full=model(input_ids=ids,labels=labels).loss; full.backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True); sparse,_=sparse_mlm_loss(model,ids,labels); sparse.backward()
    torch.testing.assert_close(full,sparse,rtol=1e-5,atol=1e-6)
    maximum=0.
    for n,p in model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(grads[n],p.grad,rtol=3e-4,atol=2e-6)
            maximum=max(maximum,(grads[n]-p.grad).abs().max().item())
    del model,grads; gc.collect(); torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32=previous_tf32
    return dict(full_loss=full.item(),sparse_loss=sparse.item(),max_gradient_absolute_error=maximum,
        dropout_disabled_for_equivalence_only=True)

def run(name,batch,mode,a):
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    model=make_model(name).cuda().train()
    decay=[p for n,p in model.named_parameters() if p.ndim>=2]
    other=[p for n,p in model.named_parameters() if p.ndim<2]
    opt=torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':other,'weight_decay':0}],
        lr=1e-4,betas=(.9,.999),eps=1e-8,fused=True)
    pool=None; stream=None
    if mode=='resident':
        pool=torch.randint(104,30522,(batch,128),device='cuda'); pool[:,0]=101; pool[:,-1]=102
    else: stream=batches(a.root/'data/train_tokens.npy',batch)
    input_waits=[]
    def step():
        supply_started=time.perf_counter()
        ids=pool if pool is not None else next(stream)
        input_waits.append(time.perf_counter()-supply_started)
        corrupted,labels=mask_tokens(ids)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss,masked=sparse_mlm_loss(model,corrupted,labels)
        loss.backward(); opt.step()
        return loss.detach(),masked
    started=time.perf_counter()
    for _ in range(50): step()
    torch.cuda.synchronize(); warmup_seconds=time.perf_counter()-started
    torch.cuda.reset_peak_memory_stats(); telemetry=Telemetry(a.gpu); telemetry.start()
    durations=[]; losses=[]; targets=0; began=time.perf_counter()
    for i in range(a.updates):
        tick=time.perf_counter(); loss,count=step(); torch.cuda.synchronize()
        durations.append(time.perf_counter()-tick); losses.append(loss); targets+=count
    elapsed=time.perf_counter()-began; telemetry.finish()
    peak=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
    assert torch.isfinite(torch.stack(losses)).all(),'Nonfinite MLM loss'
    # Inspect actual dispatched attention operators outside the timed window.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile: step(); torch.cuda.synchronize()
    kernels=[e.key for e in profile.key_averages() if 'attention' in e.key.lower()]
    utilization=[r['gpu_util_percent'] for r in telemetry.rows if 'gpu_util_percent' in r]
    value=dict(model=name,width=SPECS[name][0],depth=SPECS[name][1],heads=SPECS[name][2],
        parameters=sum(p.numel() for p in model.parameters()),batch_sequences=batch,sequence_length=128,
        mode=mode,gpu=a.gpu,warmup_updates=50,measured_updates=a.updates,warmup_seconds=warmup_seconds,
        mean_host_input_wait_seconds=float(np.mean(input_waits[50:50+a.updates])),
        host_input_wait_fraction=float(sum(input_waits[50:50+a.updates])/elapsed),
        input_wait_scope='Host next-batch wait plus asynchronous H2D enqueue; device-copy completion not isolated',
        seconds=elapsed,seconds_per_update=elapsed/a.updates,p50_seconds=np.median(durations),p95_seconds=np.quantile(durations,.95),
        input_tokens_per_second=batch*128*a.updates/elapsed,text_tokens_per_second=batch*126*a.updates/elapsed,
        masked_targets_per_second=targets/elapsed,masked_fraction=targets/(batch*126*a.updates),
        gpu_utilization_mean=np.mean(utilization) if utilization else None,
        gpu_utilization_min=min(utilization) if utilization else None,
        gpu_utilization_max=max(utilization) if utilization else None,telemetry_samples=len(utilization),
        peak_allocated_gib=peak/2**30,peak_reserved_gib=reserved/2**30,
        measured_gpu_hours=elapsed/3600,warmup_and_measured_gpu_hours=(elapsed+warmup_seconds)/3600,
        attention_operators=kernels,step_seconds=durations,telemetry=telemetry.rows,
        last_loss=losses[-1].item(),precision='BF16 autocast, FP32 parameters/AdamW',
        dropout=.1,compile=False,optimizer='fused AdamW',masking='dynamic GPU Bernoulli15%,80/10/10')
    print(json.dumps({k:v for k,v in value.items() if k not in ['step_seconds','telemetry']}),flush=True)
    del model,opt,pool,stream,losses; gc.collect(); torch.cuda.empty_cache()
    return value

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--models',nargs='+',choices=SPECS,required=True)
    p.add_argument('--updates',type=int,choices=[100,200],default=100)
    p.add_argument('--batches',type=int,nargs='+',choices=[256,512,1024],default=[256,512,1024])
    p.add_argument('--modes',nargs='+',choices=['resident','pipeline'],default=['resident','pipeline'])
    p.add_argument('--output-name',default='measurements'); a=p.parse_args()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=True
    out=a.root/a.output_name; out.mkdir(parents=True,exist_ok=True)
    checks=verify(); (out/f'correctness_gpu{a.gpu}.json').write_text(json.dumps(checks,indent=2))
    environment=dict(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),total_vram=torch.cuda.get_device_properties(0).total_memory)
    (out/f'environment_gpu{a.gpu}.json').write_text(json.dumps(environment,indent=2))
    for mode in a.modes:
        if mode=='pipeline':
            deadline=time.time()+900
            while not (a.root/'data/READY.json').exists():
                if time.time()>deadline: raise TimeoutError('Real-data preparation unavailable after15minutes; resident results retained')
                time.sleep(5)
        for name in a.models:
            for batch in a.batches:
                path=out/f'{name}_{mode}_b{batch}.json'
                if path.exists(): continue
                try: result=run(name,batch,mode,a)
                except torch.cuda.OutOfMemoryError:
                    result=dict(model=name,mode=mode,batch_sequences=batch,error='CUDA OOM')
                    gc.collect(); torch.cuda.empty_cache()
                path.write_text(json.dumps(result,indent=2))
    (out/f'COMPLETE_gpu{a.gpu}.json').write_text(json.dumps(dict(models=a.models,formal_training=False)))

if __name__=='__main__': main()
