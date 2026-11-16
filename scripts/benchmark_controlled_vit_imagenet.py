"""Run four disposable throughput pilots on both GPUs after shared preprocessing."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--val-root',type=Path,required=True); p.add_argument('--micro-batch',type=int,default=1024)
    p.add_argument('--tag',default='preprocessed120_full')
    a=p.parse_args()
    packed=a.root/'data/preprocessed'
    while not (packed/'PREPROCESSED_READY.json').exists(): time.sleep(10)
    folder=f'benchmarks_packed_b{a.micro_batch}_{a.tag}'
    def worker(gpu,names):
      for name in names:
        output=a.root/folder/name
        if (output/'benchmark.json').exists(): continue
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with (a.root/f'benchmark_{name}_b{a.micro_batch}_{a.tag}.log').open('a') as log:
            subprocess.run([sys.executable,'-u','scripts/train_controlled_imagenet.py','--model',name,
                '--train-root',str(a.val_root),'--val-root',str(a.val_root),'--output',str(output),
                '--benchmark-steps','100','--benchmark-full-data','--workers','6','--micro-batch',str(a.micro_batch),
                '--packed',str(packed)],
                stdout=log,stderr=subprocess.STDOUT,check=True,cwd=a.root,env=env)
        print(name,'BENCHMARK_COMPLETE',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,0,['V1','V3']),pool.submit(worker,1,['V2','V5'])]
        for f in futures: f.result()
    measured={n:json.loads((a.root/folder/n/'benchmark.json').read_text()) for n in ['V1','V2','V3','V5']}
    # Architectural FLOPs used only to interpolate unmeasured configurations.
    rates={n:v['estimated_train_gpu_hours'] for n,v in measured.items()}
    rates['V4']=rates['V2']*9/12; rates['V6']=rates['V2']; rates['V7']=rates['V2']
    report=dict(measured=measured,estimated_train_gpu_hours_by_model=rates,total_gpu_hours=sum(rates.values()),
        ideal_two_gpu_days=sum(rates.values())/2/24,estimated_models=['V4','V6','V7'],micro_batch=a.micro_batch,
        estimated_queue_days=max(sum(rates[n] for n in ['V1','V3','V4','V6']),sum(rates[n] for n in ['V2','V5','V7']))/24,
        initial_budget_epochs=120,
        limitation='Two concurrent single-GPU100-update pilots on the complete cleaned training index after preprocessing finishes; '
        'no pilot weights reused. Epoch IO variation, validation/checkpoint IO and periodic CFS add uncertainty.',unix=time.time())
    (a.root/'BENCHMARK_PREPROCESSED_COMPLETE.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='measured'}),flush=True)

if __name__=='__main__': main()
