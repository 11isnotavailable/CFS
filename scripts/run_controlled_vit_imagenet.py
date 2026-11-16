"""Persistent seven-model queue with bounded epoch-resume retries and offline finish."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.controlled_vit_imagenet import SPECS, controls, protocol_hash


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()


def record(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)); temp.replace(path)


def run(args,root,log,env):
    started=time.time()
    with log.open('a') as f:
        result=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
    return result.returncode,time.time()-started


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--historical',type=Path,default=Path('/data/cfs_rebuttal_20261114'))
    p.add_argument('--lock',action='store_true'); a=p.parse_args(); root=a.root.resolve()
    if a.lock:
        files=sorted((root/'src').glob('*.py'))+sorted((root/'scripts').glob('*controlled*imagenet*.py'))
        files += [root/'scripts'/name for name in ['analyze_controlled_vit_stability.py','collect_predictor_dataset.py','run_experiment0.py',
            'analyze_matched_scaling.py','analyze_scaling_cfs.py','analyze_functional_capacity.py','analyze_qwen_cfs.py']]
        files+=sorted((root/'configs').rglob('*.json'))
        payload=dict(protocol_hash=protocol_hash(),created_unix=time.time(),
                     sources={str(f.relative_to(root)):digest(f) for f in files})
        lock=root/'PROTOCOL_LOCK.json'
        if lock.exists(): raise RuntimeError('Protocol already locked; do not overwrite')
        record(lock,payload); print('PROTOCOL_LOCKED',flush=True); return
    handle=(root/'pipeline.lock').open('w'); fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    assert lock['protocol_hash']==protocol_hash()
    for name,expected in lock['sources'].items():
        if digest(root/name)!=expected: raise RuntimeError(f'Locked source modified: {name}')
    state=root/'STATUS.json'
    for requirement in ['BENCHMARK_PREPROCESSED_COMPLETE.json','data/DATA_READY.json','data/preprocessed/PREPROCESSED_READY.json']:
        while not (root/requirement).exists():
            record(state,dict(stage='waiting',requirement=requirement,unix=time.time()))
            time.sleep(30)
    micro_batch=int(json.loads((root/'BENCHMARK_PREPROCESSED_COMPLETE.json').read_text())['micro_batch'])
    if micro_batch not in [256,512,1024]: raise ValueError('Unexpected benchmark microbatch')
    if not (root/'smoke_only/CHECKS_PASSED.json').exists(): raise RuntimeError('CFS integration checks must pass before training')
    # Bind all future paired comparisons to the same byte-checked input identities.
    import numpy as np
    provenance=json.loads((a.historical/'imagenet_provenance.json').read_text())
    metadata=json.loads((a.historical/'outputs/vit_small/metadata.json').read_text())
    indices=metadata['sample_indices']; assert len(indices)==512
    assert indices==np.linspace(0,49999,512,dtype=int).tolist()
    images=sorted(provenance['images'],key=lambda x:x['path']); valroot=a.historical/'data/imagenet1k_val'
    chosen=[]
    for i in indices:
        image=images[i]; assert digest(valroot/image['path'])==image['sha256']
        chosen.append(dict(dataset_index=i,**image))
    identity=hashlib.sha256(json.dumps(chosen,sort_keys=True).encode()).hexdigest()
    record(root/'input_identity.json',dict(sha256=identity,images=chosen))
    # Parameter counts come from instantiated audit, never an assumed formula.
    architectures={x['model']:x for x in json.loads((root/'audit/architectures.json').read_text())}
    entries=[]
    for name,(w,d,h) in SPECS.items():
        ns,ks=controls(name)
        entries.append(dict(name=name,family='ViT',model_id=f'controlled_{name}',parameters=architectures[name]['parameters'],
            heads=h,samples=512,layers=[d-1],N=ns,K=ks,path=f'cfs/{name}',input_identity_sha256=identity))
    record(root/'cfs_manifest.json',dict(models=entries))
    record(state,dict(stage='training',queues={'0':['V1','V3','V4','V6'],'1':['V2','V5','V7']},unix=time.time()))
    def worker(gpu,names):
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        for name in names:
            output=root/'runs'/name; output.mkdir(parents=True,exist_ok=True)
            if not (output/'COMPLETE.json').exists():
                success=False
                for attempt in range(3):
                    cmd=[sys.executable,'-u','scripts/train_controlled_imagenet.py','--model',name,
                        '--train-root',str(root/'data/train'),'--val-root',str(valroot),'--output',str(output),
                        '--workers','6','--micro-batch',str(micro_batch),'--resume','--packed',str(root/'data/preprocessed')]
                    code,seconds=run(cmd,root,output/f'attempt_{time.time_ns()}.log',env)
                    with (output/'attempts.jsonl').open('a') as log:
                        log.write(json.dumps(dict(exit_code=code,seconds=seconds,gpu=gpu,attempt=attempt))+'\n')
                    if code==0 and (output/'COMPLETE.json').exists(): success=True; break
                    time.sleep(15)
                if not success: raise RuntimeError(f'{name}: three failed training attempts; inspect logs')
            checkpoint=output/'epoch_120.pt'; cfs=root/'cfs'/name; cfs.mkdir(parents=True,exist_ok=True)
            if not (cfs/'COMPLETE.json').exists():
                code,seconds=run([sys.executable,'-u','scripts/collect_controlled_vit_imagenet.py',
                    '--model',f'controlled_{name}','--checkpoint',str(checkpoint),'--data-root',str(valroot),
                    '--output-dir',str(cfs),'--num-samples','512','--indices-metadata',str(a.historical/'outputs/vit_small/metadata.json'),
                    '--layers',str(SPECS[name][1]-1),'--alpha-steps','31','--alpha-min','0','--alpha-max','3',
                    '--shard-size','32','--variant-batch-size','128','--save-alpha-error-cube'],root,cfs/'collect.log',env)
                if code: raise RuntimeError(f'{name} CFS collection failed; rerun queue to resume shards')
                assert len(list(cfs.glob('shard_*.npz')))==16
                record(cfs/'COMPLETE.json',dict(checkpoint_sha256=digest(checkpoint),input_identity_sha256=identity,
                                               seconds=seconds,samples=512))
            print(name,'TRAIN_AND_CFS_COMPLETE',flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,0,['V1','V3','V4','V6']),pool.submit(worker,1,['V2','V5','V7'])]
            for future in futures: future.result()
        record(root/'COMMON_BUDGET.json',dict(epoch=120,selection='prespecified common120-epoch primary checkpoint'))
        record(state,dict(stage='analysis',unix=time.time()))
        code,_=run([sys.executable,'scripts/analyze_controlled_vit_imagenet.py','--root',str(root),
                    '--output',str(root/'analysis')],root,root/'analysis.log',os.environ.copy())
        if code: raise RuntimeError('Offline analysis failed; raw data remain intact')
        code,_=run([sys.executable,'scripts/analyze_controlled_vit_stability.py','--root',str(root),
                    '--output',str(root/'stability')],root,root/'stability.log',os.environ.copy())
        if code: raise RuntimeError('CE/CFS stability review failed')
        record(state,dict(stage='complete',unix=time.time(),report=str(root/'analysis/REPORT.md')))
    except Exception as exc:
        record(state,dict(stage='failed',error=repr(exc),unix=time.time())); raise


if __name__=='__main__': main()
