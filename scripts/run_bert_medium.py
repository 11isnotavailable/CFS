"""Persistent bounded seven-model2B queue; no automatic3B extension."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.bert_medium import P,SPECS,fingerprint

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def run(cmd,root,env,log):
    started=time.time()
    with log.open('a') as f:code=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    return code,time.time()-started

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--lock',action='store_true');a=p.parse_args();root=a.root.resolve()
    sources=list((root/'scripts').glob('*bert_medium*.py'))+list((root/'src').glob('bert_medium*.py'))
    sources += [root/'src/bert_throughput.py',root/'src/text_head_intervention.py']
    sources += [root/'scripts'/s for s in ['benchmark_bert_controlled.py','collect_bert_mlm_scaling.py','analyze_matched_scaling.py','analyze_functional_capacity.py','analyze_qwen_cfs.py','analyze_scaling_cfs.py']]
    lock=root/'PROTOCOL_LOCK.json'
    if a.lock:
        if lock.exists():raise RuntimeError('Do not overwrite frozen lock')
        atomic(lock,dict(protocol=P,specs=SPECS,protocol_hash=fingerprint(),sources={str(p.relative_to(root)):sha(p) for p in sources}));return
    handle=(root/'queue.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    payload=json.loads(lock.read_text());assert payload['protocol_hash']==fingerprint()
    for name,digest in payload['sources'].items():assert sha(root/name)==digest,f'Locked code changed: {name}'
    assert (root/'smoke/CFS_CHECKS_PASSED.json').exists()
    for gpu in [0,1]:assert (root/f'benchmark/measurements/COMPLETE_gpu{gpu}.json').exists()
    while not (root/'data/READY.json').exists():
        atomic(root/'STATUS.json',dict(stage='waiting_for_frozen_data',unix=time.time()));time.sleep(30)
    data=json.loads((root/'data/READY.json').read_text());assert data['protocol_sha256']==fingerprint() and data['input_tokens']==2000000000
    assert sha(root/'data/validation.npz')==data['validation_sha256'] and sha(root/'data/cfs_inputs.npz')==data['cfs_inputs_sha256']
    if not (root/'smoke/TRAIN_RESUME_CHECKS_PASSED.json').exists():
        atomic(root/'STATUS.json',dict(stage='training_resume_preflight',unix=time.time()))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        code,_=run([sys.executable,'-u','scripts/check_bert_medium_resume.py','--root',str(root)],root,env,root/'resume_preflight.log')
        if code:
            atomic(root/'STATUS.json',dict(stage='failed',error='Training/resume preflight failed',unix=time.time()))
            raise RuntimeError('Training/resume preflight failed before production')
    queues={0:['B2','B7','B5'],1:['B1','B3','B4','B6']}
    atomic(root/'STATUS.json',dict(stage='training_and_cfs',queues=queues,unix=time.time()))
    def worker(gpu,names):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        for name in names:
            out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
            if not (out/'COMPLETE.json').exists():
                for attempt in range(3):
                    code,seconds=run([sys.executable,'-u','scripts/train_bert_medium.py','--model',name,'--root',str(root),'--output',str(out),'--resume'],root,env,out/f'attempt_{time.time_ns()}.log')
                    with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(attempt=attempt,gpu=gpu,exit_code=code,seconds=seconds))+'\n')
                    if code==0 and (out/'COMPLETE.json').exists():break
                    if attempt==2:raise RuntimeError(f'{name} exhausted3 training attempts')
                    time.sleep(10)
            for budget,count in [(1000000000,64),(2000000000,256)]:
                dest=root/'cfs'/name/f'{budget//1000000000}B';dest.mkdir(parents=True,exist_ok=True)
                if (dest/'COMPLETE.json').exists():continue
                code,seconds=run([sys.executable,'-u','scripts/collect_bert_medium.py','--model',name,
                    '--checkpoint',str(out/f'model_{budget}.pt'),'--data',str(root/'data/cfs_inputs.npz'),
                    '--output',str(dest),'--count',str(count)],root,env,dest/'collect.log')
                if code:raise RuntimeError(f'{name} CFS failed at {budget}; preserved shards can resume')
                atomic(dest/'WALL_COST.json',dict(gpu_hours=seconds/3600))
            print(name,'TRAIN_AND_CFS_COMPLETE',flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(worker,gpu,names) for gpu,names in queues.items()]
            for future in pending:future.result()
        atomic(root/'STATUS.json',dict(stage='analysis',unix=time.time()))
        code,seconds=run([sys.executable,'scripts/analyze_bert_medium.py','--root',str(root)],root,os.environ.copy(),root/'analysis.log')
        if code:raise RuntimeError('Analysis failed; raw data retained')
        atomic(root/'STATUS.json',dict(stage='complete_2B',automatic_extension=False,report=str(root/'analysis/REPORT.md'),unix=time.time()))
    except Exception as exc:
        atomic(root/'STATUS.json',dict(stage='failed',error=repr(exc),unix=time.time()));raise

if __name__=='__main__':main()
