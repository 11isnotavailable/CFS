"""Exact common-grid pair and joint teachers, cached/resumable per input."""
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.benchmark_coalitional_cfs import load,dataset
from src.coalitional_cfs import candidate_pool,relations,grid_oracle


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',choices=['train','tune','test'],required=True)
    p.add_argument('--pool-seed',type=int,default=42);a=p.parse_args();root=a.root
    if a.split=='test':assert (root/'selection_lock.json').exists(),'Lock every trained predictor before test teacher'
    out=root/f'teacher/pool{a.pool_seed}/{a.split}';out.mkdir(parents=True,exist_ok=True)
    model,engine,transform=load(root);data,ds=dataset(root,a.split,transform);pool=candidate_pool(a.pool_seed);pr=relations(pool);jr=relations(pool,True)
    config=dict(pool_seed=a.pool_seed,split=a.split,alpha_steps=31,alpha_bounds=[0,3],dtype='float32',valid_drop=1e-5,
        input_sha256=hashlib.sha256((root/f'data/{a.split}.npz').read_bytes()).hexdigest())
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:(out/'config.json').write_text(json.dumps(config,indent=2))
    begin=time.perf_counter()
    for i in range(len(ds)):
        path=out/f'{i:05d}.npz'
        if path.exists():
            with np.load(path) as f:assert str(f['sha256'])==data['sha256'][i]
            continue
        image=ds[i][0][None].cuda();torch.cuda.synchronize();t=time.perf_counter();ref,feature=engine.capture(image);torch.cuda.synchronize();capture=time.perf_counter()-t
        t=time.perf_counter();drops=engine.losses(1-torch.eye(12,device='cuda'));torch.cuda.synchronize();drop_time=time.perf_counter()-t
        t=time.perf_counter();pk,pa,pc=grid_oracle(engine,pr);torch.cuda.synchronize();pair_time=time.perf_counter()-t
        # Retain the deletion-only feasible observation, alpha=1.
        improve=drops[torch.tensor(pr[:,0],device='cuda')]<pk
        pk[improve]=drops[torch.tensor(pr[:,0],device='cuda')][improve];pa[improve]=1
        t=time.perf_counter();jk,ja,jc=grid_oracle(engine,jr);torch.cuda.synchronize();joint_time=time.perf_counter()-t
        idx=0;fallback=0
        for source,row in enumerate(pool):
            for j in range(4):
                for k in range(j+1,4):
                    for local,dimension in ((j,0),(k,1)):
                        at=source*4+local
                        if pk[at]<jk[idx]:jk[idx]=pk[at];ja[idx]=1;ja[idx,dimension]=pa[at,0];fallback+=1
                    idx+=1
        for source in range(12):assert jk.reshape(12,6)[source].min()<=pk.reshape(12,4)[source].min()+1e-10
        record=dict(sample=i,path=data['paths'][i],sha256=data['sha256'][i],feature=feature[0].cpu().numpy(),drop=drops.cpu().numpy(),
            pair_kl=pk.reshape(12,4).cpu().numpy(),pair_alpha=pa.reshape(12,4).cpu().numpy(),joint_kl=jk.reshape(12,6).cpu().numpy(),joint_alpha=ja.reshape(12,6,2).cpu().numpy(),
            pair_rel=pr,joint_rel=jr,pool=pool,capture_seconds=capture,drop_seconds=drop_time,pair_seconds=pair_time,joint_seconds=joint_time,
            pair_variants=pc['variants'],joint_variants=jc['variants'],pair_forward_calls=pc['forward_calls'],joint_forward_calls=jc['forward_calls'],
            backward_calls=0,feasible_fallbacks=fallback,**{key:value.cpu().numpy() for key,value in engine.cache.items()})
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**record);tmp.replace(path)
        if (i+1)%64==0:print('collected',a.split,a.pool_seed,i+1,'seconds',time.perf_counter()-begin,flush=True)
    engine.close();(out/'COMPLETE.json').write_text(json.dumps(dict(inputs=len(ds),seconds=time.perf_counter()-begin)))


if __name__=='__main__':main()
