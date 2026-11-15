"""Device-event timing and both-task continuous-alpha sensitivity."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.coalitional_cfs import RelationPredictor,grid_oracle,gradient_oracle
from scripts.benchmark_coalitional_cfs import load,pair_initial


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    out=root/'profile';out.mkdir(exist_ok=False);model,engine,_=load(root);timings=[];sensitivity=[]
    # Predetermined train8 timing inputs; matched count excludes source drops/capture.
    for sample in range(8):
        with np.load(root/f'teacher/pool42/train/{sample:05d}.npz') as f:
            engine.cache={k:torch.tensor(f[k],device='cuda') for k in ('heads','residual','reference')}
            pair,joint=f['pair_rel'],f['joint_rel']
        for kind,rel in [('pair',pair),('joint',joint)]:
            grid_oracle(engine,rel) # warm-up
            for repeat in range(3):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();t=time.perf_counter();start.record();_,_,counts=grid_oracle(engine,rel);end.record();torch.cuda.synchronize()
                timings.append(dict(sample=sample,repeat=repeat,kind=kind,labels=len(rel),wall_seconds=time.perf_counter()-t,
                    gpu_event_seconds=start.elapsed_time(end)/1000,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,**counts))
    pd.DataFrame(timings).to_csv(out/'teacher_events.csv',index=False)
    # Continuous refinement sensitivity: same32test inputs, BOTH tasks refined.
    for sample in range(32):
        with np.load(root/f'teacher/pool42/test/{sample:05d}.npz') as f:
            engine.cache={k:torch.tensor(f[k],device='cuda') for k in ('heads','residual','reference')}
            pair,joint=f['pair_rel'],f['joint_rel'];pa=torch.tensor(f['pair_alpha'].reshape(-1,1),device='cuda');pk=torch.tensor(f['pair_kl'].ravel(),device='cuda')
            jk=torch.tensor(f['joint_kl'].ravel(),device='cuda');ja=torch.tensor(f['joint_alpha'].reshape(-1,2),device='cuda');drop=f['drop'];pool=f['pool']
        pv,palpha,_=gradient_oracle(engine,pair,torch.stack((pa,torch.ones_like(pa)),1),120)
        better=pv<pk;pk2=torch.minimum(pk,pv);pa2=pa.clone();pa2[better]=palpha[better]
        initial=torch.cat((pair_initial(pool,pa2),ja[:,None]),1)
        jv,jalpha,_=gradient_oracle(engine,joint,initial,120);jk2=torch.minimum(jk,jv)
        # Include both refined pair feasible solutions explicitly.
        idx=0
        for source in range(12):
            for j in range(4):
                for k in range(j+1,4):jk2[idx]=torch.minimum(jk2[idx],torch.minimum(pk2[source*4+j],pk2[source*4+k]));idx+=1
        valid=drop>1e-5
        r1=1-pk.reshape(12,4).min(-1).values.cpu().numpy()/np.maximum(drop,1e-12)
        r2=1-jk.reshape(12,6).min(-1).values.cpu().numpy()/np.maximum(drop,1e-12)
        r1c=1-pk2.reshape(12,4).min(-1).values.cpu().numpy()/np.maximum(drop,1e-12)
        r2c=1-jk2.reshape(12,6).min(-1).values.cpu().numpy()/np.maximum(drop,1e-12)
        sensitivity.append(dict(sample=sample,pair_grid=float(r1[valid].mean()),joint_grid=float(r2[valid].mean()),pair_continuous=float(r1c[valid].mean()),joint_continuous=float(r2c[valid].mean()),grid_gain=float((r2-r1)[valid].mean()),continuous_gain=float((r2c-r1c)[valid].mean())))
        np.savez_compressed(out/f'refinement_{sample}.npz',pair_kl=pk2.cpu().numpy(),pair_alpha=pa2.cpu().numpy(),joint_kl=jk2.cpu().numpy(),joint_alpha_optimized=jalpha.cpu().numpy(),joint_optimized_kl=jv.cpu().numpy())
    pd.DataFrame(sensitivity).to_csv(out/'continuous_sensitivity.csv',index=False);engine.close()
    # Same fitted seed17 predictors, single relationship vs whole pool, no backbone.
    latency=[]
    for kind,count in [('pair',48),('joint',72)]:
        path=root/f'runs/{kind}_labels_20000_seed17/model.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        predictor=RelationPredictor().cuda().eval();predictor.load_state_dict(payload['state_dict'])
        with np.load(root/'teacher/pool42/train/00000.npz') as f:
            rel=f[f'{kind}_rel'];x=torch.tensor((f['feature'][None]-payload['global_mean'])/payload['global_std'],device='cuda')
        if kind=='pair':rel=np.concatenate((rel,rel[:,-1:]),1)
        rel=torch.tensor(rel,device='cuda')[None]
        for scope,n in [('single_relation',1),('complete_pool',count)]:
            with torch.inference_mode():
                for _ in range(10):predictor(x,rel[:,:n])
                samples=[]
                for _ in range(100):
                    torch.cuda.synchronize();t=time.perf_counter();predictor(x,rel[:,:n]);torch.cuda.synchronize();samples.append(time.perf_counter()-t)
            latency.append(dict(kind=kind,scope=scope,relations=n,parameters=sum(p.numel() for p in predictor.parameters()),weight_bytes=path.stat().st_size,
                median_seconds=float(np.median(samples)),mean_seconds=float(np.mean(samples)),
                dense_linear_flops=2*(2304*256+n*(304*256+256)),flops_scope='linear MAC=2 FLOPs, global encoder shared; excludes activations, embeddings and normalization',includes_backbone=False))
    pd.DataFrame(latency).to_csv(out/'predictor_cost.csv',index=False);print('profiling complete')


if __name__=='__main__':main()
