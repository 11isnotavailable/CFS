"""Measure H12 full132/660 relation spaces; H40 remains combinatorics only."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.benchmark_coalitional_cfs import load
from src.coalitional_cfs import candidate_pool,relations,grid_oracle


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    out=root/'profile/full_relation_space.csv';assert not out.exists();_,engine,_=load(root);rows=[]
    for sample in range(4):
        with np.load(root/f'teacher/pool42/train/{sample:05d}.npz') as f:engine.cache={k:torch.tensor(f[k],device='cuda') for k in ('heads','residual','reference')}
        for kind,rel in [('pair',relations(candidate_pool(width=11))),('joint',relations(candidate_pool(width=11),True))]:
            torch.cuda.synchronize();start=time.perf_counter();event=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);event.record()
            _,_,counts=grid_oracle(engine,rel);end.record();torch.cuda.synchronize()
            rows.append(dict(sample=sample,kind=kind,relations=len(rel),seconds=time.perf_counter()-start,gpu_event_seconds=event.elapsed_time(end)/1000,**counts))
    pd.DataFrame(rows).to_csv(out,index=False);engine.close();print(pd.DataFrame(rows).groupby('kind')[['seconds','gpu_event_seconds']].mean())


if __name__=='__main__':main()
