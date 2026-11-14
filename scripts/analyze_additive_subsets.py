"""Offline held-out mask cross-validation of per-input additive KL models."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.subset_interactions import additive_crossfit


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--measurements',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for split in ('train','tune'):
        # Materialize compressed arrays once; repeated NpzFile indexing decompresses
        # the complete training cache on every sample/budget iteration.
        data=dict(np.load(args.root/f'baseline/learned_gating_inputs_20261115/{split}.npz'))
        predictions=np.empty_like(data['kl']);folds=np.empty_like(data['kl'],dtype=np.int8)
        for i in range(len(data['feature'])):
            for b,k in enumerate((3,6,9)):
                stat,pred,fold=additive_crossfit(data['masks'][i,b],data['kl'][i,b])
                rows.append(dict(split=split,source='cached_anchor_pool',sample=i,keep=k,**stat))
                predictions[i,b]=pred;folds[i,b]=fold
            if (i+1)%1024==0:print(f'crossfit {split} {i+1}/{len(data["feature"])}',flush=True)
        np.savez_compressed(args.output/f'{split}_mask_predictions.npz',prediction=predictions,fold=folds,dataset_index=data['dataset_index'])
    if args.measurements:
        for path in sorted(args.measurements.glob('*.npz')):
            a=np.load(path);split=str(a['split']);sample=int(a['sample'])
            for b,k in enumerate((3,6,9)):
                ids=a['random_ids'][b];stat,pred,fold=additive_crossfit(a['masks'][ids],a['kl'][ids])
                rows.append(dict(split=split,source='new_uniform_masks',sample=sample,keep=k,**stat))
    frame=pd.DataFrame(rows);frame.to_csv(args.output/'additive_per_input.csv',index=False)
    summaries=[];rng=np.random.default_rng(42)
    for keys,g in frame.groupby(['split','source','keep']):
        draws=rng.integers(0,len(g),(2000,len(g)))
        for metric in ('rmse','mae','train_rmse','constant_rmse','residual_sd'):
            x=g[metric].to_numpy();ci=np.quantile(x[draws].mean(1),[.025,.975])
            summaries.append(dict(split=keys[0],source=keys[1],keep=keys[2],metric=metric,n=len(g),mean=x.mean(),sd=x.std(ddof=1),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(summaries).to_csv(args.output/'additive_summary.csv',index=False)
    print(frame.groupby(['split','source','keep'])[['rmse','train_rmse','constant_rmse','rank_min']].mean().to_string())


if __name__=='__main__':main()
