"""Additional descriptive checks; no model fitting, selection or threshold tuning."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.analyze_cfs_interactions import boot


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;args.output.mkdir(parents=True,exist_ok=False)
    pair=pd.read_csv(root/'interaction_analysis/pair_interactions.csv');pair=pair[pair.split=='test']
    rows=[]
    for keys,g in pair.groupby(['keep','pair','i','j']):
        for metric in ('interaction','magnitude'):
            rows.append(dict(keep=keys[0],pair=keys[1],i=keys[2],j=keys[3],metric=metric,**boot(g[metric])))
    pd.DataFrame(rows).to_csv(args.output/'all_head_pair_statistics.csv',index=False)
    raw=pd.read_csv(root/'routing_per_input.csv');average=raw.groupby(['sample','method','keep','compensation']).kl.mean().reset_index()
    ablation=[]
    for comp in ('hard','shared_r2_merge'):
        for keep in (3,6,9):
            pivot=average.query('keep==@keep and compensation==@comp').pivot(index='sample',columns='method',values='kl')
            for left,right in [('A3','A1'),('A2','A1'),('A3','A2')]:
                ablation.append(dict(keep=keep,compensation=comp,contrast=f'{left}_minus_{right}',**boot(pivot[left]-pivot[right])))
    pd.DataFrame(ablation).to_csv(args.output/'direct_auxiliary_ablation.csv',index=False)
    conditional=pd.read_csv(root/'routing_analysis/conditional_per_input.csv');contrasts=[]
    for keep in (3,6,9):
        group=conditional[conditional.keep==keep];hi=group[group.group=='high'];lo=group[group.group=='low']
        for method in ('A0','R2','A1','A2','A3'):
            high=(hi.Gating-hi[method]).to_numpy();low=(lo.Gating-lo[method]).to_numpy();rng=np.random.default_rng(42)
            values=high[rng.integers(0,len(high),(2000,len(high)))].mean(1)-low[rng.integers(0,len(low),(2000,len(low)))].mean(1)
            ci=np.quantile(values,[.025,.975]);contrasts.append(dict(keep=keep,method=method,n_high=len(high),n_low=len(low),
                high_minus_low_gain=high.mean()-low.mean(),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(contrasts).to_csv(args.output/'high_minus_low_advantage.csv',index=False)
    checks=json.loads((root/'interactions/checks.json').read_text());noise=[]
    for row in checks:
        a=np.load(root/f'interactions/{row["split"]}_{row["sample"]:04d}.npz')
        noise.append(dict(split=row['split'],sample=row['sample'],noop_kl=row['noop_kl'],
            fraction_absI_below_four_noop=float((abs(a['interaction'])<=4*row['noop_kl']).mean())))
    pd.DataFrame(noise).to_csv(args.output/'numerical_noise_diagnostic.csv',index=False)
    predictions=pd.read_csv(root/'interaction_analysis/heldout_pair_predictions.csv').query('target=="interaction"')
    sign=[]
    for keys,g in predictions.groupby(['method','keep','sample']):
        positive=g.actual>=0;negative=~positive;estimate=g.predicted>=0
        balanced=.5*((estimate[positive]).mean()+(~estimate[negative]).mean()) if positive.any() and negative.any() else np.nan
        sign.append(dict(method=keys[0],keep=keys[1],sample=keys[2],accuracy=(estimate==positive).mean(),
            always_positive_accuracy=positive.mean(),balanced_accuracy=balanced))
    signs=pd.DataFrame(sign);signs.to_csv(args.output/'sign_prediction_per_input.csv',index=False)
    results=[]
    for keys,g in signs.groupby(['method','keep']):
        for metric in ('accuracy','always_positive_accuracy','balanced_accuracy'):
            results.append(dict(method=keys[0],keep=keys[1],metric=metric,**boot(g[metric])))
    pd.DataFrame(results).to_csv(args.output/'sign_prediction_summary.csv',index=False)
    for path in args.output.glob('*.csv'):
        if 'per_input' not in path.name:
            frame=pd.read_csv(path);frame.to_json(path.with_suffix('.json'),orient='records',indent=2)


if __name__=='__main__':main()
