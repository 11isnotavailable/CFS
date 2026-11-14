"""Routing ablations and prethresholded interaction-conditioned advantages."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scripts.analyze_cfs_interactions import boot,rho


def correlation_interval(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
    draws=np.random.default_rng(42).integers(0,len(a),(2000,len(a)))
    values=np.asarray([rho(a[ix],b[ix]) for ix in draws]);ci=np.nanquantile(values,[.025,.975])
    return dict(n=len(a),rho=rho(a,b),ci_low=ci[0],ci_high=ci[1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;args.output.mkdir(parents=True,exist_ok=False)
    raw=pd.read_csv(root/'routing_per_input.csv');assert not raw.duplicated(['sample','seed','method','keep','compensation']).any()
    averaged=raw.groupby(['sample','method','keep','compensation'])[['kl','correct']].mean().reset_index()
    summary=[];per_seed=[]
    for keys,g in raw.groupby(['method','keep','compensation','seed']):
        for metric in ('kl','correct'):per_seed.append(dict(method=keys[0],keep=keys[1],compensation=keys[2],seed=keys[3],metric=metric,**boot(g[metric])))
    seed_frame=pd.DataFrame(per_seed);seed_frame.to_csv(args.output/'per_seed_summary.csv',index=False)
    for keys,g in averaged.groupby(['method','keep','compensation']):
        method,keep,comp=keys
        for metric in ('kl','correct'):
            sd=seed_frame.query('method==@method and keep==@keep and compensation==@comp and metric==@metric')['mean'].std(ddof=1)
            summary.append(dict(method=method,keep=keep,compensation=comp,metric=metric,seed_mean_sd=sd,**boot(g[metric])))
    summary=pd.DataFrame(summary);summary.to_csv(args.output/'routing_summary.csv',index=False)
    paired=[];seed_paired=[]
    for comp in ('hard','shared_r2_merge'):
        for keep in (3,6,9):
            rows=averaged.query('keep==@keep and compensation==@comp')
            for metric in ('kl','correct'):
                pivot=rows.pivot(index='sample',columns='method',values=metric)
                for method in ('A1','A2','A3'):
                    for reference in ('A0','Gating','R2'):
                        paired.append(dict(method=method,reference=reference,keep=keep,compensation=comp,metric=metric,**boot(pivot[method]-pivot[reference])))
                        for seed in (17,23,41):
                            g=raw.query('seed==@seed and keep==@keep and compensation==@comp').pivot(index='sample',columns='method',values=metric)
                            seed_paired.append(dict(method=method,reference=reference,keep=keep,compensation=comp,metric=metric,seed=seed,**boot(g[method]-g[reference])))
    pd.DataFrame(paired).to_csv(args.output/'paired_routing.csv',index=False)
    pd.DataFrame(seed_paired).to_csv(args.output/'per_seed_paired_routing.csv',index=False)
    full=[]
    for keep in (3,6,9):
        sub=averaged[averaged.keep==keep];hard=sub[sub.compensation=='hard'].pivot(index='sample',columns='method',values='kl')
        merge=sub[sub.compensation=='shared_r2_merge'].pivot(index='sample',columns='method',values='kl')
        for method in ('A0','A1','A2','A3','R2'):
            for name,diff in [('Gating_hard_minus_router_merge',hard.Gating-merge[method]),
                              ('Gating_hard_minus_router_hard',hard.Gating-hard[method]),('router_compensation_gain',hard[method]-merge[method])]:
                full.append(dict(method=method,keep=keep,contrast=name,**boot(diff)))
    pd.DataFrame(full).to_csv(args.output/'pipeline_decomposition.csv',index=False)
    complexity=pd.read_csv(root/'interaction_analysis/input_complexity.csv')
    additive=pd.read_csv(root/'additive_final/additive_per_input.csv').query('source=="new_uniform_masks"')[['split','sample','keep','rmse']]
    complexity=complexity.merge(additive,on=['split','sample','keep'],validate='one_to_one')
    groups=[];correlations=[];thresholds={};conditional_rows=[]
    for keep in (3,6,9):
        threshold=float(complexity.query('split=="tune" and keep==@keep').complexity.median());thresholds[keep]=threshold
        measured=complexity.query('split=="test" and keep==@keep').set_index('sample')
        table=averaged.query('keep==@keep and compensation=="hard"').pivot(index='sample',columns='method',values='kl')
        joined=table.join(measured[['complexity','rmse','cfs_mean','cfs_asymmetry']],how='inner')
        joined['group']=np.where(joined.complexity>threshold,'high','low');joined['keep']=keep
        conditional_rows.append(joined.reset_index())
        for group,g in [('all448',table),('measured128',joined),('low',joined[joined.group=='low']),('high',joined[joined.group=='high'])]:
            for method in ('Gating','A0','R2','A1','A2','A3'):
                groups.append(dict(keep=keep,group=group,method=method,metric='kl',**boot(g[method])))
                if method!='Gating':groups.append(dict(keep=keep,group=group,method=method,metric='gating_minus_router',**boot(g.Gating-g[method])))
        for feature in ('complexity','rmse','cfs_mean','cfs_asymmetry'):
            for method in ('Gating','A0','R2','A1','A2','A3'):
                y=joined.Gating if method=='Gating' else joined.Gating-joined[method]
                correlations.append(dict(keep=keep,feature=feature,outcome='gating_kl' if method=='Gating' else f'gating_minus_{method}',**correlation_interval(joined[feature],y)))
    pd.DataFrame(groups).to_csv(args.output/'conditional_routing.csv',index=False)
    pd.concat(conditional_rows).to_csv(args.output/'conditional_per_input.csv',index=False)
    pd.DataFrame(correlations).to_csv(args.output/'conditional_correlations.csv',index=False)
    (args.output/'thresholds.json').write_text(json.dumps(dict(source='64 tune input medians',thresholds=thresholds),indent=2))
    for filename in ('routing_summary','paired_routing','pipeline_decomposition','conditional_routing'):
        frame=pd.read_csv(args.output/f'{filename}.csv');frame.to_json(args.output/f'{filename}.json',orient='records',indent=2)
        frame.to_latex(args.output/f'{filename}.tex',index=False,float_format='%.6f')
    print(summary[(summary.metric=='kl')&(summary.compensation=='hard')].to_string(index=False))


if __name__=='__main__':main()
