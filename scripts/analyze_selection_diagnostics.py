"""Input bootstrap and diagnostic upper bounds; never selects a new router."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    data=root/'collected';out=root/'analysis';out.mkdir(exist_ok=False)
    raw=pd.read_csv(data/'per_input.csv');coverage=pd.read_csv(data/'coverage.csv');cross=pd.read_csv(data/'cross_method.csv')
    grouped=raw.groupby(['split','sample','method','keep']).mean(numeric_only=True).reset_index()
    cov=coverage.groupby(['split','sample','method','keep']).mean(numeric_only=True).reset_index()
    cr=cross.groupby(['split','sample','keep']).mean(numeric_only=True).reset_index()
    cache={};draw_archive={}
    def weights(n):
        if n not in cache:
            draws=np.random.default_rng(42).integers(0,n,(2000,n),dtype=np.int32);draw_archive[f'n{n}']=draws
            cache[n]=np.stack([np.bincount(v,minlength=n) for v in draws]).astype(np.float64)
        return cache[n]
    def stats(matrix):
        x=np.asarray(matrix,float)
        if x.ndim==1:x=x[:,None]
        valid=np.isfinite(x);counts=weights(len(x));den=counts@valid.astype(float)
        boot=np.divide(counts@np.where(valid,x,0),den,out=np.full_like(den,np.nan),where=den>0)
        lo,hi=np.nanquantile(boot,[.025,.975],axis=0)
        return [dict(n_inputs=len(x),n_valid=int(valid[:,i].sum()),mean=float(np.nanmean(x[:,i])),sd_inputs=float(np.nanstd(x[:,i],ddof=1)),ci_low=float(lo[i]),ci_high=float(hi[i])) for i in range(x.shape[1])]
    def ratio(num,den):
        a=np.asarray(num,float);b=np.asarray(den,float);w=weights(len(a));d=w@b
        values=np.divide(w@a,d,out=np.full(len(w),np.nan),where=d>1e-15);lo,hi=np.nanquantile(values,[.025,.975])
        return dict(n_inputs=len(a),mean=float(a.sum()/b.sum()) if b.sum()>1e-15 else np.nan,ci_low=float(lo),ci_high=float(hi))
    cols=[c for c in raw.columns if c not in ('split','seed','method','keep','sample','chosen')]
    seed_summary=raw.groupby(['split','method','keep','seed'])[cols].mean().reset_index();seed_summary.to_csv(out/'per_seed_summary.csv',index=False)
    rows=[];ratios=[]
    for (split,method,k),g in grouped.groupby(['split','method','keep']):
        g=g.sort_values('sample');summaries=stats(g[cols])
        sd=seed_summary.query('split==@split and method==@method and keep==@k')[cols].std(ddof=1)
        for metric,s in zip(cols,summaries):rows.append(dict(split=split,method=method,keep=k,metric=metric,seed_mean_sd=float(sd[metric]),**s))
        for size in (1,3,5,10,20):
            ratios.append(dict(split=split,method=method,keep=k,metric=f'top{size}_regret_removable_fraction',**ratio(g.regret-g[f'top{size}_regret'],g.regret)))
        for label,column in [('near_regret_share','near_regret'),('far_regret_share','far_regret')]:
            ratios.append(dict(split=split,method=method,keep=k,metric=label,**ratio(g[column],g.regret)))
    summary=pd.DataFrame(rows);bounds=pd.DataFrame(ratios);paired=[]
    for (split,k),g in grouped.groupby(['split','keep']):
        for metric in ('chosen_kl','regret','spearman','top5_pair_accuracy','top5_regret','top5_retrieval'):
            pivot=g.pivot(index='sample',columns='method',values=metric)
            for left,right in [('A3','R2'),('A3','Gating'),('A3','A0'),('R2','Gating')]:
                paired.append(dict(split=split,keep=k,metric=metric,left=left,right=right,**stats(pivot[left]-pivot[right])[0]))
    coverage_summary=[]
    for (split,method,k),g in cov.groupby(['split','method','keep']):
        for metric in ('full_choice_known','score_gap_standardized'):
            coverage_summary.append(dict(split=split,method=method,keep=k,metric=metric,**stats(g.sort_values('sample')[metric])[0]))
    cross_summary=[];union_bounds=[];conditional=[]
    cross_cols=[c for c in cross.columns if c not in ('split','seed','sample','keep')]
    for (split,k),g in cr.groupby(['split','keep']):
        g=g.sort_values('sample')
        for metric,s in zip(cross_cols,stats(g[cross_cols])):cross_summary.append(dict(split=split,keep=k,metric=metric,**s))
        for method in ('A3','R2','Gating'):
            for size in (1,3,5):
                delta=g[f'{method}_kl']-g[f'union_top{size}_kl']
                union_bounds.append(dict(split=split,keep=k,method=method,shortlist_per_model=size,
                    quantity='oracle_KL_reduction',**stats(delta)[0]))
                union_bounds.append(dict(split=split,keep=k,method=method,shortlist_per_model=size,
                    quantity='regret_removable_fraction',**ratio(delta,g[f'{method}_kl']-g.oracle_kl)))
    # Condition on same-seed disagreement, then average within each independent input.
    for (split,k),g in cross.groupby(['split','keep']):
        for group in (0,1):
            h=g[g.A3_R2_pool_disagree==group].groupby('sample').mean(numeric_only=True)
            if len(h)<2:continue
            delta=h.A3_kl-h.R2_kl
            conditional.append(dict(split=split,keep=k,disagree=group,metric='A3_minus_R2_KL',**stats(delta)[0]))
    tables=dict(diagnostic_summary=summary,paired_methods=pd.DataFrame(paired),shortlist_bounds=bounds,
        coverage_summary=pd.DataFrame(coverage_summary),disagreement_summary=pd.DataFrame(cross_summary),
        union_bounds=pd.DataFrame(union_bounds),disagreement_conditional=pd.DataFrame(conditional))
    for name,t in tables.items():
        t.to_csv(out/f'{name}.csv',index=False);t.to_json(out/f'{name}.json',orient='records',indent=2)
        t.to_latex(out/f'{name}.tex',index=False,float_format='%.7f')
    # Candidate queue is prediction-defined. It contains no new interventions.
    inventory=pd.read_csv(data/'training_candidate_inventory.csv')
    priority=cr[cr.split=='train'][['sample','keep','A3_R2_full_swaps']]
    selected=priority.sort_values(['keep','A3_R2_full_swaps','sample'],ascending=[True,False,True]).groupby('keep').head(64)
    queue=inventory.merge(selected,on=['sample','keep'],validate='many_to_one')
    queue=queue[(queue.origin=='full') & ~queue.label_known].sort_values(['keep','A3_R2_full_swaps','sample'],ascending=[True,False,True])
    queue=queue.drop_duplicates(['sample','keep','mask_code'])
    queue.to_csv(out/'unmeasured_training_candidate_queue.csv',index=False)
    np.savez_compressed(out/'bootstrap_draws.npz',**draw_archive)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'A3':'#D55E00','R2':'#000000','Gating':'#0072B2','A0':'#009E73'}
    fig,axes=plt.subplots(1,2,figsize=(7,2.7))
    for ax,split in zip(axes,('train','tune')):
        for offset,(method,color) in enumerate(colors.items()):
            r=summary.query('split==@split and method==@method and metric=="regret"').sort_values('keep')
            ax.errorbar(r.keep+(offset-1.5)*.1,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o',capsize=2,ms=3,label=method,color=color)
        ax.set(title=f'{split}: same 48 labelled candidates',xlabel='Retained heads K',ylabel='Selection regret (KL)',xticks=[3,6,9])
    axes[0].legend(frameon=False,fontsize=6);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'figure1_pool_regret.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(7,4.2))
    for row,split in enumerate(('train','tune')):
        for col,k in enumerate((3,6,9)):
            ax=axes[row,col]
            for method,color in colors.items():
                values=[]
                for size in (1,3,5,10,20):
                    metric=f'top{size}_regret_removable_fraction';v=bounds.query('split==@split and keep==@k and method==@method and metric==@metric').iloc[0];values.append(v['mean'])
                ax.plot([1,3,5,10,20],values,'o-',ms=2,label=method,color=color)
            ax.set(title=f'{split}, K={k}',xlabel='Predicted shortlist size',ylabel='Oracle-removable regret',ylim=(0,1.02))
    axes[0,0].legend(frameon=False,fontsize=5);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'figure2_shortlist_upper_bounds.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    compact=[]
    for (split,method,k),g in summary.groupby(['split','method','keep']):
        row=dict(split=split,method=method,keep=k)
        for metric in ('chosen_kl','oracle_kl','regret','exact_best','chosen_true_rank','spearman','pair_accuracy','top5_pair_accuracy','top5_retrieval'):
            row[metric]=float(g[g.metric==metric]['mean'].iloc[0])
        for metric in ('top5_regret_removable_fraction','far_regret_share'):
            row[metric]=float(bounds.query('split==@split and method==@method and keep==@k and metric==@metric')['mean'].iloc[0])
        row['full_choice_known']=float(tables['coverage_summary'].query('split==@split and method==@method and keep==@k and metric=="full_choice_known"')['mean'].iloc[0])
        compact.append(row)
    compact=pd.DataFrame(compact);compact.to_csv(out/'compact_summary.csv',index=False)
    result=dict(inputs={'train':4096,'tune':64},methods=list(colors),training=False,new_labels=False,
        queue_candidates=len(queue),queue_inputs=int(queue['sample'].nunique()),bootstrap=2000,
        source='existing labelled pool; no global-KL inference for unlabelled candidates')
    (out/'analysis_metadata.json').write_text(json.dumps(result,indent=2))
    print(compact.to_string(index=False));print('UNMEASURED QUEUE',len(queue))


if __name__=='__main__':main()
