"""Input-cluster statistics for MLP groups; never pool partition seeds."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def corr(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0: return np.nan
    return float(spearmanr(a,b).statistic)


def input_metrics(data):
    s=data['cfs']; drop=data['single_drop']; valid=data['valid']; g=len(drop)
    taylor=data['taylor']; cosine=data['contribution_cosine']; off=~np.eye(g,dtype=bool)
    pairs=valid[:,None]&off
    oracle=np.nanmax(s,axis=1); best=np.nanargmax(s,axis=1)
    t=np.broadcast_to(taylor,(g,g)).copy(); np.fill_diagonal(t,-np.inf)
    c=cosine.copy(); np.fill_diagonal(c,-np.inf)
    tr=t.argmax(1); cr=c.argmax(1); tr_value=s[np.arange(g),tr]; cr_value=s[np.arange(g),cr]
    m={}
    def add(name,values):
        values=np.asarray(values,dtype=float).reshape(-1); values=values[np.isfinite(values)]
        m[name]=(values.sum(),len(values))
    add('raw_oracle',oracle[valid]); add('random_recovery',s[pairs])
    replacement=np.asarray(data['substitution']).copy(); np.fill_diagonal(replacement,np.nan)
    add('oracle_replacement_kl',np.nanmin(replacement,axis=1)[valid])
    add('valid_drop_kl_mean',drop[valid])
    add('taylor_recovery',tr_value[valid]); add('cosine_recovery',cr_value[valid])
    add('oracle_minus_taylor',(oracle-tr_value)[valid]); add('oracle_minus_cosine',(oracle-cr_value)[valid])
    add('oracle_minus_random',(oracle-np.nanmean(s,axis=1))[valid])
    add('taylor_rep_match',(tr==best)[valid]); add('cosine_rep_match',(cr==best)[valid])
    add('pair_recovery_above_half',(s[pairs]>.5)); add('source_oracle_above_half',(oracle[valid]>.5))
    add('pair_negative_fraction',s[pairs]<0)
    for q,label in [(5,'p05'),(50,'median'),(95,'p95')]:
        add('pair_cfs_'+label,[np.percentile(s[pairs],q)] if pairs.any() else [])
    add('valid_source_fraction',valid); add('near_zero_drop_fraction',~valid)
    add('nonfinite_pair_fraction',~np.isfinite(s[off]))
    add('unresolved_valid_source_fraction',valid&(drop<=10*float(data['no_op_kl'])))
    add('no_op_batch_kl',[data['no_op_kl']]); add('drop_kl_mean',drop)
    for q,label in [(5,'p05'),(50,'median'),(95,'p95')]: add('input_drop_kl_'+label,[np.percentile(drop,q)])
    both=valid[:,None]&valid[None,:]&np.triu(np.ones((g,g),bool),1)
    add('directed_asymmetry_abs',np.abs(s-s.T)[both])
    add('directional_half_threshold_disagreement',((s>.5)!=(s.T>.5))[both])
    rt=[]; rc=[]; sources=[]
    for i in range(g):
        candidates=np.arange(g)!=i
        ct=corr(taylor[candidates],s[i,candidates]) if valid[i] else np.nan
        cc=corr(cosine[i,candidates],s[i,candidates]) if valid[i] else np.nan
        if valid[i]: rt.append(ct); rc.append(cc)
        sources.append(dict(source=i,valid=bool(valid[i]),drop_kl=drop[i],taylor=taylor[i],
            raw_oracle=oracle[i],oracle_rep=int(best[i]),taylor_rep=int(tr[i]),cosine_rep=int(cr[i]),
            taylor_recovery=tr_value[i],cosine_recovery=cr_value[i],candidate_taylor_spearman=ct,candidate_cosine_spearman=cc))
    add('candidate_taylor_spearman',rt); add('candidate_cosine_spearman',rc)
    add('source_taylor_drop_spearman',[corr(taylor,drop)])
    return m,sources


def conditional_statistics(scores,valid,indices):
    """Fixed-partition, same-pair across-input variability and representative changes."""
    g=scores.shape[-1]; off=~np.eye(g,dtype=bool)
    mask=valid[:,:,None]&off[None]
    x=np.where(mask,scores,0.)
    best=np.nanargmax(scores,axis=-1)
    choices=np.eye(g)[best]*valid[:,:,None]
    output=[]
    for start in range(0,len(indices),100):
        idx=indices[start:start+100]
        counts=mask[idx].sum(1); sums=x[idx].sum(1); squares=(x[idx]**2).sum(1)
        variance=np.divide(squares-np.divide(sums*sums,counts,out=np.zeros_like(sums),where=counts>0),counts-1,out=np.full_like(sums,np.nan),where=counts>1)
        pair_sd=np.nanmean(np.sqrt(np.maximum(variance,0)),axis=(1,2))
        freq=choices[idx].sum(1); total=freq.sum(-1)
        # This is a between-DISTINCT-input U-statistic. A bootstrap copy of
        # input i must not create an artificial no-switch pair with i itself.
        weights=np.eye(len(scores))[idx].sum(1)
        self_pairs=(weights**2)@valid.astype(float)
        denominator=total**2-self_pairs
        disagreement=np.divide(total**2-(freq**2).sum(-1),denominator,
            out=np.full_like(total,np.nan),where=denominator>0)
        switch=np.nanmean(disagreement,axis=1)
        output.append(np.stack([pair_sd,switch],axis=1))
    return np.concatenate(output)


def plot_all(configs,summary,out):
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    partitions=['contiguous','42','123','2027']; colors=['#0072B2','#E69F00','#009E73','#CC79A7']
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
    for g in [8,16]:
        fig,axes=plt.subplots(2,4,figsize=(7.2,4.0),layout='constrained')
        for row,layer in enumerate([6,11]):
            cfg=configs[f'G{g}_contiguous_L{layer}']
            for col in range(4):
                score=cfg['scores'][col].copy(); score[~cfg['valid'][col]]=np.nan
                im=axes[row,col].imshow(score,vmin=-.1,vmax=1,cmap=cmap,interpolation='nearest')
                axes[row,col].set(title=f"L{layer}, input {cfg['ids'][col]}",xlabel='Representative group',ylabel='Source group')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='CFS (display range −0.1 to 1; gray = invalid/self)',shrink=.8)
        fig.suptitle(f'G={g}: first four preselected inputs, contiguous groups')
        for ext in ['pdf','png']: fig.savefig(out/f'heatmaps_G{g}.{ext}',dpi=300)
        plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(7.2,4),layout='constrained')
        for row,layer in enumerate([6,11]):
            for col,part in enumerate(partitions):
                cfg=configs[f'G{g}_{part}_L{layer}']; scores=np.where(cfg['valid'][:,:,None],cfg['scores'],np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning); mean=np.nanmean(scores,axis=0)
                im=axes[row,col].imshow(mean,vmin=-.1,vmax=1,cmap=cmap)
                axes[row,col].set(title=f'L{layer}, {part}',xlabel='Representative',ylabel='Source')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Mean CFS; group identities differ across partitions',shrink=.8)
        for ext in ['pdf','png']: fig.savefig(out/f'mean_heatmaps_G{g}.{ext}',dpi=300)
        plt.close(fig)
    for name,metrics in [('recovery',['raw_oracle','taylor_recovery','cosine_recovery','random_recovery']),('structure',['pair_recovery_above_half','directed_asymmetry_abs','best_rep_input_switch_probability'])]:
        fig,axes=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained')
        for row,layer in enumerate([6,11]):
            for col,g in enumerate([8,16]):
                ax=axes[row,col]
                for k,metric in enumerate(metrics):
                    vals=summary[(summary.layer==layer)&(summary.G==g)&(summary.metric==metric)].set_index('partition').reindex(partitions)
                    x=np.arange(4)+(k-(len(metrics)-1)/2)*.06
                    ax.errorbar(x,vals['mean'],yerr=np.maximum(0,np.stack([vals['mean']-vals.ci_low,vals.ci_high-vals['mean']])),fmt='o-',markersize=3,capsize=2,label=metric.replace('_',' '),color=colors[k])
                ax.set(title=f'Layer {layer}, G={g}',xticks=np.arange(4),xticklabels=partitions,ylabel='Mean and input-bootstrap 95% CI')
                ax.legend(fontsize=6,frameon=False); ax.grid(alpha=.2)
        for ext in ['pdf','png']: fig.savefig(out/f'{name}_partition_comparison.{ext}',dpi=300)
        plt.close(fig)
    for kind in ['cfs','drop']:
        fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),layout='constrained')
        for row,layer in enumerate([6,11]):
            for col,g in enumerate([8,16]):
                ax=axes[row,col]
                for part,color in zip(partitions,colors):
                    cfg=configs[f'G{g}_{part}_L{layer}']
                    values=cfg['scores'][cfg['valid'][:,:,None]&~np.eye(g,dtype=bool)[None]] if kind=='cfs' else cfg['drop'].ravel()
                    values=np.sort(values[np.isfinite(values)])
                    ax.plot(values,np.arange(1,len(values)+1)/len(values),label=part,color=color)
                if kind=='drop': ax.set_xscale('symlog',linthresh=1e-6)
                ax.set(title=f'Layer {layer}, G={g}',xlabel='CFS' if kind=='cfs' else 'Deletion KL',ylabel='Empirical CDF (descriptive)'); ax.legend(fontsize=6,frameon=False)
        for ext in ['pdf','png']: fig.savefig(out/f'{kind}_distribution.{ext}',dpi=300)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    if not (args.input/'COMPLETE.json').exists(): raise ValueError('Collection incomplete')
    args.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((args.input/'metadata.json').read_text()); n=meta['samples']
    indices=np.random.default_rng(args.seed).integers(n,size=(args.bootstrap,n))
    all_rows=[]; summaries=[]; source_rows=[]; boot_rows=[]; configs={}; instability=[]
    for directory in sorted(args.input.glob('G*_L*')):
        key=directory.name; g=int(key.split('_')[0][1:]); part=key.split('_')[1]; layer=int(key.split('_')[-1][1:])
        data=[dict(np.load(p,allow_pickle=False)) for p in sorted(directory.glob('input_*.npz'))]
        assert len(data)==n
        ids=[int(d['input_id']) for d in data]; assert ids==meta['sample_indices']
        scores=np.stack([d['cfs'] for d in data]); valid=np.stack([d['valid'] for d in data]); drops=np.stack([d['single_drop'] for d in data])
        configs[key]=dict(scores=scores,valid=valid,drop=drops,ids=ids)
        inputs=[]
        for d in data:
            metrics,sources=input_metrics(d); inputs.append(metrics)
            identity=dict(config=key,G=g,partition=part,layer=layer,input_id=int(d['input_id']))
            source_rows += [identity|row for row in sources]
            for metric,(num,den) in metrics.items(): all_rows.append(identity|dict(metric=metric,numerator=num,denominator=den,value=num/den if den else np.nan))
        config_boot={}; config_point={}
        def record(metric,point,boot,sd):
            low,high=np.nanquantile(boot,[.025,.975]); config_boot[metric]=boot; config_point[metric]=point
            summaries.append(dict(config=key,G=g,partition=part,layer=layer,metric=metric,mean=point,ci_low=low,ci_high=high,input_sd=sd,inputs=n,bootstrap=args.bootstrap,seed=args.seed))
            boot_rows.extend(dict(config=key,metric=metric,replicate=i,value=v) for i,v in enumerate(boot))
        for metric in inputs[0]:
            num=np.array([x[metric][0] for x in inputs]); den=np.array([x[metric][1] for x in inputs])
            denom=den[indices].sum(1); boot=np.divide(num[indices].sum(1),denom,out=np.full(args.bootstrap,np.nan),where=denom>0)
            values=np.divide(num,den,out=np.full(n,np.nan),where=den>0)
            record(metric,num.sum()/den.sum() if den.sum() else np.nan,boot,np.nanstd(values,ddof=1))
        point=conditional_statistics(scores,valid,np.arange(n)[None])[0]
        boot=conditional_statistics(scores,valid,indices)
        for j,metric in enumerate(['same_pair_across_input_sd','best_rep_input_switch_probability']): record(metric,point[j],boot[:,j],np.nan)
        configs[key]['boot']=config_boot; configs[key]['point']=config_point
        print('ANALYZED',key,flush=True)
    summary=pd.DataFrame(summaries); summary.to_csv(args.output/'summary.csv',index=False); summary.to_json(args.output/'summary.json',orient='records',indent=2)
    summary.to_latex(args.output/'summary.tex',index=False,float_format='%.6f')
    pd.DataFrame(all_rows).to_csv(args.output/'per_input.csv',index=False); pd.DataFrame(source_rows).to_csv(args.output/'per_source.csv',index=False)
    pd.DataFrame(boot_rows).to_csv(args.output/'bootstrap.csv',index=False)
    for key,cfg in configs.items():
        if '_contiguous_' in key: continue
        g=key.split('_')[0]; layer=key.split('_')[-1]; baseline=configs[f'{g}_contiguous_{layer}']
        for metric in cfg['boot']:
            diff=cfg['boot'][metric]-baseline['boot'][metric]; low,high=np.nanquantile(diff,[.025,.975])
            instability.append(dict(config=key,reference=f'{g}_contiguous_{layer}',metric=metric,delta=cfg['point'][metric]-baseline['point'][metric],ci_low=low,ci_high=high))
    pd.DataFrame(instability).to_csv(args.output/'partition_contrasts.csv',index=False)
    plot_all(configs,summary,args.output)
    (args.output/'analysis_metadata.json').write_text(json.dumps(dict(input=str(args.input),bootstrap=args.bootstrap,seed=args.seed,cohort=meta['sample_indices'],aggregation='within-config source/pair ratios; resample whole images; never pool partition seeds',conditional_statistics='within fixed partition: sample SD of each directed pair across valid inputs; weighted U-statistic for best-representative disagreement, excluding self-pairs between bootstrap copies of the same original image',no_op_caution='10x measured no-op KL is a descriptive numerical sensitivity flag, not a rigorous error bound'),indent=2))
    print('ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__': main()
