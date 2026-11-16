"""Original BERT input aggregation and final-paper graph definitions; input bootstrap."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.bert_medium import SPECS,AXES,P
from scripts.analyze_matched_scaling import matched,subsets,fixed_metrics
from scripts.analyze_qwen_cfs import minimum_cover
from scripts.analyze_functional_capacity import spectral_effective_rank

def load(path):
    ids=[];drops=[];subs=[]
    for p in sorted(path.glob('shard_*.npz')):
        with np.load(p) as z:ids.extend(z['dataset_index'].tolist());drops.append(z['single_drop'][:,0]);subs.append(z['substitution'][:,0])
    order=np.argsort(ids);ids=np.asarray(ids)[order];drop=np.concatenate(drops)[order].astype(float)
    sub=np.concatenate(subs)[order].astype(float);h=drop.shape[-1]
    assert len(np.unique(ids))==len(ids)
    scores=1-sub/np.maximum(drop[:,:,None],1e-12);scores[:,np.arange(h),np.arange(h)]=np.nan
    assert np.isfinite(scores[:,~np.eye(h,dtype=bool)]).all()
    return ids,scores,drop>1e-5,drop

def summarize(model,path,out,full=True):
    out.mkdir(parents=True,exist_ok=True);ids,scores,valid,drop=load(path);n,h=valid.shape;details=[];mc=[]
    for k,input_id in enumerate(ids):
        score=scores[k];v=valid[k];candidates=score[~np.eye(h,dtype=bool)].reshape(h,h-1)
        values=dict(raw=float(candidates.max(-1)[v].mean()) if v.any() else np.nan,
            cover_full=minimum_cover(score,v,.5)/h,rank_full=spectral_effective_rank(score[None],v[None])[0]/h,
            valid_source_ratio=v.mean(),mean_deletion_kl=drop[k][v].mean() if v.any() else np.nan,
            mean_deletion_kl_all=drop[k].mean())
        for requested in range(1,min(5,h-1)+1):values[f'matched_{requested}']=matched(candidates,requested)[v].mean() if v.any() else np.nan
        if full:
            draws=subsets(h,4,500,np.random.default_rng(np.random.SeedSequence([42,int(input_id),h])))
            metrics=fixed_metrics(score,v,draws)
            values['cover_4'],values['rank_4']=metrics.mean(0)
            np.savez_compressed(out/f'fixedK_input{input_id}.npz',heads=draws,cover=metrics[:,0],rank=metrics[:,1])
            for col,metric in enumerate(['cover_4','rank_4']):
                mc.append(dict(model=model,input_id=int(input_id),metric=metric,draws=len(draws),
                    subgraph_mc_se=float(metrics[:,col].std(ddof=1)/np.sqrt(len(draws))) if len(draws)>1 else 0.,
                    mean100=float(metrics[:min(100,len(draws)),col].mean()),mean250=float(metrics[:min(250,len(draws)),col].mean()),mean500=float(metrics[:,col].mean())))
        details.extend(dict(model=model,input_id=int(input_id),layer=SPECS[model][1]-1,H=h,metric=metric,value=float(value),valid_sources=int(v.sum()),seed=42) for metric,value in values.items())
    frame=pd.DataFrame(details);frame.to_csv(out/'per_input.csv',index=False);pd.DataFrame(mc).to_csv(out/'subgraph_sampling.csv',index=False)
    piv=frame.pivot(index='input_id',columns='metric',values='value');ix=np.random.default_rng(42).integers(n,size=(2000,n))
    rows=[];boots={}
    for metric in piv.columns:
        value=piv[metric].to_numpy();sim=np.nanmean(value[ix],axis=1);boots[metric]=sim
        rows.append(dict(model=model,metric=metric,mean=np.nanmean(value),ci_low=np.nanquantile(sim,.025),ci_high=np.nanquantile(sim,.975),
            input_sd=np.nanstd(value,ddof=1),inputs=n,finite_inputs=int(np.isfinite(value).sum()),bootstrap=2000,seed=42))
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(boots).to_csv(out/'bootstrap.csv',index=False)
    return summary,piv,pd.DataFrame(boots)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'analysis';out.mkdir(exist_ok=True)
    all_summary=[];boots={};points={};performance=[];stability=[];stable_input=[]
    for model in SPECS:
        root=a.root/'runs'/model;assert (root/'COMPLETE.json').exists()
        history=pd.DataFrame(json.loads((root/'history.json').read_text()));history.to_csv(out/f'{model}_training.csv',index=False)
        final=history[history.tokens==2000000000].iloc[-1]
        architecture=json.loads((a.root/'audit/architectures.json').read_text());parameters=next(x['parameters'] for x in architecture if x['model']==model)
        c=json.loads((root/'COMPLETE.json').read_text());attempts=[json.loads(line) for line in (root/'attempts.jsonl').read_text().splitlines()]
        performance.append(dict(model=model,parameters=parameters,validation_ce=final.validation_ce,validation_accuracy=final.validation_accuracy,
            train_ce=final.train_ce,training_tokens=int(c['tokens']),masked_positions=c['mlm_targets'],document_visits=c['document_visits'],
            training_gpu_hours=c['gpu_hours'],reserved_training_attempt_gpu_hours=sum(x['seconds'] for x in attempts)/3600))
        s,piv,b=summarize(model,a.root/'cfs'/model/'2B',out/model);all_summary.append(s);boots[model]=b;points[model]=s.set_index('metric')['mean']
        with np.load(root/'validation_2000000000.npz') as validation:
            vn=len(validation['token_count']);vi=np.random.default_rng(42).integers(vn,size=(2000,vn))
            for metric,key in [('validation_ce','loss_sum'),('validation_accuracy','correct')]:
                draws=validation[key][vi].sum(1)/validation['token_count'][vi].sum(1)
                boots[model][metric]=draws;points[model][metric]=performance[-1][metric]
                performance[-1][metric+'_ci_low']=float(np.quantile(draws,.025));performance[-1][metric+'_ci_high']=float(np.quantile(draws,.975))
        meta1=json.loads((a.root/'cfs'/model/'1B/metadata.json').read_text());meta2=json.loads((a.root/'cfs'/model/'2B/metadata.json').read_text())
        assert meta1['input_identity_sha256']==meta2['input_identity_sha256']
        s1,p1,b1=summarize(model,a.root/'cfs'/model/'1B',out/model/'probe1B',full=False)
        ids1,x1,v1,_=load(a.root/'cfs'/model/'1B');ids2,x2,v2,_=load(a.root/'cfs'/model/'2B')
        index=np.searchsorted(ids2,ids1);assert np.array_equal(ids2[index],ids1)
        p2=piv.loc[ids1,p1.columns];ix=np.random.default_rng(42).integers(len(ids1),size=(2000,len(ids1)))
        for metric in p1:
            change=p2[metric].to_numpy()-p1[metric].to_numpy();draws=np.nanmean(change[ix],1)
            stability.append(dict(model=model,metric=metric,mean_1B=np.nanmean(p1[metric]),mean_2B=np.nanmean(p2[metric]),
                change=np.nanmean(change),ci_low=np.nanquantile(draws,.025),ci_high=np.nanquantile(draws,.975),inputs=64,finite_pairs=int(np.isfinite(change).sum())))
        h=SPECS[model][2]
        for i,j in enumerate(index):
            common=(v1[i]&v2[j])[:,None]&~np.eye(h,dtype=bool);aa=x1[i][common];bb=x2[j][common]
            stable_input.append(dict(model=model,input_id=int(ids1[i]),shared_valid_source_ratio=(v1[i]&v2[j]).mean(),
                matrix_mae=np.abs(aa-bb).mean() if len(aa) else np.nan,
                matrix_spearman=spearmanr(aa,bb).statistic if len(aa)>2 and np.ptp(aa)>0 and np.ptp(bb)>0 else np.nan))
    summary=pd.concat(all_summary);summary.to_csv(out/'summary.csv',index=False);summary.to_json(out/'summary.json',orient='records',indent=2)
    summary.to_latex(out/'summary.tex',index=False,float_format='%.6f')
    table=pd.DataFrame(performance).set_index('model').join(summary.pivot(index='model',columns='metric',values='mean'))
    table.to_csv(out/'main_table.csv');table.to_latex(out/'main_table.tex',float_format='%.6f')
    stab=pd.DataFrame(stability);stab.to_csv(out/'stability_1B_2B.csv',index=False);pd.DataFrame(stable_input).to_csv(out/'matrix_stability_per_input.csv',index=False)
    comparisons=[];trends=[]
    for axis,names in AXES.items():
        pairs=set(zip(names[:-1],names[1:]))|{(n,'B2') for n in names if n!='B2'}
        metrics=['raw','cover_full','rank_full','valid_source_ratio','mean_deletion_kl','cover_4','rank_4']+([f'matched_{i}' for i in [1,2,3]] if axis=='heads' else [f'matched_{i}' for i in [2,4,5]])
        for left,right in sorted(pairs):
            for metric in metrics+['validation_ce','validation_accuracy']:
                change=boots[right][metric]-boots[left][metric]
                comparisons.append(dict(axis=axis,left=left,right=right,metric=metric,change=points[right][metric]-points[left][metric],
                    ci_low=change.quantile(.025),ci_high=change.quantile(.975),bootstrap_unit='independent input',training_seeds=1))
        for metric in metrics+['validation_ce']:
            delta=np.diff(table.loc[names,metric]);direction='increasing' if np.all(delta>=0) else 'decreasing' if np.all(delta<=0) else 'non-monotonic'
            trends.append(dict(axis=axis,metric=metric,direction=direction))
    paired=pd.DataFrame(comparisons);paired.to_csv(out/'paired.csv',index=False);paired.to_latex(out/'paired.tex',index=False,float_format='%.6f')
    trend=pd.DataFrame(trends);trend.to_csv(out/'trends.csv',index=False)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/f'{name}.pdf',bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=200,bbox_inches='tight');plt.close(fig)
    for axis,names in AXES.items():
        x=[SPECS[n][{'width':0,'depth':1,'heads':2}[axis]] for n in names];fig,axs=plt.subplots(1,3,figsize=(7.1,2.5))
        for ax,metrics in zip(axs,[['validation_ce'],['raw','matched_2'],['cover_full','rank_full']]):
            for metric in metrics:
                ax.plot(x,table.loc[names,metric],'o-',label=metric)
                if metric!='validation_ce':
                    rows=summary[summary.metric==metric].set_index('model').loc[names];ax.fill_between(x,rows.ci_low,rows.ci_high,alpha=.15)
            ax.set_xlabel(axis);ax.set_xticks(x);ax.legend(fontsize=6);ax.grid(alpha=.15)
        save(fig,axis+'_scaling')
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.5))
    for col,(axis,names) in enumerate(AXES.items()):
        x=np.arange(3)
        for row,metrics in enumerate([[f'matched_{i}' for i in ([1,2,3] if axis=='heads' else [2,4,5])],['cover_4','rank_4']]):
            for metric in metrics:
                values=summary[summary.metric==metric].set_index('model').loc[names]
                axs[row,col].plot(x,values['mean'],'o-',label=metric);axs[row,col].fill_between(x,values.ci_low,values.ci_high,alpha=.12)
            axs[row,col].set_xticks(x,names);axs[row,col].legend(fontsize=6);axs[row,col].set_title(axis)
    save(fig,'count_controls')
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6))
    for model in SPECS:
        history=pd.read_csv(out/f'{model}_training.csv').dropna(subset=['validation_ce'])
        for ax,metric in zip(axs,['validation_ce','validation_accuracy']):ax.plot(history.tokens/1e9,history[metric],label=model);ax.set_xlabel('Input tokens (billions)');ax.set_ylabel(metric)
    axs[0].legend(ncol=2,fontsize=6);save(fig,'learning_curves')
    text='# BERT-Medium controlled scaling at the prespecified2B checkpoint\n\n'
    text+='All seven models trained from scratch with one common seed and exactly2,000,000,000 nonpadding input positions; B2 is reused across axes. '
    text+='No individual early stopping, favorable-CFS checkpoint selection or3B continuation was performed.\n\n'
    text+=table.to_markdown(floatfmt='.6f')+'\n\n## Q1: cost and learning\n\nActual training hours and validation curves are above; benchmark estimates are in ../benchmark/. '
    text+='GPU-hours in the completion state exclude replayed attempts; reserved_training_attempt_gpu_hours includes them.\n\n'
    text+='## Q2: functional stability and degeneracy\n\n'+stab.to_markdown(index=False,floatfmt='.6f')+'\n\n'
    text+='Valid-source fractions and deletion KL quantify intervention degeneracy. The64-input paired changes assess1B to2B, not an equivalence guarantee. '
    text+='Inspect matrix_stability_per_input.csv as aggregate stability can hide changing pairwise relationships. Low MLM accuracy alone does not prove sufficient task difficulty; slowly improving CE alone does not require unlimited training. '
    text+='Any proposed3B extension must be decided jointly for all seven after these results; it is not started automatically.\n\n'
    text+='## Q3/Q4: observed width, depth and fixed-capacity trends\n\n'+trend.to_markdown(index=False)+'\n\n'
    text+='Paired differences and intervals are in paired.csv. Head decomposition holds parameter tensors/count fixed at initialization, but performance associations do not prove that changing CFS causes improvement.\n\n'
    text+='## Q5 and statistical limits\n\nThe original intervention and MLM output-KL protocol is evaluated in a second bidirectional controlled family. '
    text+='Whether trends agree with GPT-2 is an empirical comparison, not an assumed result; three-point trends cannot establish universal laws. '
    text+='Bootstrap resamples whole independent validation documents2000 times. The original BERT mean-over-input Oracle aggregation is retained, rather than substituting valid-source-weighted global ratios. '
    text+='Rank uses the final-paper clipped affinity and unit self-connections from build_section5_delivery.py; the older BERT-only no-self rank is obsolete. '
    text+='FixedK4 has500 independent subsets per input (H=4 once), with Monte Carlo SE and100/250/500-draw checks saved separately. Input CIs do not capture independent training-seed variation.\n\n'
    for name in ['learning_curves','width_scaling','depth_scaling','heads_scaling','count_controls']:text+=f'![{name}]({name}.png)\n\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    reply='We added seven randomly initialized BERT-style MLM encoders with shared2B-input-token training, separating width, depth, and parameter-identical head decomposition. '
    reply+='We evaluate the same256 held-out documents and fixed masked positions using the original CFS intervention and alpha grid. '
    reply+='The table reports observed directions under matched candidate counts and fixed graph sizes, together with input-paired intervals and1B/2B functional stability. '
    reply+='This provides a controlled bidirectional-encoder test beyond the original causal-decoder family; it does not establish universal scaling or causal performance benefits. '
    reply+='All confidence intervals condition on one common training seed.\n\n'+trend.to_markdown(index=False)+'\n'
    (out/'REBUTTAL_WCQn.md').write_text(reply,encoding='utf-8');(out/'COMPLETE.json').write_text(json.dumps(dict(models=list(SPECS),training_tokens=2000000000,cfs_inputs=256)))

if __name__=='__main__':main()
