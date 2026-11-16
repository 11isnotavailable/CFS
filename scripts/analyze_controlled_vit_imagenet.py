"""Offline primary-layer controls, paired CIs, and publication figures."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analyze_matched_scaling import analyze
from src.controlled_vit_imagenet import AXES, SPECS


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.root/'cfs_manifest.json').read_text())
    entries=manifest['models']; summaries=[]; boots={}; performances=[]; details=[]
    for e in entries:
        name=e['name']; out=a.output/name; out.mkdir(exist_ok=True)
        # Completed analysis markers are immutable and avoid repeating subgraph draws.
        if not (out/'ANALYSIS_COMPLETE.json').exists():
            analyze(e,a.root,out,500,2000,42)
            (out/'ANALYSIS_COMPLETE.json').write_text(json.dumps(dict(model=name)))
        summaries.append(pd.read_csv(out/'summary.csv'))
        b=pd.read_csv(out/'bootstrap.csv').pivot(index='replicate',columns='metric',values='value')
        samples=[]; ids=[]
        for f in sorted((a.root/e['path']).glob('shard_*.npz')):
            with np.load(f) as z: samples.append(z['single_drop'][:,0]); ids.extend(z['dataset_index'].tolist())
        drops=np.concatenate(samples)[np.argsort(ids)].astype(float); indices=np.random.default_rng(42).integers(512,size=(2000,512))
        extra=[]
        for metric,v in [('valid_source_ratio',(drops>1e-5).mean(-1)),('mean_deletion_kl',drops.mean(-1))]:
            draws=v[indices].mean(-1); b[metric]=draws
            extra.append(dict(family='ViT',model=name,parameters=e['parameters'],H=e['heads'],metric=metric,
                mean=v.mean(),ci_low=np.quantile(draws,.025),ci_high=np.quantile(draws,.975),input_sd=v.std(ddof=1)))
            details.extend(dict(model=name,input_id=int(i),metric=metric,value=float(x)) for i,x in zip(sorted(ids),v))
        summaries.append(pd.DataFrame(extra)); boots[name]=b
        perf=json.loads((a.root/'runs'/name/'COMPLETE.json').read_text())
        history=pd.DataFrame(json.loads((a.root/'runs'/name/'history.json').read_text()))
        selected_epoch=perf['selected_epoch']; stopped_epoch=perf['stopped_epoch']
        assert selected_epoch==stopped_epoch==120
        history.to_csv(a.output/f'{name}_training.csv',index=False)
        last=history.iloc[-20:]; slope=np.polyfit(last.epoch,last.val_ce,1)[0]
        attempts=[json.loads(line) for line in (a.root/'runs'/name/'attempts.jsonl').read_text().splitlines()]
        wall_hours=sum(r['seconds'] for r in attempts if not r.get('extension'))/3600
        performances.append(dict(model=name,parameters=e['parameters'],width=SPECS[name][0],depth=SPECS[name][1],
            heads=SPECS[name][2],head_dim=SPECS[name][0]//SPECS[name][2],val_ce=perf['val_ce'],top1=perf['top1'],
            selected_epoch=selected_epoch,stopped_epoch=stopped_epoch,evaluation_epoch=120,
            gpu_hours=wall_hours,epoch_train_validation_gpu_hours=perf['gpu_hours'],last20_ce_slope=slope,
            last20_relative_ce_improvement=(last.iloc[0].val_ce-last.iloc[-1].val_ce)/last.iloc[0].val_ce))
    summary=pd.concat(summaries,ignore_index=True)
    performance=pd.DataFrame(performances).set_index('model')
    table=performance.join(summary.pivot(index='model',columns='metric',values='mean'))
    table['delta_params_vs_V2']=table.parameters-table.loc['V2','parameters']
    table['delta_ce_vs_V2']=table.val_ce-table.loc['V2','val_ce']
    table['ce_improvement_per_added_million']=np.where(table.delta_params_vs_V2>0,
        -table.delta_ce_vs_V2/(table.delta_params_vs_V2/1e6),np.nan)
    table.reset_index().to_csv(a.output/'main_table.csv',index=False)
    table.reset_index().to_latex(a.output/'main_table.tex',index=False,float_format='%.6f')
    summary.to_csv(a.output/'summary.csv',index=False); summary.to_json(a.output/'summary.json',orient='records',indent=2)
    pd.DataFrame(details).to_csv(a.output/'validity_deletion_per_input.csv',index=False)
    comparisons=[]
    for axis,names in AXES.items():
        pairs=list(zip(names[:-1],names[1:]))+[(n,'V2') for n in names if n!='V2']
        allowed={'raw','matched_1','cover_full','rank_full','valid_source_ratio','mean_deletion_kl'}
        allowed|={'cover_2','rank_2'} if axis=='heads' else {'cover_3','rank_3','matched_2'}
        if axis=='depth': allowed.add('matched_3')
        for left,right in pairs:
            for metric in sorted(set(boots[left])&set(boots[right])&allowed):
                difference=boots[right][metric]-boots[left][metric]
                comparisons.append(dict(axis=axis,contrast=f'{right}_minus_{left}',metric=metric,
                    mean=table.loc[right,metric]-table.loc[left,metric],ci_low=difference.quantile(.025),
                    ci_high=difference.quantile(.975),training_seeds=1,bootstrap_unit='input'))
    paired=pd.DataFrame(comparisons); paired.to_csv(a.output/'paired.csv',index=False)
    paired.to_latex(a.output/'paired.tex',index=False,float_format='%.6f')
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def panel(ax,axis,metrics):
        names=AXES[axis]; x=[SPECS[n][{'width':0,'depth':1,'heads':2}[axis]] for n in names]
        for metric in metrics:
            y=table.loc[names,metric].to_numpy()
            if metric=='val_ce': ax.plot(x,y,'o-',label='Validation CE')
            else:
                rows=summary[summary.metric==metric].set_index('model').loc[names]
                ax.plot(x,y,'o-',label=metric.replace('_',' ')); ax.fill_between(x,rows.ci_low,rows.ci_high,alpha=.15)
        ax.set_xticks(x); ax.set_xlabel(axis.capitalize()); ax.legend(frameon=False,fontsize=7); ax.grid(alpha=.15)
    def save(fig,name):
        fig.tight_layout(); fig.savefig(a.output/f'{name}.pdf',bbox_inches='tight'); fig.savefig(a.output/f'{name}.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(6.8,4.5))
    for col,axis in enumerate(['width','depth']):
        panel(axs[0,col],axis,['val_ce']); panel(axs[1,col],axis,['raw','matched_2']); axs[0,col].set_title(axis.capitalize()+' scaling')
    save(fig,'figure1_controlled_scaling')
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.5))
    for ax,axis in zip(axs,['width','depth']): panel(ax,axis,['cover_full','rank_full']); ax.set_title(axis.capitalize())
    save(fig,'figure2_functional_organization')
    fig,axs=plt.subplots(1,3,figsize=(6.8,2.4))
    for ax,metrics in zip(axs,[['val_ce'],['raw'],['rank_full']]): panel(ax,'heads',metrics)
    save(fig,'figure3_fixed_capacity')
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.5))
    for col,axis in enumerate(AXES):
        panel(axs[0,col],axis,['matched_1'] if axis=='heads' else ['matched_1','matched_2'])
        k=2 if axis=='heads' else 3; panel(axs[1,col],axis,[f'cover_{k}',f'rank_{k}']); axs[0,col].set_title(axis.capitalize())
    save(fig,'figure4_count_controls')
    fig,axs=plt.subplots(1,2,figsize=(6.8,2.8))
    for name in SPECS:
        history=pd.read_csv(a.output/f'{name}_training.csv')
        for ax,metric in zip(axs,['val_ce','top1']): ax.plot(history.epoch,history[metric],label=name); ax.set_ylabel(metric); ax.set_xlabel('Epoch')
    axs[0].legend(ncol=2,fontsize=6); save(fig,'training_curves')
    trends=[]
    for axis,names in AXES.items():
        for m in ['val_ce','raw','matched_1','cover_full','rank_full']+([] if axis=='heads' else ['matched_2']):
            dif=np.diff(table.loc[names,m]); direction='increasing' if np.all(dif>=0) else 'decreasing' if np.all(dif<=0) else 'non-monotonic'
            trends.append(dict(axis=axis,metric=m,direction=direction))
    trend=pd.DataFrame(trends); trend.to_csv(a.output/'trends.csv',index=False)
    report='# Controlled ImageNet-1K ViT scaling\n\nSeven from-scratch models; one training seed42; shared preprocessing, schedule and120-epoch sample budget. '
    report+='Every primary checkpoint is epoch120, regardless of CE/CFS trends. Input bootstrap CIs do not quantify training-seed uncertainty. '
    report+='See ../stability/REPORT.md for the64-image CFS trajectory and convergence review before considering a uniform extension.\n\n'
    report+=table.to_markdown(floatfmt='.6f')+'\n\n## Q1–Q4: observed trends\n\n'+trend.to_markdown(index=False)
    report+='\n\nSee paired.csv for input-paired differences; monotonic point estimates are not significance tests. '
    report+='Q1 uses width Raw/Matched2; Q2 uses depth CE and functional metrics; Q3 uses the parameter-identical H2/H4/H8 axis; '
    report+='Q4 uses shared Matched1/2 and FixedK3/K2 support. Interpret K2 topology cautiously.\n\n'
    report+='## Q5 and limitations\n\nThis is controlled evidence in a visual classification architecture, regardless of whether every '
    report+='trend agrees with GPT-2. It does not prove universal scaling or that lowering CFS causes better accuracy. '
    report+='Inspect last20 CE slopes and all learning curves before asserting convergence or a nonsaturated task. '
    report+='FLOPs and GPU-hours are separate from parameter efficiency; no extra independent seeds were trained.\n\n'
    for name in ['figure1_controlled_scaling','figure2_functional_organization','figure3_fixed_capacity','figure4_count_controls','training_curves']:
        report+=f'![{name}]({name}.png)\n\n'
    (a.output/'REPORT.md').write_text(report,encoding='utf-8')
    english='We added a controlled ImageNet-1K experiment with seven ViTs trained from scratch under a shared DeiT-style recipe, shared offline preprocessing, and the same120-epoch initial budget. '
    english+='The study separates width, depth, and parameter-identical head decomposition, reusing one shared baseline checkpoint. '
    english+='Every primary comparison uses epoch120 rather than model-specific checkpoint selection. A shared64-image probe at epochs60/80/100/120 tracks functional stability alongside validation CE. '
    english+='We evaluate the same512 CFS images and report exact matched-candidate controls, fixed-size induced graphs, '
    english+='and input-paired bootstrap intervals. The attached tables report the observed directions and differences without assuming '
    english+='agreement with GPT-2. These intervals are conditional on a single training seed and do not establish causal performance benefits.\n'
    (a.output/'REBUTTAL_WCQn.md').write_text(english)
    (a.output/'COMPLETE.json').write_text(json.dumps(dict(models=list(SPECS),inputs=512,bootstrap=2000)))


if __name__=='__main__': main()
