"""Paired input bootstrap and fixed target checks for budget-aware continuation."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;args.output.mkdir(exist_ok=False,parents=True)
    frame=pd.read_csv(root/'routing_per_input.csv');assert len(frame)==448*3*9*3*2
    assert not frame.duplicated(['sample','method','seed','keep','compensation']).any()
    averaged=frame.groupby(['sample','method','keep','compensation'])[['kl','correct']].mean().reset_index()
    draws=np.random.default_rng(42).integers(0,448,(2000,448));np.savez_compressed(args.output/'bootstrap_draws.npz',draws=draws)
    def stat(values):
        x=np.asarray(values);assert len(x)==448
        ci=np.quantile(x[draws].mean(1),[.025,.975])
        return dict(mean=float(x.mean()),sd_inputs=float(x.std(ddof=1)),ci_low=float(ci[0]),ci_high=float(ci[1]))
    seeds=[]
    for (method,keep,comp,seed),g in frame.groupby(['method','keep','compensation','seed']):
        for metric in ('kl','correct'):
            seeds.append(dict(method=method,keep=keep,compensation=comp,seed=seed,metric=metric,**stat(g.sort_values('sample')[metric])))
    per_seed=pd.DataFrame(seeds);per_seed.to_csv(args.output/'per_seed_summary.csv',index=False)
    summaries=[]
    for (method,keep,comp),g in averaged.groupby(['method','keep','compensation']):
        for metric in ('kl','correct'):
            s=per_seed.query('method==@method and keep==@keep and compensation==@comp and metric==@metric')['mean']
            summaries.append(dict(method=method,keep=keep,compensation=comp,metric=metric,seed_mean_sd=float(s.std(ddof=1)),**stat(g.sort_values('sample')[metric])))
    summary=pd.DataFrame(summaries);summary.to_csv(args.output/'summary.csv',index=False)
    comparisons=[]
    for comp in ('hard','shared_r2_merge'):
        for keep in (3,6,9):
            for metric in ('kl','correct'):
                pivot=averaged.query('keep==@keep and compensation==@comp').pivot(index='sample',columns='method',values=metric)
                for method in ('W1','W1.5','W2','W3'):
                    for reference in ('A3','R2','Gating','W1'):
                        if reference==method:continue
                        comparisons.append(dict(method=method,reference=reference,keep=keep,compensation=comp,metric=metric,
                                                **stat(pivot[method]-pivot[reference])))
    paired=pd.DataFrame(comparisons);paired.to_csv(args.output/'paired.csv',index=False)
    lock=json.loads((root/'selection_lock.json').read_text())
    primary='A3' if lock['fallback_to_A3'] else f'W{lock["primary"]["weight9"]:g}'
    hard=summary.query('compensation=="hard" and metric=="kl"').pivot(index='method',columns='keep',values='mean')
    targets=[]
    for method,row in hard.iterrows():
        targets.append(dict(method=method,kl3=row[3],kl6=row[6],kl9=row[9],pass_K3=bool(row[3]<=.10517),
            pass_K6=bool(row[6]<=.02480),pass_K9=bool(row[9]<.00700),all_targets=bool(row[3]<=.10517 and row[6]<=.02480 and row[9]<.00700),
            primary=method==primary))
    pd.DataFrame(targets).to_csv(args.output/'targets.csv',index=False)
    histories=[];gradients=[];probes=[]
    for run in sorted((root/'runs').glob('w*_seed*')):
        if not run.is_dir():continue
        config=json.loads((run/'config.json').read_text())
        for filename,parts in [('training_history.csv',histories),('gradients.csv',gradients),('probe_losses.csv',probes)]:
            a=pd.read_csv(run/filename);a['weight9']=config['weight9'];a['seed']=config['seed'];parts.append(a)
    history=pd.concat(histories);gradient=pd.concat(gradients);probe=pd.concat(probes)
    history.to_csv(args.output/'training_history.csv',index=False);gradient.to_csv(args.output/'gradient_diagnostics.csv',index=False);probe.to_csv(args.output/'probe_losses.csv',index=False)
    for name in ('summary','paired','targets'):
        table=pd.read_csv(args.output/f'{name}.csv');table.to_json(args.output/f'{name}.json',orient='records',indent=2)
        table.to_latex(args.output/f'{name}.tex',index=False,float_format='%.6f')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'A3':'#777777','R2':'#000000','Gating':'#56B4E9','W1':'#009E73','W1.5':'#E69F00','W2':'#CC79A7','W3':'#D55E00'}
    fig,axes=plt.subplots(1,2,figsize=(7,2.8))
    for method,color in colors.items():
        r=summary.query('method==@method and compensation=="hard" and metric=="kl"').sort_values('keep')
        axes[0].plot(r.keep,r['mean'],'o-',ms=3,label=method,color=color)
        if method.startswith('W'):
            r=paired.query('method==@method and reference=="A3" and compensation=="hard" and metric=="kl"').sort_values('keep')
            offset=(list(colors).index(method)-5.5)*.1
            axes[1].errorbar(r.keep+offset,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o',ms=3,capsize=2,label=method,color=color)
    axes[0].set(xlabel='Retained heads K',ylabel='Mean hard-gating KL',xticks=[3,6,9]);axes[0].legend(fontsize=6,ncol=2,frameon=False)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xlabel='Retained heads K',ylabel='KL(update) − KL(A3), 95% CI',xticks=[3,6,9]);axes[1].legend(frameon=False,fontsize=6)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(args.output/f'figure1_budget_finetuning.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7,2.6))
    candidates=pd.read_csv(root/'validation_candidates.csv')
    for ax,k in zip(axes,(3,6,9)):
        for weight,g in candidates.groupby('weight9'):ax.plot(g.epoch,g[f'val_K{k}'],label=f'w9={weight:g}')
        ax.axhline(lock['baseline_tune'][f'val_K{k}'],color='black',ls='--',lw=.7,label='A3 start')
        ax.set(title=f'K={k}',xlabel='Fine-tuning epoch',ylabel='Mean tune regret')
    axes[0].legend(fontsize=6,frameon=False);fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(args.output/f'figure2_validation_tradeoffs.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (args.output/'primary.json').write_text(json.dumps(dict(primary=primary,selection=lock['primary'],fallback=lock['fallback_to_A3']),indent=2))
    print(hard.to_string());print('PRIMARY',primary)


if __name__=='__main__':main()
