"""Compact conference figures and complete gradient-diagnostic summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,path):
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(f'{path}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;args.output.mkdir(parents=True,exist_ok=False)
    lock=json.loads((root/'selection_lock.json').read_text());headline=lock['headline']
    gradients=[];losses=[];histories=[]
    for variant,choice in lock['selected'].items():
        for seed in (17,23,41):
            run=root/f'runs/{variant}_lambda{choice["strength"]:g}_seed{seed}'
            for filename,frames in [('gradients.csv',gradients),('probe_losses.csv',losses),('training_history.csv',histories)]:
                frame=pd.read_csv(run/filename);frame['variant']=variant;frame['seed']=seed;frame['strength']=choice['strength'];frames.append(frame)
    for path in sorted((root/'endpoints').glob('*/gradients.csv')):
        frame=pd.read_csv(path);frame['variant']=path.parent.name.split('_')[0];frame['seed']=int(path.parent.name.split('seed')[-1]);frame['strength']=np.nan;gradients.append(frame)
    gradient=pd.concat(gradients,ignore_index=True);gradient.to_csv(args.output/'all_gradient_diagnostics.csv',index=False)
    loss=pd.concat(losses,ignore_index=True);loss.to_csv(args.output/'selected_probe_losses.csv',index=False)
    history=pd.concat(histories,ignore_index=True);history.to_csv(args.output/'selected_training_history.csv',index=False)
    summaries=[]
    for keys,g in gradient.groupby(['variant','left','right']):
        summaries.append(dict(variant=keys[0],left=keys[1],right=keys[2],snapshots=len(g),mean_cosine=g.cosine.mean(),
            min_cosine=g.cosine.min(),max_cosine=g.cosine.max(),negative_fraction=(g.cosine<0).mean(),left_norm_mean=g.left_norm.mean(),right_norm_mean=g.right_norm.mean()))
    pd.DataFrame(summaries).to_csv(args.output/'gradient_summary.csv',index=False)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    palette={'Gating':'#0072B2','A0':'#009E73','R2':'#777777','A1':'#E69F00','A2':'#CC79A7','A3':'#D55E00'}
    labels={'Gating':'Gating','A0':'Joint only','R2':'Original R2','A1':'Joint+CFS','A2':'Joint+CFS+Del','A3':'Joint+CFS+Alpha'}
    summary=pd.read_csv(root/'routing_analysis/routing_summary.csv');pilot=pd.read_csv(root/'pilot_validation.csv')
    fig,axes=plt.subplots(1,2,figsize=(7,2.8))
    for variant in ('A1','A2','A3'):
        rows=pilot[pilot.variant==variant].sort_values('strength')
        axes[0].plot(rows.strength,rows.val_joint_regret,'o-',ms=3,label=labels[variant],color=palette[variant])
    axes[0].set(xscale='log',xlabel='Auxiliary weight λ',ylabel='Tune candidate-set KL regret',title='Fixed seed-17 pilots');axes[0].legend(frameon=False,fontsize=6.5)
    for method,color in palette.items():
        rows=summary.query('method==@method and compensation=="hard" and metric=="kl"').sort_values('keep')
        axes[1].plot(rows.keep,rows['mean'],'o-',ms=3,label=labels[method],color=color)
        axes[1].fill_between(rows.keep,rows.ci_low,rows.ci_high,alpha=.1,color=color)
    axes[1].set(xlabel='Retained heads K',xticks=[3,6,9],ylabel='Held-out routing KL',title='Shared hard gating, 95% CI');axes[1].legend(frameon=False,fontsize=6,ncol=2)
    save(fig,args.output/'figure1_regularization_ablation')
    fig,axes=plt.subplots(1,3,figsize=(7,2.6))
    for term in ('joint','cfs','deletion','alpha'):
        rows=loss[(loss.variant==headline)&(loss.loss==term)].groupby('epoch').value.mean()
        axes[0].plot(rows.index,rows.values,'o-',ms=2,label=term)
    axes[0].set(yscale='log',xlabel='Epoch',ylabel='Raw probe loss',title=f'{headline}: loss scales');axes[0].legend(frameon=False,fontsize=6)
    for term in ('cfs','deletion','alpha'):
        rows=gradient[(gradient.variant==headline)&(((gradient.left==term)&(gradient.right=='joint'))|((gradient.right==term)&(gradient.left=='joint')))]
        grouped=rows.groupby('epoch').cosine
        axes[1].plot(grouped.mean().index,grouped.mean().values,'o-',ms=2,label=term)
        axes[1].fill_between(grouped.mean().index,grouped.min().values,grouped.max().values,alpha=.12)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xlabel='Epoch',ylabel='Gradient cosine with joint',title='Shared encoder + embeddings');axes[1].legend(frameon=False,fontsize=6)
    for variant in ('A1','A2','A3'):
        rows=history[history.variant==variant].groupby('epoch').val_joint_regret.mean()
        axes[2].plot(rows.index,rows.values,label=variant,color=palette[variant])
    axes[2].set(xlabel='Epoch',ylabel='Tune joint KL regret',title='Selected λ, three seeds');axes[2].legend(frameon=False,fontsize=6)
    save(fig,args.output/'figure2_loss_gradient_diagnostics')
    corr=pd.read_csv(root/'interaction_analysis/correlation_summary.csv');inc=pd.read_csv(root/'interaction_analysis/incremental_prediction.csv')
    fig,axes=plt.subplots(1,2,figsize=(7,2.7))
    for offset,feature in enumerate(('cfs_mean','cfs_asymmetry','cosine','taylor_mean')):
        r=corr.query('split=="test" and feature==@feature and target=="magnitude"').sort_values('keep')
        axes[0].errorbar(r.keep+(offset-1.5)*.12,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o',ms=3,capsize=2,label=feature)
    axes[0].axhline(0,color='black',lw=.7);axes[0].set(xticks=[3,6,9],xlabel='Quartet upper budget K',ylabel='Mean within-input Spearman ρ',title='Relation to measured |interaction|');axes[0].legend(frameon=False,fontsize=6)
    for offset,target in enumerate(('interaction','magnitude')):
        r=inc[inc.target==target].sort_values('keep')
        axes[1].errorbar(r.keep+(offset-.5)*.15,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o',ms=3,capsize=3,label=target)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xticks=[3,6,9],xlabel='Quartet upper budget K',ylabel='MSE(controls) − MSE(+CFS)',title='Disjoint-input prediction, 95% CI');axes[1].legend(frameon=False,fontsize=6)
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    save(fig,args.output/'figure3_cfs_interaction_prediction')
    conditional=pd.read_csv(root/'routing_analysis/conditional_routing.csv')
    fig,axes=plt.subplots(1,3,figsize=(7,2.7),sharey=True)
    groups=['all448','measured128','low','high']
    for ax,keep in zip(axes,(3,6,9)):
        for offset,method in enumerate(('A0','R2',headline)):
            r=conditional.query('keep==@keep and method==@method and metric=="gating_minus_router"').set_index('group').loc[groups]
            ax.errorbar(np.arange(4)+(offset-1)*.12,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o',ms=3,capsize=2,label=labels[method],color=palette[method])
        ax.axhline(0,color='black',lw=.7);ax.set(xticks=np.arange(4),xticklabels=['All448','Sub128','Low','High'],title=f'K={keep}');ax.tick_params(axis='x',rotation=35)
    axes[0].set_ylabel('KL(gating) − KL(router)');axes[0].legend(frameon=False,fontsize=6)
    save(fig,args.output/'figure4_interaction_conditioned_routing')


if __name__=='__main__':main()
