"""Combine completed offline runs, paired contrasts and publication figures."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    dirs=sorted(x.parent for x in a.input.glob('*/summary.csv'))
    if not dirs: raise ValueError('No completed results')
    summary=pd.concat([pd.read_csv(x/'summary.csv') for x in dirs],ignore_index=True)
    summary.to_csv(a.output/'all_results.csv',index=False)
    summary.to_json(a.output/'all_results.json',orient='records',indent=2)
    summary[['family','model','metric','mean','ci_low','ci_high']].to_latex(a.output/'all_results.tex',index=False,float_format='%.6f')
    byname={pd.read_csv(x/'summary.csv').model.iloc[0]:x for x in dirs}
    contrasts=[]; trends=[]
    for family,group in summary.groupby('family'):
        order=group[['model','parameters']].drop_duplicates().sort_values('parameters').model.tolist()
        for metric,g in group.groupby('metric'):
            values=g.set_index('model').reindex(order)['mean'].to_numpy()
            dif=np.diff(values)
            trends.append(dict(family=family,metric=metric,models=len(values),monotonic_increasing=bool(len(values)>1 and np.all(dif>=0)),monotonic_decreasing=bool(len(values)>1 and np.all(dif<=0))))
        for left,right in zip(order[:-1],order[1:]):
            lp=json.loads((byname[left]/'provenance.json').read_text()); rp=json.loads((byname[right]/'provenance.json').read_text())
            # Context offsets and token-position hashes do NOT prove shared text.
            lh=lp['entry'].get('input_identity_sha256'); rh=rp['entry'].get('input_identity_sha256')
            if not lh or lh!=rh: raise ValueError(f'Cannot pair {left}/{right}: frozen input identity hash required')
            for key in ['seed','bootstrap','ordered_input_ids']:
                if key not in lp or lp[key]!=rp.get(key):
                    raise ValueError(f'Cannot pair bootstrap replicates: {key} differs or is missing')
            l=pd.read_csv(byname[left]/'per_input_layer.csv'); r=pd.read_csv(byname[right]/'per_input_layer.csv')
            if sorted(l.input_id.unique())!=sorted(r.input_id.unique()): raise ValueError('Input IDs differ')
            lb=pd.read_csv(byname[left]/'bootstrap.csv'); rb=pd.read_csv(byname[right]/'bootstrap.csv')
            for metric in sorted(set(lb.metric)&set(rb.metric)):
                x=lb[lb.metric==metric].sort_values('replicate').value.to_numpy()
                y=rb[rb.metric==metric].sort_values('replicate').value.to_numpy()
                d=y-x; lo,hi=np.quantile(d,[.025,.975])
                means=group[group.metric==metric].set_index('model')['mean']
                contrasts.append(dict(family=family,left=left,right=right,metric=metric,delta=means[right]-means[left],ci_low=lo,ci_high=hi,paired_inputs=l.input_id.nunique()))
    pd.DataFrame(contrasts).to_csv(a.output/'paired_contrasts.csv',index=False)
    pd.DataFrame(trends).to_csv(a.output/'trends.csv',index=False)
    stability=pd.concat([pd.read_csv(x/'sampling_stability.csv') for x in dirs],ignore_index=True)
    stability.groupby(['model','metric','draws'],as_index=False)['mean'].mean().to_csv(a.output/'sampling_stability_summary.csv',index=False)
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def panel(ax,family,metrics,title,ylabel):
        g=summary[summary.family==family]
        unit=1e6 if family=='ViT' else 1e9
        for metric in metrics:
            v=g[g.metric==metric].sort_values('parameters')
            if v.empty: continue
            x=v.parameters.to_numpy()/unit
            reference=metric.endswith('full') or metric=='raw'
            value=metric.split('_')[-1]
            label='Raw Oracle' if metric=='raw' else ('Full H' if reference else ('N=' if metric.startswith('matched_') else 'K=')+value)
            colors={'2':'#0072B2','4':'#E69F00','5':'#009E73','6':'#009E73','8':'#CC79A7','11':'#D55E00','12':'#D55E00'}
            ax.errorbar(x,v['mean'],yerr=np.maximum(0,np.stack([v['mean']-v.ci_low,v.ci_high-v['mean']])),fmt='o--' if reference else 'o-',color='#555555' if reference else colors[value],markersize=3,capsize=2,label=label)
        if g.empty: ax.text(.5,.5,'Awaiting source matrices',ha='center',transform=ax.transAxes)
        else:
            ticks=g[['parameters','model']].drop_duplicates().sort_values('parameters')
            ax.set_xticks(ticks.parameters/unit,[f'{v/unit:.3g}' for v in ticks.parameters])
            ax.legend(frameon=False,fontsize=6)
        ax.set(title=title,xlabel='Parameters (M)' if family=='ViT' else 'Parameters (B)',ylabel=ylabel); ax.grid(alpha=.2)
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.5),layout='constrained')
    for ax,fam in zip(axes,['Qwen','ViT']): panel(ax,fam,['raw','matched_2','matched_4','matched_5','matched_8','matched_11'],fam,'Oracle recovery')
    for ext in ['pdf','png']: fig.savefig(a.output/f'figure1_matched_N.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.0,4.6),layout='constrained')
    for row,fam in zip(axes,['Qwen','ViT']):
        for ax,prefix in zip(row,['cover','rank']): panel(ax,fam,[f'{prefix}_{k}' for k in ['full',4,6,8,12]],fam,f'{prefix.title()} / nodes')
    for ext in ['pdf','png']: fig.savefig(a.output/f'figure2_fixed_K.{ext}',dpi=300)
    plt.close(fig)

if __name__=='__main__': main()
