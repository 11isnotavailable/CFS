"""Partial B1/B2 learning report; not a seven-model CFS analysis."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
root=parser.parse_args().root
out=root/'partial_training';out.mkdir(exist_ok=True)
rows=[];values={};draws={}
for name in ['B1','B2']:
    history=pd.DataFrame(json.loads((root/'runs'/name/'history.json').read_text()))
    checkpoints=history[history.tokens.isin([500000000,1000000000,1500000000,2000000000])]
    for _,row in checkpoints.iterrows():
        rows.append(dict(model=name,**row[['tokens','train_ce','train_accuracy','validation_ce','validation_accuracy','gpu_hours']].to_dict()))
    values[name]=np.load(root/'runs'/name/'validation_2000000000.npz')
    ix=np.random.default_rng(42).integers(len(values[name]['token_count']),size=(2000,len(values[name]['token_count'])))
    draws[name]={k:values[name][key][ix].sum(1)/values[name]['token_count'][ix].sum(1) for k,key in [('ce','loss_sum'),('accuracy','correct')]}
pd.DataFrame(rows).to_csv(out/'checkpoints.csv',index=False)
assert np.array_equal(values['B1']['document_sha256'],values['B2']['document_sha256'])
summary=[]
for metric in ['ce','accuracy']:
    for model in ['B1','B2','B2-B1']:
        d=draws[model][metric] if model in draws else draws['B2'][metric]-draws['B1'][metric]
        key='loss_sum' if metric=='ce' else 'correct'
        p={name:values[name][key].sum()/values[name]['token_count'].sum() for name in values}
        mean=float(p[model]) if model in p else float(p['B2']-p['B1'])
        summary.append(dict(model=model,metric=metric,mean=mean,ci_low=float(np.quantile(d,.025)),ci_high=float(np.quantile(d,.975)),bootstrap_unit='document',bootstrap=2000,training_seeds=1))
pd.DataFrame(summary).to_csv(out/'validation_paired.csv',index=False)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.1,2.6))
for name in values:
    h=pd.DataFrame(json.loads((root/'runs'/name/'history.json').read_text())).dropna(subset=['validation_ce'])
    for ax,metric in zip(axs,['validation_ce','validation_accuracy']):
        ax.plot(h.tokens/1e9,h[metric],label=name);ax.set_xlabel('Input tokens (billions)');ax.set_ylabel(metric);ax.grid(alpha=.15)
axs[0].legend();fig.suptitle('Partial results: 2 of 7 models trained');fig.tight_layout()
fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf')
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(summary).to_string(index=False))
