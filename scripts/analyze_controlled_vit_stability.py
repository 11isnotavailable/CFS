"""Review CE and fixed64-input functional trajectories; never select checkpoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.controlled_vit_imagenet import SPECS, PROTOCOL
from scripts.analyze_matched_scaling import load


def matrices(path):
    meta=json.loads((path/'metadata.json').read_text())
    layers=meta.get('selected_layers',meta.get('layers')); assert len(layers)==1
    entry=dict(path=str(path.resolve()),family='ViT',model_id=meta['model'],heads=meta['heads'],
               samples=len(meta['sample_indices']),layers=layers)
    scores,valid,ids,_,_,_=load(entry,path)
    return ids,scores[:,0],valid[:,0]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); epochs=PROTOCOL['light_cfs']['epochs']
    summaries=[]; differences=[]; matrix_rows=[]; ce_rows=[]
    indices=np.random.default_rng(42).integers(64,size=(2000,64))
    for model in SPECS:
        stages={}; point={}; boot={}
        for epoch in epochs:
            root=a.root/'runs'/model/'light_cfs'/f'epoch_{epoch:03d}'
            assert (root/'COMPLETE.json').exists()
            stages[epoch]=matrices(root/'raw'); frame=pd.read_csv(root/'analysis/summary.csv'); frame['epoch']=epoch
            summaries.append(frame); point[epoch]=frame.set_index('metric')['mean']
            boot[epoch]=pd.read_csv(root/'analysis/bootstrap.csv').pivot(index='replicate',columns='metric',values='value')
        for old,new in zip(epochs[:-1],epochs[1:]):
            ia,sa,va=stages[old]; ib,sb,vb=stages[new]; assert np.array_equal(ia,ib) and len(ia)==64
            h=sa.shape[-1]; mask=(va&vb)[:,:,None]&~np.eye(h,dtype=bool)[None]
            maes=[]; correlations=[]; shared=[]
            for i in range(64):
                xa,xb=sa[i][mask[i]],sb[i][mask[i]]
                mae=float(np.mean(np.abs(xb-xa))) if len(xa) else np.nan
                rho=float(spearmanr(xa,xb).statistic) if len(xa)>2 and np.ptp(xa)>0 and np.ptp(xb)>0 else np.nan
                maes.append(mae); correlations.append(rho); shared.append(float((va[i]&vb[i]).mean()))
                matrix_rows.append(dict(model=model,from_epoch=old,to_epoch=new,input_id=int(ia[i]),
                    cfs_matrix_mae=mae,spearman=rho,shared_valid_source_ratio=shared[-1]))
            for metric in boot[old].columns:
                delta=boot[new][metric]-boot[old][metric]
                differences.append(dict(model=model,from_epoch=old,to_epoch=new,metric=metric,
                    mean=point[new][metric]-point[old][metric],ci_low=delta.quantile(.025),ci_high=delta.quantile(.975),
                    definition='new minus old; paired input bootstrap'))
            for metric,values in [('matrix_mae',maes),('matrix_spearman',correlations),('shared_valid_source_ratio',shared)]:
                v=np.asarray(values); draws=np.nanmean(v[indices],axis=1)
                differences.append(dict(model=model,from_epoch=old,to_epoch=new,metric=metric,mean=np.nanmean(v),
                    ci_low=np.nanquantile(draws,.025),ci_high=np.nanquantile(draws,.975),
                    definition='paired-input transition diagnostic; common-valid source rows only'))
        history=pd.DataFrame(json.loads((a.root/'runs'/model/'history.json').read_text()))
        old=history[history.epoch.between(91,100)].val_ce.mean(); new=history[history.epoch.between(111,120)].val_ce.mean()
        ce_rows.append(dict(model=model,ce_epochs91_100=old,ce_epochs111_120=new,relative_ce_change=(new-old)/old,
            ce_epoch120=float(history[history.epoch==120].val_ce.iloc[0]),
            ce_last20_slope=np.polyfit(history.tail(20).epoch,history.tail(20).val_ce,1)[0]))
    summary=pd.concat(summaries,ignore_index=True); delta=pd.DataFrame(differences); ce=pd.DataFrame(ce_rows)
    summary.to_csv(a.output/'trajectory.csv',index=False); delta.to_csv(a.output/'paired_epoch_changes.csv',index=False)
    pd.DataFrame(matrix_rows).to_csv(a.output/'matrix_changes_per_input.csv',index=False)
    ce.to_csv(a.output/'ce_convergence.csv',index=False)
    summary.to_latex(a.output/'trajectory.tex',index=False,float_format='%.6f')
    delta.to_latex(a.output/'paired_epoch_changes.tex',index=False,float_format='%.6f')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(8,4.7))
    for ax,metric in zip(axs.flat,['raw','matched_1','matched_2','cover_full','rank_full','val_ce']):
        for name in SPECS:
            if metric=='val_ce':
                history=pd.DataFrame(json.loads((a.root/'runs'/name/'history.json').read_text()))
                ax.plot(history.epoch,history.val_ce,label=name)
            else:
                values=summary[(summary.model==name)&(summary.metric==metric)].sort_values('epoch')
                if len(values): ax.plot(values.epoch,values['mean'],'o-',label=name); ax.fill_between(values.epoch,values.ci_low,values.ci_high,alpha=.08)
        ax.set_title(metric); ax.set_xlabel('Epoch'); ax.grid(alpha=.15)
    axs[0,0].legend(ncol=2,fontsize=6); fig.tight_layout()
    fig.savefig(a.output/'functional_stability.pdf',bbox_inches='tight'); fig.savefig(a.output/'functional_stability.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    last=delta[delta.to_epoch==120]
    report='# CE and functional stability review at the common120-epoch budget\n\n'
    report+='All primary models remain epoch120. The fixed64-image probe and alpha grid are unchanged across epochs60/80/100/120. '
    report+='This report does not automatically extend training or select a favorable CFS checkpoint.\n\n'
    report+=ce.to_markdown(index=False,floatfmt='.6f')+'\n\n'+last.to_markdown(index=False,floatfmt='.6f')
    report+='\n\n![Trajectory](functional_stability.png)\n\nReview both80→100 and100→120 changes, their paired intervals, '
    report+='matrix MAE/rank correlation, valid-source overlap and CE curves. A CI spanning zero is not an equivalence test. '
    report+='Stable averages can hide changing pairwise matrices; matrix diagnostics use only sources valid at both checkpoints '
    report+='and their overlap rate is reported. H2 correlations have very few candidate pairs and can be undefined. '
    report+='A single seed and64 inputs limit precision. If substantial CE improvement or functional drift persists in any model, '
    report+='review a common extension for all seven; if CE worsens, inspect overfitting before extending. '
    report+='The next budget must be decided jointly after this review, never per-model based on a desired scaling direction.\n'
    (a.output/'REPORT.md').write_text(report,encoding='utf-8')
    (a.output/'REVIEW_READY.json').write_text(json.dumps(dict(initial_budget_complete=120,
        automatic_extension=False,models=list(SPECS),requires_joint_convergence_review=True),indent=2))


if __name__=='__main__': main()
