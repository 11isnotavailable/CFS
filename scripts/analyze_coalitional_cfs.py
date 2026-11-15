"""Paired independent-input bootstraps, budget curves and paper figures."""
import argparse,json,sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scripts.train_coalitional_cfs import arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root;out=a.output;out.mkdir(exist_ok=False)
    draws=np.random.default_rng(42).integers(0,256,(2000,256));np.savez_compressed(out/'bootstrap_draws.npz',draws=draws)
    def stat(x,y=None):
        x=np.asarray(x,float);mask=np.isfinite(x)
        if y is not None:y=np.asarray(y,float);mask&=np.isfinite(y)
        x=x[mask];assert len(x)>0
        local=draws if len(x)==256 else np.random.default_rng(42).integers(0,len(x),(2000,len(x)))
        point=x.mean();boot=x[local].mean(-1)
        if y is not None:
            y=y[mask];point/=y.mean();boot/=y[local].mean(-1)
        lo,hi=np.quantile(boot,[.025,.975]);return dict(mean=point,ci_low=lo,ci_high=hi,sd_inputs=x.std(ddof=1) if y is None else np.nan,n_inputs=len(x))
    frames=[pd.read_csv(p) for p in sorted((root/'evaluation').glob('*/per_input.csv'))];raw=pd.concat(frames,ignore_index=True)
    raw.to_csv(out/'per_input.csv',index=False);raw.to_json(out/'per_input.json',orient='records')
    keys=['pool','method','curve','budget','labels'];metrics=['oracle','selected','regret','random_recovery','random_regret','spearman','undefined_spearman','best_hit','top2_recall','random_best_hit','random_top2_recall','mse','constant_mse','prediction_below_zero','prediction_above_one']
    avg=raw.groupby(keys+['sample'])[metrics].mean().reset_index();summary=[];seed_rows=[]
    allmetrics=metrics+['nmse','normalized_regret','chance_adjusted_hit','chance_adjusted_top2']
    def value(g,metric):
        if metric=='nmse':return g.mse,g.constant_mse
        if metric=='normalized_regret':return g.regret,g.random_regret
        if metric=='chance_adjusted_hit':return g.best_hit-g.random_best_hit,1-g.random_best_hit
        if metric=='chance_adjusted_top2':return g.top2_recall-g.random_top2_recall,1-g.random_top2_recall
        return g[metric],None
    for group,g in raw.groupby(keys+['seed']):
        g=g.sort_values('sample')
        for metric in allmetrics:
            x,y=value(g,metric);seed_rows.append(dict(zip(keys+['seed'],group),metric=metric,**stat(x,y)))
    seeds=pd.DataFrame(seed_rows)
    for group,g in avg.groupby(keys):
        g=g.sort_values('sample');sub=seeds
        for key,val in zip(keys,group):sub=sub[sub[key]==val]
        for metric in allmetrics:
            x,y=value(g,metric);v=sub[sub.metric==metric]['mean'];summary.append(dict(zip(keys,group),metric=metric,seed_sd=float(v.std(ddof=1)) if len(v)>1 else 0.,**stat(x,y)))
    summary=pd.DataFrame(summary);paired=[];paired_inputs=[]
    for pool in (42,123,2027):
        for budget in (2500,5000,10000,20000):
            left=avg.query('pool==@pool and method=="pair" and curve=="labels" and budget==@budget').sort_values('sample')
            for curve in ('labels','cost','evaluations'):
                right=avg.query('pool==@pool and method=="joint" and curve==@curve and budget==@budget').sort_values('sample')
                if not len(right):continue
                gain=right.oracle.to_numpy()-left.oracle.to_numpy();selected=right.selected.to_numpy()-left.selected.to_numpy()
                for metric in ('selected','regret','spearman','best_hit','top2_recall','nmse','normalized_regret','chance_adjusted_hit','chance_adjusted_top2','realized_gain_ratio'):
                    if metric=='realized_gain_ratio':s=stat(selected,gain)
                    elif metric in ('nmse','normalized_regret','chance_adjusted_hit','chance_adjusted_top2'):
                        rx,ry=value(right,metric);lx,ly=value(left,metric)
                        values=np.stack([np.asarray(z) for z in (rx,ry,lx,ly)],1);values=values[np.isfinite(values).all(-1)]
                        local=draws if len(values)==256 else np.random.default_rng(42).integers(0,len(values),(2000,len(values)))
                        rx,ry,lx,ly=values.T;point=rx.mean()/ry.mean()-lx.mean()/ly.mean()
                        boot=rx[local].mean(-1)/ry[local].mean(-1)-lx[local].mean(-1)/ly[local].mean(-1)
                        lo,hi=np.quantile(boot,[.025,.975]);s=dict(mean=point,ci_low=lo,ci_high=hi,sd_inputs=np.nan,n_inputs=len(values))
                    else:s=stat(right[metric].to_numpy()-left[metric].to_numpy())
                    paired.append(dict(pool=pool,budget=budget,curve=curve,metric=metric,contrast='Joint2_minus_Pairwise' if metric!='realized_gain_ratio' else 'selected_gain_over_oracle_gain',**s))
                for i in range(256):paired_inputs.append(dict(pool=pool,budget=budget,curve=curve,sample=i,selected_difference=selected[i],oracle_difference=gain[i]))
    d2=[]
    for pool in (42,123,2027):
        for budget in (2500,5000,10000,20000):
            direct=avg.query('pool==@pool and method=="joint" and curve=="labels" and budget==@budget').sort_values('sample')
            for reference in ('predicted_pair_max','predicted_pair_mean'):
                other=avg.query('pool==@pool and method==@reference and budget==@budget').sort_values('sample')
                for metric in ('selected','regret','spearman'):
                    d2.append(dict(pool=pool,budget=budget,metric=metric,contrast='direct_joint_minus_'+reference,**stat(direct[metric].to_numpy()-other[metric].to_numpy())))
    oracle_rows=[];oracle_input=[];validity=[]
    for pool in (42,123,2027):
        d=arrays(root,'test',pool)
        for threshold in (1e-5,1e-4,1e-3):
            valid=d['drop']>threshold;den=valid.sum(-1)
            r1=d['pair'].max(-1);r2=d['joint'].max(-1);diff=r2-r1
            assert diff[d['valid']].min()>-1e-5
            def aggregate(v):return np.divide(np.where(valid,v,0).sum(-1),den,out=np.full(len(den),np.nan),where=den>0)
            for metric,v in [('pair_oracle',r1),('joint_oracle',r2),('oracle_gain',diff),('source_gain_fraction',diff>1e-6)]:
                x=aggregate(v);oracle_rows.append(dict(pool=pool,threshold=threshold,metric=metric,**stat(x)))
                if threshold==1e-5:
                    for i,z in enumerate(x):oracle_input.append(dict(pool=pool,sample=i,metric=metric,value=z))
            g=aggregate(diff);oracle_rows.append(dict(pool=pool,threshold=threshold,metric='input_gain_fraction',**stat(np.where(np.isfinite(g),g>1e-6,np.nan))))
            validity.append(dict(pool=pool,threshold=threshold,valid_fraction=valid.mean(),inputs_with_no_valid_sources=int((den==0).sum()),minimum_drop=float(d['drop'].min()),median_drop=float(np.median(d['drop']))))
    cost_records=[];stage=[]
    for directory in sorted((root/'teacher').glob('*/*')):
        if not directory.is_dir():continue
        for file in directory.glob('[0-9][0-9][0-9][0-9][0-9].npz'):
            with np.load(file) as f:
                cost_records.append(dict(pool=directory.parent.name,split=directory.name,sample=int(f['sample']),**{key:float(f[key]) for key in ('capture_seconds','drop_seconds','pair_seconds','joint_seconds','pair_variants','joint_variants','pair_forward_calls','joint_forward_calls','backward_calls','feasible_fallbacks')}))
        stage.append(dict(pool=directory.parent.name,split=directory.name,**json.loads((directory/'COMPLETE.json').read_text())))
    costs=pd.DataFrame(cost_records);costs.to_csv(out/'teacher_per_input_cost.csv',index=False)
    pd.DataFrame(stage).to_csv(out/'teacher_stage_wall.csv',index=False)
    plan=json.loads((root/'training_plan.json').read_text());theory=[]
    for H in (12,40):
        for kind,n in [('pair',H*(H-1)),('joint',H*(H-1)*(H-2)//2)]:theory.append(dict(heads=H,kind=kind,relations=n,estimated_seconds_at_ViT12_unit_cost=n*plan['cost'][kind],projection_not_measurement=True))
    configs=[json.loads(p.read_text()) for p in sorted((root/'runs').glob('*/config.json'))]
    tables=dict(summary=summary,per_seed=seeds,paired=pd.DataFrame(paired),paired_per_input=pd.DataFrame(paired_inputs),pair_information_paired=pd.DataFrame(d2),oracle=pd.DataFrame(oracle_rows),oracle_per_input=pd.DataFrame(oracle_input),validity=pd.DataFrame(validity),
        relation_complexity=pd.DataFrame(theory),training=pd.DataFrame(configs),teacher_events=pd.read_csv(root/'profile/teacher_events.csv'),predictor_cost=pd.read_csv(root/'profile/predictor_cost.csv'),continuous_sensitivity=pd.read_csv(root/'profile/continuous_sensitivity.csv'),full_relation_space=pd.read_csv(root/'profile/full_relation_space.csv'))
    for name,t in tables.items():
        t.to_csv(out/f'{name}.csv',index=False);t.to_json(out/f'{name}.json',orient='records',indent=2)
        if name not in ('paired_per_input','oracle_per_input'):t.to_latex(out/f'{name}.tex',index=False,float_format='%.6f')
    # Means and CIs in standalone paper figures.
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'pair':'#0072B2','joint':'#D55E00'}
    def save(fig,name):
        fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.6))
    for kind,count,label in [('pair',48,'Pairwise'),('joint',72,'Joint-2')]:
        r=tables['oracle'][(tables['oracle'].pool==42)&(tables['oracle'].threshold==1e-5)&(tables['oracle'].metric==kind+'_oracle')].iloc[0]
        axes[0].errorbar(plan['cost'][kind]*count*1000,r['mean'],yerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o',capsize=3,label=label,color=colors[kind])
    axes[0].set(xscale='log',xlabel='Measured batched suffix time per input (ms)',ylabel='Oracle recovery');axes[0].legend(frameon=False)
    values=tables['oracle_per_input'].query('pool==42 and metric=="oracle_gain"').value;axes[1].hist(values,bins=20,color='#666666');axes[1].set(xlabel='Joint-2 minus Pairwise oracle recovery',ylabel='Independent inputs');save(fig,'figure1_oracle_cost')
    fig,axes=plt.subplots(3,3,figsize=(7.1,6.2))
    for row,(curve,title) in enumerate([('labels','Equal labels'),('cost','Equal estimated GPU-path cost'),('evaluations','Equal grid-intervention counts')]):
        for col,metric in enumerate(('nmse','spearman','normalized_regret')):
            ax=axes[row,col]
            for kind in ('pair','joint'):
                c='labels' if kind=='pair' else curve;r=summary.query('pool==42 and method==@kind and curve==@c and metric==@metric').sort_values('budget')
                ax.errorbar(r.budget,r['mean'],yerr=[r['mean']-r.ci_low,r.ci_high-r['mean']],fmt='o-',ms=3,capsize=2,color=colors[kind],label='Pairwise' if kind=='pair' else 'Joint-2')
            ax.set(xscale='log',xlabel='Pairwise-equivalent budget' if curve!='labels' else 'Oracle labels',ylabel={'nmse':'NMSE','spearman':'Spearman correlation','normalized_regret':'Normalized regret'}[metric],title=title if col==1 else '')
            ax.set_xticks([2500,5000,10000,20000],['2.5k','5k','10k','20k']);ax.xaxis.set_minor_locator(NullLocator())
            if row==0 and col==0:ax.legend(frameon=False)
    save(fig,'figure2_predictability')
    fig,axes=plt.subplots(1,2,figsize=(7,2.6))
    for ax,kind in zip(axes,('pair','joint')):
        for metric,style in [('oracle','--'),('selected','-')]:
            r=summary.query('pool==42 and method==@kind and curve=="labels" and metric==@metric').sort_values('budget')
            ax.errorbar(r.budget,r['mean'],yerr=[r['mean']-r.ci_low,r.ci_high-r['mean']],fmt='o'+style,ms=3,capsize=2,label=metric)
        ax.set(title='Pairwise' if kind=='pair' else 'Joint-2',xscale='log',xlabel='Oracle labels',ylabel='Recovery (oracle-alpha conditioned)');ax.legend(frameon=False)
        ax.set_xticks([2500,5000,10000,20000],['2.5k','5k','10k','20k']);ax.xaxis.set_minor_locator(NullLocator())
    save(fig,'figure3_oracle_predictor_gap')
    print(tables['oracle'].query('pool==42 and threshold==1e-5').to_string(index=False))
    print(summary.query('pool==42 and budget==20000 and method in ["pair","joint"] and metric in ["nmse","spearman","normalized_regret","selected"]').to_string(index=False))
    print(tables['paired'].query('pool==42 and budget==20000').to_string(index=False))


if __name__=='__main__':main()
