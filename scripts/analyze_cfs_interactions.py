"""Input-cluster statistics and disjoint-input low-capacity interaction prediction."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def boot(values,seed=42):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,mean=np.nan,sd=np.nan,ci_low=np.nan,ci_high=np.nan)
    draws=np.random.default_rng(seed).integers(0,len(x),(2000,len(x)))
    ci=np.quantile(x[draws].mean(1),[.025,.975])
    return dict(n=len(x),mean=x.mean(),sd=x.std(ddof=1),ci_low=ci[0],ci_high=ci[1])


def rho(a,b):
    a=np.asarray(a);b=np.asarray(b);valid=np.isfinite(a)&np.isfinite(b)
    if valid.sum()<4 or np.ptp(a[valid])==0 or np.ptp(b[valid])==0:return np.nan
    return float(spearmanr(a[valid],b[valid]).statistic)


def ridge_predict(x,y,test,alpha=1.):
    mean=x.mean(0);std=x.std(0).clip(min=1e-6);ym=y.mean();ys=max(y.std(),1e-8)
    z=(x-mean)/std;target=(y-ym)/ys
    coefficient=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@target)
    prediction=((test-mean)/std)@coefficient*ys+ym
    return prediction,dict(mean=mean,std=std,y_mean=ym,y_std=ys,coefficient=coefficient,alpha=alpha)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;args.output.mkdir(parents=True,exist_ok=False)
    auxiliary={s:dict(np.load(root/f'inputs/{s}_aux.npz')) for s in ('tune','test')}
    rows=[];input_rows=[];corr_rows=[]
    for path in sorted((root/'interactions').glob('*.npz')):
        a=np.load(path);split=str(a['split']);sample=int(a['sample']);aux=auxiliary[split]
        np.testing.assert_equal(str(a['path']),str(aux['paths'][sample]))
        scores=aux['scores'][sample];drop=aux['drops'][sample];taylor=aux['taylor'][sample]
        for b,keep in enumerate((3,6,9)):
            values=a['interaction'][b];local=[]
            for pair,(i,j) in enumerate(a['pairs']):
                cij,cji=float(scores[i,j]),float(scores[j,i])
                valid=bool(drop[i]>1e-5 and drop[j]>1e-5 and np.isfinite(cij) and np.isfinite(cji))
                background=a['masks'][a['quartets'][b,pair,0]]
                row=dict(split=split,sample=sample,keep=keep,pair=pair,i=int(i),j=int(j),interaction=float(values[pair]),
                    magnitude=float(abs(values[pair])),valid=valid,cfs_ij=cij,cfs_ji=cji,cfs_mean=(cij+cji)/2,
                    cfs_max=max(cij,cji),cfs_asymmetry=abs(cij-cji),cosine=float(a['cosine'][i,j]),
                    taylor_mean=float((taylor[i]+taylor[j])/2),taylor_difference=float(abs(taylor[i]-taylor[j])),
                    log_drop_mean=float((np.log10(max(drop[i],1e-12))+np.log10(max(drop[j],1e-12)))/2),
                    log_drop_difference=float(abs(np.log10(max(drop[i],1e-12))-np.log10(max(drop[j],1e-12)))))
                row.update({f'context_{h}':int(background[h]) for h in range(12)})
                local.append(row);rows.append(row)
            frame=pd.DataFrame(local);good=frame[frame.valid]
            input_rows.append(dict(split=split,sample=sample,keep=keep,complexity=float(abs(values).mean()),
                signed_mean=float(values.mean()),positive_fraction=float((values>0).mean()),valid_fraction=float(frame.valid.mean()),
                cfs_mean=float(good.cfs_mean.mean()),cfs_asymmetry=float(good.cfs_asymmetry.mean()),
                taylor_mean=float(good.taylor_mean.mean()),cosine=float(good.cosine.mean())))
            for feature in ('cfs_ij','cfs_ji','cfs_mean','cfs_max','cfs_asymmetry','cosine','taylor_mean','taylor_difference','log_drop_mean'):
                for target in ('interaction','magnitude'):
                    corr_rows.append(dict(split=split,sample=sample,keep=keep,feature=feature,target=target,rho=rho(good[feature],good[target]),pairs=len(good)))
    pairs=pd.DataFrame(rows);inputs=pd.DataFrame(input_rows);correlations=pd.DataFrame(corr_rows)
    pairs.to_csv(args.output/'pair_interactions.csv',index=False);inputs.to_csv(args.output/'input_complexity.csv',index=False)
    correlations.to_csv(args.output/'within_input_correlations.csv',index=False)
    summaries=[]
    for keys,g in inputs.groupby(['split','keep']):
        for metric in ('complexity','signed_mean','positive_fraction','valid_fraction'):
            summaries.append(dict(split=keys[0],keep=keys[1],metric=metric,**boot(g[metric])))
    pd.DataFrame(summaries).to_csv(args.output/'interaction_summary.csv',index=False)
    corr_summary=[]
    for keys,g in correlations.groupby(['split','keep','feature','target']):
        corr_summary.append(dict(split=keys[0],keep=keys[1],feature=keys[2],target=keys[3],**boot(g.rho)))
    pd.DataFrame(corr_summary).to_csv(args.output/'correlation_summary.csv',index=False)
    prediction_rows=[];coefficients={}
    controls=['cosine','taylor_mean','taylor_difference','log_drop_mean','log_drop_difference']
    cfs=['cfs_mean','cfs_max','cfs_asymmetry']
    for keep in (3,6,9):
        train=pairs[(pairs.keep==keep)&(pairs.split=='tune')&pairs.valid].copy()
        test=pairs[(pairs.keep==keep)&(pairs.split=='test')&pairs.valid].copy()
        def features(frame,columns):
            continuous=frame[columns].to_numpy().copy()
            for idx,col in enumerate(columns):
                if col.startswith('cfs'):continuous[:,idx]=np.sign(continuous[:,idx])*np.log1p(abs(continuous[:,idx]))
            return np.concatenate([continuous,frame[[f'context_{h}' for h in range(12)]].to_numpy(),np.eye(66)[frame.pair.to_numpy()]],1)
        for name,columns in [('controls',controls),('cfs_only',cfs),('controls_cfs',controls+cfs)]:
            for target in ('interaction','magnitude'):
                predicted,coef=ridge_predict(features(train,columns),train[target].to_numpy(),features(test,columns))
                coefficients[f'{name}_{keep}_{target}']={k:v.tolist() if hasattr(v,'tolist') else v for k,v in coef.items()}
                for offset,(_,row) in enumerate(test.iterrows()):
                    prediction_rows.append(dict(method=name,keep=keep,target=target,sample=int(row['sample']),pair=int(row['pair']),
                        actual=float(row[target]),predicted=float(predicted[offset])))
    predicted=pd.DataFrame(prediction_rows)
    predicted['squared_error']=(predicted.predicted-predicted.actual)**2
    predicted['absolute_error']=abs(predicted.predicted-predicted.actual)
    predicted['sign_correct']=(predicted.predicted>=0)==(predicted.actual>=0)
    predicted.to_csv(args.output/'heldout_pair_predictions.csv',index=False)
    per_input=predicted.groupby(['method','keep','target','sample'])[['squared_error','absolute_error','sign_correct']].mean().reset_index()
    per_input.to_csv(args.output/'heldout_input_prediction_errors.csv',index=False)
    perf=[]
    for keys,g in per_input.groupby(['method','keep','target']):
        for metric in ('squared_error','absolute_error','sign_correct'):
            perf.append(dict(method=keys[0],keep=keys[1],target=keys[2],metric=metric,**boot(g[metric])))
    pd.DataFrame(perf).to_csv(args.output/'heldout_prediction_summary.csv',index=False)
    increments=[]
    for (keep,target),g in per_input.groupby(['keep','target']):
        pivot=g.pivot(index='sample',columns='method',values='squared_error')
        increments.append(dict(keep=keep,target=target,contrast='MSE_controls_minus_controls_CFS',
            **boot(pivot.controls-pivot.controls_cfs)))
    pd.DataFrame(increments).to_csv(args.output/'incremental_prediction.csv',index=False)
    (args.output/'regression_coefficients.json').write_text(json.dumps(coefficients,indent=2))
    (args.output/'regression_protocol.json').write_text(json.dumps(dict(fit_inputs=64,test_inputs=128,alpha=1,
        train_split='tune only',test_split='fixed 128 heldout inputs',valid_source_threshold=1e-5,
        controls=controls,cfs_features=cfs,shared_covariates=['background identity','pair identity'],
        bootstrap='2000 independent test inputs; no independent pair resampling'),indent=2))
    print(pd.DataFrame(increments).to_string(index=False))


if __name__=='__main__':main()
