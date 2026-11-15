"""Input-cluster metrics; no relation is treated as an independent sample."""
import numpy as np
from scipy.stats import rankdata


def input_metrics(truth,prediction,valid,constant):
    truth=np.asarray(truth,float);prediction=np.asarray(prediction,float);valid=np.asarray(valid,bool)
    C=truth.shape[-1];idx=prediction.argmax(-1);chosen=np.take_along_axis(truth,idx[...,None],-1)[...,0]
    oracle=truth.max(-1);random=truth.mean(-1)
    yr=rankdata(truth,axis=-1);pr=rankdata(prediction,axis=-1)
    yr-=yr.mean(-1,keepdims=True);pr-=pr.mean(-1,keepdims=True)
    norm=np.sqrt((yr*yr).sum(-1)*(pr*pr).sum(-1));rho=np.divide((yr*pr).sum(-1),norm,out=np.zeros_like(norm),where=norm>0)
    yt=np.argsort(-truth,axis=-1,kind='stable')[...,:2];pt=np.argsort(-prediction,axis=-1,kind='stable')[...,:2]
    recall=(yt[...,None]==pt[...,None,:]).any(-1).mean(-1)
    raw=dict(oracle=oracle,selected=chosen,regret=oracle-chosen,random_recovery=random,random_regret=oracle-random,
        spearman=rho,undefined_spearman=norm==0,best_hit=np.isclose(chosen,oracle,atol=1e-8,rtol=0),top2_recall=recall,
        random_best_hit=np.isclose(truth,oracle[...,None],atol=1e-8,rtol=0).mean(-1),random_top2_recall=np.full_like(oracle,2/C),
        mse=((prediction-truth)**2).mean(-1),constant_mse=((truth-constant)**2).mean(-1),
        prediction_below_zero=(prediction<0).mean(-1),prediction_above_one=(prediction>1).mean(-1))
    count=valid.sum(-1)
    return {key:np.divide(np.where(valid,value,0).sum(-1),count,out=np.full(len(count),np.nan),where=count>0) for key,value in raw.items()}


def bootstrap_ratio(numerator,denominator,draws):
    x=np.asarray(numerator,float);y=np.asarray(denominator,float)
    values=x[draws].mean(-1)/y[draws].mean(-1)
    return float(x.mean()/y.mean()),np.quantile(values,[.025,.975])
