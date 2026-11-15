"""Labelled-pool ranking diagnostics with explicit ties and oracle bounds."""
import numpy as np
from scipy.stats import rankdata


def mask_codes(masks):
    masks=np.asarray(masks)
    return (masks.astype(np.int64)*(1<<np.arange(masks.shape[-1]))).sum(-1)


def ranking_metrics(utility,kl,tolerance=1e-8):
    u=np.asarray(utility,dtype=np.float64);y=np.asarray(kl,dtype=np.float64)
    if u.ndim!=2 or u.shape!=y.shape or not np.isfinite(u).all() or not np.isfinite(y).all():
        raise ValueError('Finite matching [input,candidate] arrays required')
    B,C=u.shape;n=np.arange(B);order=np.argsort(-u,axis=-1,kind='stable');truth=np.argsort(y,axis=-1,kind='stable')
    chosen=order[:,0];oracle=y.min(-1);selected=y[n,chosen];regret=selected-oracle
    rank=1+(y<selected[:,None]-tolerance).sum(-1)
    result=dict(chosen=chosen,chosen_kl=selected,oracle_kl=oracle,regret=regret,
        chosen_true_rank=rank,pool_range=y.max(-1)-oracle,
        normalized_regret=regret/np.maximum(y.max(-1)-oracle,1e-12),
        exact_best=(regret<=tolerance).astype(float),chosen_true_top5=(rank<=5).astype(float),
        near_error=((regret>tolerance)&(rank<=5)).astype(float),far_error=(rank>5).astype(float),
        near_regret=regret*(rank<=5),far_regret=regret*(rank>5),
        max_score_ties=(u==u.max(-1,keepdims=True)).sum(-1))
    ranks_u=rankdata(-u,axis=-1);ranks_y=rankdata(y,axis=-1)
    a=ranks_u-ranks_u.mean(-1,keepdims=True);b=ranks_y-ranks_y.mean(-1,keepdims=True)
    denominator=np.sqrt((a*a).sum(-1)*(b*b).sum(-1))
    result['spearman']=np.divide((a*b).sum(-1),denominator,out=np.full(B,np.nan),where=denominator>0)
    def agreement(uu,yy):
        i,j=np.triu_indices(uu.shape[1],1);gap=yy[:,j]-yy[:,i];delta=uu[:,i]-uu[:,j]
        valid=abs(gap)>tolerance;credit=(delta*gap>0).astype(float)+.5*(delta==0)
        count=valid.sum(-1);mass=(abs(gap)*valid).sum(-1)
        acc=np.divide((credit*valid).sum(-1),count,out=np.full(B,np.nan),where=count>0)
        weighted=np.divide((credit*abs(gap)*valid).sum(-1),mass,out=np.full(B,np.nan),where=mass>0)
        return acc,weighted
    result['pair_accuracy'],result['gap_weighted_pair_accuracy']=agreement(u,y)
    near=truth[:,:min(5,C)];near_u=np.take_along_axis(u,near,-1);near_y=np.take_along_axis(y,near,-1)
    result['top5_pair_accuracy'],result['top5_gap_weighted_accuracy']=agreement(near_u,near_y)
    best=truth[:,0];gap=y-oracle[:,None];valid=gap>tolerance
    delta=u[n,best,None]-u;credit=(delta>0).astype(float)+.5*(delta==0)
    count=valid.sum(-1)
    result['best_vs_rest_accuracy']=np.divide((credit*valid).sum(-1),count,out=np.full(B,np.nan),where=count>0)
    for size in (1,3,5,10,20):
        m=min(size,C);loss=np.take_along_axis(y,order[:,:m],-1).min(-1)
        result[f'top{size}_oracle_kl']=loss;result[f'top{size}_regret']=loss-oracle
        result[f'top{size}_retrieval']=(loss-oracle<=tolerance).astype(float)
    return result,order


def union_oracle(utilities,kl,size):
    y=np.asarray(kl,dtype=np.float64)
    indices=np.concatenate([np.argsort(-np.asarray(u),axis=-1,kind='stable')[:,:size] for u in utilities],axis=-1)
    return np.take_along_axis(y,indices,-1).min(-1)
