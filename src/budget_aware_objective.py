"""Budget reweighting and validation-only constrained checkpoint selection."""
import numpy as np


def weighted_objective(terms, weight9):
    if weight9 <= 0:
        raise ValueError('weight9 must be positive')
    joint=(terms['joint_K3']+terms['joint_K6']+weight9*terms['joint_K9'])/(2+weight9)
    return .5*(joint+.01*(terms['cfs']+.1*terms['alpha']))


def select_validation(frame):
    """frame contains one row per weight, seed, epoch and no held-out outcomes."""
    baseline=frame[frame.epoch==0].groupby('seed')[['val_K3','val_K6','val_K9']].mean().mean()
    grouped=frame[frame.epoch>0].groupby(['weight9','epoch'])[['val_K3','val_K6','val_K9']].mean().reset_index()
    grouped['mean_regret']=grouped[['val_K3','val_K6','val_K9']].mean(axis=1)
    grouped['eligible']=(grouped.val_K3<=baseline.val_K3+1e-8)&(grouped.val_K6<=baseline.val_K6+1e-8)&(grouped.val_K9<baseline.val_K9-1e-8)
    order=['val_K9','mean_regret','weight9','epoch']
    qualified=grouped[grouped.eligible].sort_values(order)
    def record(row):
        return {k:(v.item() if isinstance(v,np.generic) else v) for k,v in row.to_dict().items()}
    primary=record(qualified.iloc[0]) if len(qualified) else None
    representatives={}
    for weight,g in grouped.groupby('weight9'):
        feasible=g[g.eligible]
        representatives[f'{weight:g}']=record((feasible if len(feasible) else g).sort_values(order).iloc[0])
    return dict(baseline_tune=baseline.to_dict(),primary=primary,representatives=representatives,
                fallback_to_A3=primary is None,test_used=False),grouped
