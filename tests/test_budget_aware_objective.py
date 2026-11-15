import pandas as pd
import pytest
import torch
from src.budget_aware_objective import weighted_objective,select_validation


def test_uniform_weights_preserve_A3_objective():
    t={f'joint_K{k}':torch.tensor(float(k),requires_grad=True) for k in (3,6,9)}
    t.update(cfs=torch.tensor(2.),alpha=torch.tensor(.2))
    old=.5*((t['joint_K3']+t['joint_K6']+t['joint_K9'])/3+.01*(t['cfs']+.1*t['alpha']))
    torch.testing.assert_close(weighted_objective(t,1),old)
    weighted_objective(t,3).backward()
    assert t['joint_K9'].grad==3*t['joint_K3'].grad
    with pytest.raises(ValueError):weighted_objective(t,0)


def test_validation_constraints_prevent_k9_only_cherrypicking():
    rows=[]
    for seed in (17,23,41):
        for w in (1.,3.):
            rows.append(dict(seed=seed,weight9=w,epoch=0,val_K3=.1,val_K6=.05,val_K9=.01))
            rows.append(dict(seed=seed,weight9=w,epoch=1,val_K3=.09 if w==1 else .11,val_K6=.04,val_K9=.009 if w==1 else .001))
    selected,_=select_validation(pd.DataFrame(rows))
    assert selected['primary']['weight9']==1 and not selected['representatives']['3']['eligible']
    frame=pd.DataFrame(rows);frame.loc[frame.epoch==1,'val_K6']=.06
    selected,_=select_validation(frame)
    assert selected['primary'] is None and selected['fallback_to_A3']
