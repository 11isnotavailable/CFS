"""Consistency checks for frozen selections, measured quartets and routing."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.subset_interactions import quartet_interactions,interaction_masks


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    lock=json.loads((root/'selection_lock.json').read_text());pilot=pd.read_csv(root/'pilot_validation.csv')
    assert len(pilot)==15 and lock['test_used'] is False
    assert lock['headline']==min(lock['selected'],key=lambda v:lock['selected'][v]['val_joint_regret'])
    runs=list((root/'runs').glob('*/complete.json'));assert len(runs)==21
    for path in runs:
        history=pd.read_csv(path.parent/'training_history.csv');assert len(history)==60
        state=torch.load(path.parent/'model.pt',map_location='cpu',weights_only=False)
        assert state['best_epoch']==int(history.loc[history.val_joint_regret.idxmin(),'epoch'])
        assert state['config']['test_loaded'] is False
    masks_total=0
    for path in (root/'interactions').glob('*.npz'):
        data=np.load(path);sample=int(data['sample']);split=str(data['split'])
        masks,pairs,q,random_ids=interaction_masks(42+sample+(10000 if split=='test' else 0))
        for key,value in [('masks',masks),('pairs',pairs),('quartets',q),('random_ids',random_ids)]:np.testing.assert_array_equal(data[key],value)
        np.testing.assert_array_equal(data['interaction'],quartet_interactions(data['kl'],q));masks_total+=len(masks)
        assert np.isfinite(data['kl']).all()
    assert len(list((root/'interactions').glob('tune_*.npz')))==64
    assert len(list((root/'interactions').glob('test_*.npz')))==128
    routing=pd.read_csv(root/'routing_per_input.csv')
    assert len(routing)==448*3*7*3*2
    assert routing.groupby('sample').dense_prediction.nunique().max()==1
    assert np.isfinite(routing.kl).all()
    baseline=pd.read_csv(root/'baseline/evaluation_v2/per_input.csv').replace({'method':{'R2_joint_only':'A0'}})
    keys=['sample','seed','method','keep','compensation']
    compare=routing.merge(baseline,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    diff=float(abs(compare.kl_new-compare.kl_old).max())
    result=dict(pilot_runs=15,total_new_runs=21,epochs_per_run=60,headline=lock['headline'],
        selected_strengths={k:v['strength'] for k,v in lock['selected'].items()},
        test_not_used_for_selection=True,measured_inputs=192,unique_mask_interventions=masks_total,
        measured_quartets=192*3*66,routing_rows=len(routing),baseline_max_abs_kl_difference=diff,
        bootstrap_unit='independent input; seeds averaged per input',extra_teacher_recomputed=False)
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
