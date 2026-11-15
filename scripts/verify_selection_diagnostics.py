"""Validate frozen-only execution, known-label coverage and oracle upper bounds."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.selection_diagnostics import mask_codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve();d=root/'collected'
    meta=json.loads((d/'metadata.json').read_text());assert all(not meta[k] for k in ('training_run','backbone_run','new_labels_collected','test_loaded','confirmation_loaded'))
    for path,sha in json.loads((d/'source_hashes.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    raw=pd.read_csv(d/'per_input.csv');cov=pd.read_csv(d/'coverage.csv');cross=pd.read_csv(d/'cross_method.csv')
    assert len(raw)==(4096+64)*4*3*3 and len(cov)==len(raw) and len(cross)==(4096+64)*3*3
    assert not raw.duplicated(['split','sample','method','seed','keep']).any()
    assert np.allclose(raw.regret,raw.chosen_kl-raw.oracle_kl,atol=1e-12)
    assert (raw.regret>=-1e-12).all();assert np.allclose(raw.near_regret+raw.far_regret,raw.regret)
    previous=raw.regret
    for size in (1,3,5,10,20):
        current=raw[f'top{size}_regret'];assert (current<=previous+1e-12).all() and (current>=-1e-12).all();previous=current
    for size in (1,3,5):
        for method in ('A3','R2','Gating'):assert (cross[f'union_top{size}_kl']<=cross[f'{method}_kl']+1e-12).all()
    matches=[]
    for split,n in [('train',4096),('tune',64)]:
        label=np.load(d/f'labels_{split}.npz');assert len(label['paths'])==n;codes=mask_codes(label['masks'])
        for seed in (17,23,41):
            for method in ('A3','R2','Gating','A0'):
                pred=np.load(d/f'predictions/{split}_{method}_seed{seed}.npz');assert pred['pool_utility'].shape==(n,3,48)
                for ki,k in enumerate((3,6,9)):
                    g=cov.query('split==@split and seed==@seed and method==@method and keep==@k').sort_values('sample')
                    known=(codes[:,ki]==pred['full_choice_code'][:,ki,None]).any(-1)
                    np.testing.assert_array_equal(known,g.full_choice_known)
                    assert g.loc[~known,'full_choice_kl'].isna().all()
                if split=='tune' and method!='R2':
                    path=root/f'source/runs/A3_lambda0.01_seed{seed}/model.pt' if method=='A3' else root/f'baseline/runs/{"gating" if method=="Gating" else "joint_only_r2"}_seed{seed}/model.pt'
                    payload=torch.load(path,map_location='cpu',weights_only=False)
                    measured=raw.query('split==@split and seed==@seed and method==@method').regret.mean()
                    error=abs(measured-payload['val_joint_regret']);assert error<1e-6
                    matches.append(dict(method=method,seed=seed,checkpoint_regret_difference=float(error)))
    inv=pd.read_csv(d/'training_candidate_inventory.csv');assert len(inv)==4096*3*3*2*3
    assert inv.loc[~inv.label_known,'measured_kl'].isna().all()
    queue=pd.read_csv(root/'analysis/unmeasured_training_candidate_queue.csv');assert not queue.label_known.any()
    assert not queue.duplicated(['sample','keep','mask_code']).any()
    checks=json.loads((d/'checks.json').read_text())
    result=dict(inputs_train=4096,inputs_tune=64,methods=4,seeds=3,budgets=3,metric_rows=len(raw),
        candidate_inventory_rows=len(inv),unmeasured_queued_candidates=len(queue),source_hashes_unchanged=True,
        training=False,new_teacher=False,backbone_inference=False,test_or_confirmation_loaded=False,
        pool_full_score_max_error=max(r['pool_vs_full_score_max_error'] for r in checks),checkpoint_validation_matches=matches,
        all_original_routers_already_have_listwise_joint_loss=True)
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
