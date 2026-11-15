"""Audit frozen weights, train-only splits, locked policies and real evaluations."""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.higher_order_residual import SetResidual


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path,value in json.loads((root/'source_hashes.json').read_text()).items():assert sha(path)==value
    counts=[];updates=0
    for kind in ('plain','cfs','shuffled'):
        for seed in (17,23,41):
            run=root/f'runs/{kind}_seed{seed}';config=json.loads((run/'config.json').read_text())
            assert config['epochs']==20 and not config['test_loaded']
            assert len(list(run.glob('epoch_*.pt')))==20
            history=pd.read_csv(run/'history.csv');assert history.epoch.tolist()==list(range(1,21))
            assert np.isfinite(history.select_dtypes('number')).all().all()
            counts.append(config['parameters']);updates+=20*64
            c=np.load(root/f'cache/train_seed{seed}.npz');assert c['encoded'].shape==(4096,256)
            assert np.all(c['calibration'][:,1]>0)
            model=SetResidual(kind);payload=torch.load(root/f'selected/{kind}_seed{seed}/model.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(payload['state_dict']);assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    assert len(set(counts))==1
    split=np.load(root/'cache/mask_split.npz');assert split['fit'].shape==(4096,3,36) and split['held'].shape==(4096,3,12)
    joined=np.concatenate((split['fit'],split['held']),-1);np.testing.assert_array_equal(np.sort(joined,-1),np.broadcast_to(np.arange(48),joined.shape))
    manifest=json.loads((root/'confirmation/manifest.json').read_text());assert manifest['count']==256 and not manifest['model_outputs_computed']
    confirm_paths=set()
    for row in manifest['images']:
        assert sha(root/'confirmation/images'/row['path'])==row['sha256'];confirm_paths.add(row['path'])
    for s in ('train','tune','test'):
        old=np.load(root/f'baseline/learned_gating_inputs_20261115/{s}.npz')
        canonical={str(v).replace('\\','/').split('imagenette2-160/')[-1] for v in old['paths']}
        assert not canonical & confirm_paths
    lock=json.loads((root/'selection_lock.json').read_text());assert not lock['test_used'] and not lock['confirmation_used']
    assert (root/'selection_lock.json').stat().st_mtime<(root/'confirmation/features.npz').stat().st_mtime
    assert (root/'docs/HIGHER_ORDER_RESIDUAL_PRETEST_AMENDMENT.md').stat().st_mtime<(root/'confirmation/features.npz').stat().st_mtime
    checks=[];negative=[]
    for cohort,n in [('test',448),('confirmation',256)]:
        frame=pd.read_csv(root/f'eval_{cohort}/per_input.csv');assert len(frame)==n*3*9*3
        assert not frame.duplicated(['sample','seed','method','keep']).any()
        assert np.isfinite(frame.kl).all()
        masks=np.load(root/f'eval_{cohort}/masks.npz')
        for key in masks.files:assert np.all(masks[key].sum(-1)==int(key.rsplit('_',1)[-1]))
        for kind,selection in lock['variants'].items():
            for ki,k in enumerate((3,6,9)):
                if selection['gamma'][ki]==0:
                    for seed in (17,23,41):np.testing.assert_array_equal(masks[f'{seed}_{kind}_{k}'],masks[f'{seed}_A3_{k}'])
        preds=np.load(root/f'eval_{cohort}/candidate_predictions.npz')
        for key in preds.files:
            v=preds[key];assert np.isfinite(v).all()
            negative.append(dict(cohort=cohort,predictor=key,minimum=float(v.min()),negative_fraction=float((v<0).mean())))
        check=json.loads((root/f'eval_{cohort}/checks.json').read_text());assert check['dense_reference_exact'];checks.append(check)
    pd.DataFrame(negative).to_csv(root/'analysis/prediction_range_audit.csv',index=False)
    result=dict(runs=9,epochs_per_run=20,updates_total=updates,added_parameters=counts[0],total_parameters=825091+counts[0],
        frozen_source_hashes_unchanged=True,residual_fit_inputs=4096,fit_masks_per_budget=36,held_masks_per_budget=12,
        validation_inputs=64,old_test_inputs=448,confirmation_inputs=256,confirmation_path_disjoint=True,
        confirmation_hashes_checked=True,selection_precedes_confirmation=True,amendment_precedes_confirmation=True,
        primary=lock['primary'],backbone_trained=False,old_quartets_used_for_fit=False,evaluations=checks)
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
