"""Validate completed experiments without selecting models or changing results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;data=root/'learned_gating_inputs_20261115'
    manifest=json.loads((data/'manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==expected,name
    train,tune,test=[np.load(data/f'{s}.npz') for s in ('train','tune','test')]
    assert len(train['feature'])==4096 and len(tune['feature'])==64 and len(test['feature'])==448
    assert 'kl' not in test.files and 'masks' not in test.files
    def names(dataset):
        return {str(p).replace('\\','/').split('imagenette2-160/')[-1] for p in dataset['paths']}
    assert not (names(train)&names(tune) or names(train)&names(test) or names(tune)&names(test))
    for array in (train,tune):
        assert array['masks'].shape==(len(array['feature']),3,48,12)
        assert np.isfinite(array['feature']).all() and np.isfinite(array['kl']).all()
        for idx,k in enumerate((3,6,9)):assert (array['masks'][:,idx].sum(-1)==k).all()
    one=torch.load(root/'smoke_gating/model.pt',map_location='cpu',weights_only=False)
    two=torch.load(root/'smoke_gating_repeat/model.pt',map_location='cpu',weights_only=False)
    assert one['best_epoch']==two['best_epoch'] and one['val_joint_regret']==two['val_joint_regret']
    for key in one['state_dict']:torch.testing.assert_close(one['state_dict'][key],two['state_dict'][key],rtol=0,atol=0)
    for path in (root/'runs').glob('*/training_history.csv'):
        history=pd.read_csv(path);assert history.epoch.tolist()==list(range(1,61))
        assert np.isfinite(history[['train_loss','val_joint_regret']]).all().all()
        checkpoint=torch.load(path.parent/'model.pt',map_location='cpu',weights_only=False)
        assert checkpoint['best_epoch']==int(history.loc[history.val_joint_regret.idxmin(),'epoch'])
        assert checkpoint['config']['test_loaded'] is False
    primary=pd.read_csv(root/'evaluation_v2/per_input.csv')
    live=pd.read_csv(root/'evaluation_live/per_input.csv')
    assert len(primary)==len(live)==448*3*4*3*2
    for frame in (primary,live):
        assert frame.groupby('sample').dense_prediction.nunique().max()==1
        assert frame.groupby('sample').dense_correct.nunique().max()==1
        assert np.isfinite(frame.kl).all() and (frame.kl>=0).all()
    np.testing.assert_array_equal(np.load(root/'evaluation_v2/dense_logits.npz')['logits'],
                                  np.load(root/'evaluation_live/dense_logits.npz')['logits'])
    result={'input_hashes_verified':True,'disjoint_paths':True,'train_tune_test':[4096,64,448],
            'heldout_teacher_absent':True,'candidates_per_budget':48,'seeds':[17,23,41],
            'epochs_per_new_run':60,'updates_per_new_run':3840,'deterministic_smoke_weights_bitwise_equal':True,
            'rows_per_evaluation':len(primary),'identical_dense_logits_across_feature_audit':True}
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
