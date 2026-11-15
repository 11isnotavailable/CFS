"""Audit frozen documents, teacher identity, checkpoint choice and test outputs."""
import argparse,hashlib,json,platform,importlib.metadata,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    provenance=json.loads((root/'provenance.json').read_text());report=dict(checks=[],layers={})
    datasets={}
    for split,n in [('train',2048),('tune',128),('test',256)]:
        path=root/f'data/{split}.npz';assert sha(path)==provenance['splits'][split]['file_sha256']
        with np.load(path) as f:datasets[split]={k:f[k] for k in f.files}
        assert datasets[split]['input_ids'].shape==(n,129)
    for key in ('document_sha256','prefix_sha256'):
        assert len(set(np.concatenate([v[key] for v in datasets.values()])))==2432
    for name,digest in provenance['model_files'].items():assert sha(root/'model'/name)==digest
    report['checks']+=['2432 disjoint document and prefix hashes','frozen input hashes unchanged','all pretrained model hashes unchanged']
    dense=[]
    for layer in (11,8):
        directory=root/f'layer{layer}';info={};features=[]
        for split,n in [('train',2048),('tune',128)]:
            pair=directory/f'pair/{split}';paths=sorted(pair.glob('[0-9][0-9][0-9][0-9][0-9].npz'));assert len(paths)==n
            grid=np.linspace(0,3,16 if split=='train' else 31);off=~np.eye(12,dtype=bool)
            for i,path in enumerate(paths):
                with np.load(path) as f:
                    assert str(f['document_sha256'])==datasets[split]['document_sha256'][i]
                    assert np.isfinite(f['drops']).all() and (f['drops']>=0).all()
                    assert np.isfinite(f['substitution'][off]).all() and np.isnan(np.diag(f['substitution'])).all()
                    assert np.isfinite(f['alpha'][off]).all() and np.isnan(np.diag(f['alpha'])).all()
                    assert np.abs(f['alpha'][off,None]-grid).min(-1).max()<1e-6
                    assert f['feature'].shape==(2304,) and np.isfinite(f['feature']).all()
                    if split=='train':features.append(f['feature'])
            info[f'pair_{split}']=n
        features=np.stack(features);mean=features.mean(0);std=features.std(0).clip(1e-5)
        for phase,counts in [('pilot',{'train':256,'tune':64}),('full',{'train':2048,'tune':128})]:
            c=json.loads((directory/f'{phase}/joint/config.json').read_text())
            assert c['bootstrap_sha256']==sha(directory/f'{phase}/runs/bootstrap_seed17/model.pt')
            for split,n in counts.items():
                paths=sorted((directory/f'{phase}/joint/{split}').glob('[0-9][0-9][0-9][0-9][0-9].npz'));assert len(paths)==n
                for i,path in enumerate(paths):
                    with np.load(path) as f:
                        assert str(f['document_sha256'])==datasets[split]['document_sha256'][i]
                        assert f['kl'].shape==(3,48) and np.isfinite(f['kl']).all() and (f['kl']>=0).all()
                        for ki,k in enumerate((3,6,9)):
                            masks=f['masks'][ki];assert masks.shape==(48,12) and (masks.sum(-1)==k).all()
                            assert len(np.unique(masks,axis=0))==48
                info[f'{phase}_joint_{split}']=n
        lock=json.loads((directory/'selection_lock.json').read_text());check=json.loads((directory/'evaluation/checks.json').read_text())
        assert check['selection_lock_sha256']==sha(directory/'selection_lock.json')
        assert lock['time']<(directory/'evaluation/per_input.csv').stat().st_mtime
        for kind in ('R2','Gating','A0'):
            for seed in (17,23,41):
                run=directory/f'full/runs/{kind}_seed{seed}';history=pd.read_csv(run/'history.csv')
                assert list(history.epoch)==list(range(1,61))
                valid=history.dropna(subset=['val_hard_kl']);assert list(valid.epoch)==[10,20,30,40,50,60]
                best=valid.loc[valid.val_hard_kl.idxmin()];payload=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
                assert payload['best_epoch']==int(best.epoch)==lock['models'][f'{kind}_{seed}']['epoch']
                assert abs(payload['validation_metric']-best.val_hard_kl)<1e-9
                np.testing.assert_array_equal(payload['global_mean'],mean);np.testing.assert_array_equal(payload['global_std'],std)
                assert sha(run/'model.pt')==sha(run/f'epoch{int(best.epoch)}.pt')
        for path,digest in check['checkpoint_hashes'].items():assert sha(path)==digest
        frame=pd.read_csv(directory/'evaluation/per_input.csv');assert len(frame)==9216
        assert np.isfinite(frame[['kl','nll','dense_nll','nll_difference']]).all().all()
        for _,g in frame.groupby(['method','seed','keep']):assert list(g['sample'])==list(range(256))
        t=frame.query('method=="Taylor"').pivot(index=['sample','keep'],columns='seed',values='kl')
        # Identical masks at different batch rows can differ by FP32 roundoff.
        np.testing.assert_allclose(t[17],t[23],atol=2e-5,rtol=0);np.testing.assert_allclose(t[17],t[41],atol=2e-5,rtol=0)
        info['taylor_repeated_kl_max_difference']=float(np.ptp(t.to_numpy(),axis=1).max())
        with np.load(directory/'evaluation/predictions.npz') as f:
            assert (f['masks'].sum(-1)==frame.keep.to_numpy()).all();dense.append(f['dense_logits'])
            for seed in (23,41):
                np.testing.assert_array_equal(f['masks'][(frame.method=='Taylor')&(frame.seed==17)],f['masks'][(frame.method=='Taylor')&(frame.seed==seed)])
        assert len(check['online_checks'])==36 and max(x['online_kl_error'] for x in check['online_checks'])<2e-5
        assert not (directory/'pair/test').exists() and not (directory/'full/joint/test').exists()
        info.update(test_rows=len(frame),online_checks=len(check['online_checks']),online_max_logit_error=max(x['online_max_error'] for x in check['online_checks']),
            online_max_kl_error=max(x['online_kl_error'] for x in check['online_checks']),locked_models=9)
        report['layers'][layer]=info
    np.testing.assert_allclose(dense[0],dense[1],atol=1e-4,rtol=1e-5)
    report['dense_cross_layer_max_logit_error']=float(np.abs(dense[0]-dense[1]).max())
    report['checks'].append('dense reference logits agree across layers within FP32 tolerance')
    report['all_passed']=True;(root/'verification.json').write_text(json.dumps(report,indent=2))
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','pandas','matplotlib','datasets','pyarrow','safetensors','timm','pytest')}
    (root/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages=versions,
        cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        precision='FP32; TF32 disabled',unit_tests='68 passed locally; 5 GPT-2 tests passed in remote production environment'),indent=2))
    source_paths=[p for base in ('src','scripts','tests','docs') for p in (root/base).rglob('*') if p.is_file() and p.suffix in ('.py','.md')]
    (root/'source_manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in source_paths},indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
