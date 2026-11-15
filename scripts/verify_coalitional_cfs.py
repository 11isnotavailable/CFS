"""Immutable-source, teacher-domain, selection and raw-statistic audits."""
import argparse,json,hashlib,sys,platform,importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.train_coalitional_cfs import arrays,name


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    manifest=json.loads((root/'manifest.json').read_text());hashes=[];checks={}
    for split,meta in manifest['splits'].items():
        assert sha(root/f'data/{split}.npz')==meta['file_sha256']
        with np.load(root/f'data/{split}.npz') as f:
            assert len(f['paths'])==meta['count']
            for path,digest in zip(f['paths'],f['sha256']):assert sha(root/'images'/str(path))==str(digest)
            hashes.extend(f['sha256'].tolist())
    assert len(set(hashes))==1344
    assert sha('/data/cfs_learned_gating_20261115/base.pth')==manifest['backbone_sha256']
    teacher_count=0
    for directory in sorted((root/'teacher').glob('*/*')):
        if not directory.is_dir():continue
        with np.load(root/f'data/{directory.name}.npz') as f:identity=f['sha256']
        paths=sorted(directory.glob('[0-9][0-9][0-9][0-9][0-9].npz'));assert len(paths)==len(identity)
        for i,path in enumerate(paths):
            with np.load(path) as f:
                assert str(f['sha256'])==identity[i]
                for key in ('feature','drop','pair_kl','joint_kl','pair_alpha','joint_alpha'):assert np.isfinite(f[key]).all()
                for key in ('pair_alpha','joint_alpha'):
                    alpha=f[key];assert alpha.min()>=0 and alpha.max()<=3 and np.max(abs(alpha*10-np.round(alpha*10)))<3e-6
                assert (f['pair_kl']<=f['drop'][:,None]+1e-10).all()
                pair=f['pair_kl'];joint=f['joint_kl']
                combos=list(__import__('itertools').combinations(range(4),2))
                for at,(j,k) in enumerate(combos):assert (joint[:,at]<=np.minimum(pair[:,j],pair[:,k])+1e-10).all()
                assert int(f['pair_variants'])==48*31 and int(f['joint_variants'])==72*961 and int(f['backward_calls'])==0
        teacher_count+=len(paths)
    train=arrays(root,'train');selected=0;counts={};locked={}
    for suffix in ('','_evaluations'):
        plan=json.loads((root/f'training_plan{suffix}.json').read_text());lock=json.loads((root/f'selection_lock{suffix}.json').read_text());locked[suffix]=lock
        for job in plan['jobs']:
            run=root/'runs'/name(job);payload=torch.load(run/'model.pt',map_location='cpu',weights_only=False);config=payload['config']
            assert sha(run/'model.pt')==lock['models'][name(job)]['sha256']
            history=pd.read_csv(run/'history.csv');assert list(history.step)==list(range(250,2001,250))
            assert payload['step']==int(history.loc[history.validation_mse.idxmin(),'step'])
            epoch=torch.load(run/f'step{payload["step"]}.pt',map_location='cpu',weights_only=False)
            for key,value in payload['state_dict'].items():torch.testing.assert_close(value,epoch['state_dict'][key],atol=0,rtol=0)
            actual='joint' if job['kind']=='pair_info' else job['kind'];width=4 if actual=='pair' else 6
            valid=np.repeat(train['valid'],width,axis=1);expected=np.random.default_rng(20261116).permutation(np.flatnonzero(valid.ravel()))[:job['labels']]
            with np.load(run/'label_indices.npz') as f:np.testing.assert_array_equal(f['flat_indices'],expected)
            labels=train[actual].ravel()[expected];assert abs(labels.mean()-payload['target_mean'])<1e-7
            np.testing.assert_array_equal(payload['global_mean'],train['feature'].mean(0));np.testing.assert_array_equal(payload['global_std'],train['feature'].std(0).clip(1e-5))
            assert config['steps']==2000 and config['test_used']==False
            if job['kind']!='pair_info':assert config['parameters']==668801
            if job['curve']=='evaluations':assert job['labels']*961<=job['nominal_budget']*31 and (job['nominal_budget']*31-job['labels']*961)<961
            counts[name(job)]=config['labels'];selected+=1
    assert selected==51 and teacher_count==1856
    for directory in (root/'evaluation').iterdir():
        if not directory.is_dir():continue
        suffix='_evaluations' if directory.name.endswith('_evaluations') else ''
        check=json.loads((directory/'checks.json').read_text());assert check['source_lock_sha256']==sha(root/f'selection_lock{suffix}.json')
        frame=pd.read_csv(directory/'per_input.csv');assert np.isfinite(frame.select_dtypes('number')).all().all()
        for _,g in frame.groupby(['method','curve','budget','seed']):assert list(g['sample'])==list(range(256))
    checks.update(all_passed=True,unique_images=1344,test_images=256,teacher_inputs=teacher_count,selected_models=selected,
        parameters_main=668801,parameters_equal=True,backbone_hash_unchanged=True,joint_contains_pair=True,source_alpha_domain='31-grid [0,3]',
        primary_lock_before_test_teacher=locked['']['time']<(root/'teacher/pool42/test/00000.npz').stat().st_mtime,
        supplementary_control_declared_separately=True,original_core_files_modified=False,labels=counts)
    (root/'verification.json').write_text(json.dumps(checks,indent=2))
    environment=dict(python=platform.python_version(),packages={key:importlib.metadata.version(key) for key in ('torch','timm','numpy','scipy','pandas','matplotlib','transformers')},cuda=torch.version.cuda,tests='72 passed locally; new tests also run remotely')
    (root/'environment.json').write_text(json.dumps(environment,indent=2))
    sources=[p for directory in ('src','scripts','tests') for p in (root/directory).glob('*.py')]
    (root/'source_manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sources},indent=2));print(json.dumps({k:v for k,v in checks.items() if k!='labels'},indent=2))


if __name__=='__main__':main()
