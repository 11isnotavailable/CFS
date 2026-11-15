"""Freeze label-independent pools and fresh confirmation inputs, no inference."""
import argparse,hashlib,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.coalitional_cfs import candidate_pool


def canonical(path):return str(path).replace('\\','/').split('imagenette2-160/')[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--images',type=Path,required=True)
    p.add_argument('--old',type=Path,default=Path('/data/cfs_learned_gating_20261115/learned_gating_inputs_20261115'))
    p.add_argument('--confirmation',type=Path,default=Path('/data/cfs_residual_20261115/confirmation/manifest.json'))
    p.add_argument('--checkpoint',type=Path,default=Path('/data/cfs_learned_gating_20261115/base.pth'));a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);out=a.root/'data';out.mkdir(exist_ok=False)
    old=a.old;excluded=set();hashes=set();targets={};old_data={}
    for split in ('train','tune','test'):
        with np.load(old/f'{split}.npz') as f:old_data[split]={k:f[k] for k in ('paths','target')}
        for raw,target in zip(old_data[split]['paths'],old_data[split]['target']):
            name=canonical(raw);excluded.add(name);targets[Path(name).parent.name]=int(target)
            hashes.add(hashlib.sha256((a.images/name).read_bytes()).hexdigest())
    historical=json.loads(a.confirmation.read_text())
    for item in historical['images']:excluded.add(item['path']);hashes.add(item['sha256'])
    selected={};rng=np.random.default_rng(20261116)
    selected['train']=[canonical(old_data['train']['paths'][i]) for i in sorted(rng.choice(len(old_data['train']['paths']),1024,replace=False))]
    selected['tune']=[canonical(x) for x in old_data['tune']['paths']]
    eligible=[]
    for path in sorted(a.images.glob('val/*/*.JPEG')):
        name=path.relative_to(a.images).as_posix();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if name not in excluded and digest not in hashes:eligible.append(name)
    assert len(eligible)>=256
    selected['test']=[eligible[i] for i in sorted(rng.choice(len(eligible),256,replace=False))]
    allhash=[];manifest=dict(seed=20261116,excluded_historical_paths=len(excluded),fresh_test_eligible=len(eligible),
        note='New to this routing series; not a claim of never-used public images in all research',splits={})
    for split,names in selected.items():
        h=[hashlib.sha256((a.images/n).read_bytes()).hexdigest() for n in names];allhash+=h
        for name in names:
            dest=a.root/'images'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a.images/name,dest)
        np.savez_compressed(out/f'{split}.npz',paths=np.array(names),sha256=np.array(h),target=np.array([targets[Path(n).parent.name] for n in names]))
        manifest['splits'][split]=dict(count=len(names),file_sha256=hashlib.sha256((out/f'{split}.npz').read_bytes()).hexdigest())
    assert len(set(allhash))==len(allhash)
    manifest['image_root']=str(a.images);manifest['checkpoint']=str(a.checkpoint)
    manifest['backbone_sha256']=hashlib.sha256(Path(manifest['checkpoint']).read_bytes()).hexdigest()
    manifest['pools']={str(seed):candidate_pool(seed).tolist() for seed in (42,123,2027)}
    (a.root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='pools'},indent=2))


if __name__=='__main__':main()
