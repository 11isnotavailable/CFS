"""Lock unseen routing confirmation images without running any model."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np


def canonical(path):
    return str(path).replace('\\','/').split('imagenette2-160/')[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    excluded=set();targets={};hashes=set()
    for split in ('train','tune','test'):
        a=np.load(args.data/f'{split}.npz')
        for raw,target in zip(a['paths'],a['target']):
            path=canonical(raw);excluded.add(path);targets[Path(path).parent.name]=int(target)
            hashes.add(hashlib.sha256((args.images/path).read_bytes()).hexdigest())
    all_paths=sorted(args.images.glob('val/*/*.JPEG'))
    eligible=[]
    for idx,path in enumerate(all_paths):
        relative=path.relative_to(args.images).as_posix()
        if relative in excluded:continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in hashes:eligible.append((idx,relative,digest))
    rng=np.random.default_rng(20261115);selected=sorted(rng.choice(len(eligible),256,replace=False))
    rows=[eligible[i] for i in selected]
    for _,relative,_ in rows:
        dest=args.output/'images'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.images/relative,dest)
    np.savez_compressed(args.output/'confirmation.npz',paths=np.array([r[1] for r in rows]),
        dataset_index=np.array([r[0] for r in rows]),target=np.array([targets[Path(r[1]).parent.name] for r in rows]))
    metadata=dict(seed=20261115,count=len(rows),old_paths=len(excluded),old_unique_hashes=len(hashes),
        eligible=len(eligible),model_outputs_computed=False,scope='New inputs for this routing series; same Imagenette validation population',
        images=[dict(dataset_index=i,path=path,sha256=digest) for i,path,digest in rows])
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print({k:v for k,v in metadata.items() if k!='images'})


if __name__=='__main__':main()
