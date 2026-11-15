"""Recheck historical path/content exclusion where the original image bank lives."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--confirmation',type=Path,required=True);a=p.parse_args()
    paths=set();hashes=set();sources={}
    for split in ('train','tune','test'):
        path=a.old/f'{split}.npz';sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            for raw in f['paths']:
                name=str(raw).replace('\\','/').split('imagenette2-160/')[-1];paths.add(name);hashes.add(hashlib.sha256((a.images/name).read_bytes()).hexdigest())
    for row in json.loads(a.confirmation.read_text())['images']:paths.add(row['path']);hashes.add(row['sha256'])
    with np.load(a.root/'data/test.npz') as f:
        assert not set(f['paths'])&paths and not set(f['sha256'])&hashes
    report=dict(all_passed=True,excluded_paths=len(paths),excluded_content_hashes=len(hashes),test_images=256,old_npz_hashes=sources,
        confirmation_manifest_sha256=hashlib.sha256(a.confirmation.read_bytes()).hexdigest(),scope='original routing cohorts plus later256 confirmation images')
    (a.root/'input_exclusion_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
