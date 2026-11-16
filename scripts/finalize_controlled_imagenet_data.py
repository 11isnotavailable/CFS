"""Apply one pretraining byte-overlap exclusion rule to all seven architectures."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--historical',type=Path,default=Path('/data/cfs_rebuttal_20261114')); a=p.parse_args()
    while not (a.root/'packed/PACKED_READY.json').exists(): time.sleep(20)
    shards=sorted((a.root/'train_manifests').glob('*.complete.json'))
    assert len(shards)==310
    records=[json.loads(f.read_text()) for f in shards]
    assert sum(r['images'] for r in records)==1281167
    overlaps=[x for r in records for x in r['validation_content_overlap']]
    excluded={r['path'].encode() for r in overlaps}
    full=a.root/'packed/train_full_index.npy'; index=a.root/'packed/train_index.npy'
    if not full.exists(): index.replace(full)
    values=np.load(full); assert len(values)==1281167
    mask=np.array([x not in excluded for x in values['path']])
    clean=values[mask]; assert len(values)-len(clean)==len(excluded)
    counts=np.bincount(clean['label'],minlength=1000); assert counts.min()>0
    np.save(index,clean)
    provenance=json.loads((a.historical/'imagenet_provenance.json').read_text())
    probe=json.loads((a.historical/'outputs/vit_small/metadata.json').read_text())['sample_indices']
    images=sorted(provenance['images'],key=lambda r:r['path'])
    probe_hashes=[images[i]['sha256'] for i in probe]
    if len(set(probe_hashes))!=512:
        raise ValueError('Historical 512 probe has duplicate image content: bootstrap needs a cluster audit')
    visits=(len(clean)//1024)*1024
    audit=dict(original_training_images=1281167,training_images_after_overlap_exclusion=len(clean),
        excluded_train_images=len(excluded),excluded_train_unique_hashes=len({r['sha256'] for r in overlaps}),
        probe_images_affected_before_exclusion=len(set(probe_hashes)&{r['sha256'] for r in overlaps}),
        validation_images=50000,probe_unique_images=512,class_counts=counts.tolist(),
        exclusion_rule='Exact image bytes present in validation; labels and model results not used',
        training_index_sha256=sha(index),full_training_index_sha256=sha(full),
        visits_per_epoch=visits,updates_per_epoch=visits//1024,visits_300_epochs=visits*300,
        excluded_records=overlaps,validation_root=str(a.historical/'data/imagenet1k_val'))
    (a.root/'DATA_READY.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['excluded_records','class_counts']}),flush=True)


if __name__=='__main__': main()
