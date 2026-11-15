"""Match exported masks/labels to the original immutable teacher shards."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-train',type=Path,required=True)
    p.add_argument('--original-tune',type=Path,required=True);p.add_argument('--export',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report={}
    for split,directory in [('train',args.original_train),('tune',args.original_tune)]:
        files=sorted(directory.glob('shard_*.npz'))
        if not files:raise FileNotFoundError(directory)
        arrays=[dict(np.load(f)) for f in files];exported=np.load(args.export/f'{split}.npz')
        for original,current in [('candidate_masks','masks'),('joint_kl','kl'),('dataset_index','dataset_index')]:
            np.testing.assert_array_equal(np.concatenate([a[original] for a in arrays]),exported[current])
        report[split]=dict(shards=len(files),masks_exact_match=True,kl_exact_match=True,dataset_indices_exact_match=True,
            metadata=json.loads((directory/'metadata.json').read_text()),
            original_shard_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            exported_sha256=hashlib.sha256((args.export/f'{split}.npz').read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2));print('Original teacher provenance verified for both splits')


if __name__=='__main__':main()
