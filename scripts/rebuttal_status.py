"""Read-only status and dataset integrity checks on the experiment server."""
import json
from pathlib import Path
import subprocess

root=Path('/data/cfs_rebuttal_20261114')
for p in sorted(root.glob('*.log')):
    tail=p.read_text(errors='replace').replace('\r','\n').splitlines()[-3:]
    print(p.name, '\n'.join(tail),flush=True)
for d in sorted((root/'outputs').glob('*')):
    print('SHARDS',d.name,len(list(d.glob('shard_*.npz'))),flush=True)
subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'])
if (root/'imagenet_provenance.json').exists():
    prov=json.loads((root/'imagenet_provenance.json').read_text())
    from collections import Counter
    counts=Counter(x['label'] for x in prov['images'])
    assert len(counts)==1000 and set(counts.values())=={50}
    print('IMAGENET_VERIFIED_1000_CLASSES_50_IMAGES',prov['images'][:2])
    import pyarrow.parquet as pq
    p=next((root/'data/imagenet_parquet').rglob('*.parquet'))
    md=pq.ParquetFile(p).schema_arrow.metadata
    features=json.loads(md[b'huggingface'])['info']['features']
    names=features['label']['names']
    print('PARQUET_LABELS',names[:5])
