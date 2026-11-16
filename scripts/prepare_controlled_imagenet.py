"""Download pinned full-resolution training shards; reuse the historical validation set."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
REPO = 'benjamin-paine/imagenet-1k'
REVISION = 'ac6a45918fb830d6d1555b01090d7b991e8f8a4a'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--historical', type=Path, default=Path('/data/cfs_rebuttal_20261114'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--inspect', action='store_true')
    a = p.parse_args()
    from huggingface_hub import HfApi
    import pyarrow.parquet as pq
    from timm.data import ImageNetInfo
    a.root.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    shards = sorted([s for s in info.siblings if '/train-' in s.rfilename and s.rfilename.endswith('.parquet')], key=lambda s: s.rfilename)
    assert shards and info.sha == REVISION
    description = dict(repo=REPO, revision=REVISION, shards=len(shards), bytes=sum(s.size for s in shards),
                       files=[dict(name=s.rfilename, bytes=s.size, sha256=s.lfs.sha256) for s in shards])
    (a.root / 'download_plan.json').write_text(json.dumps(description, indent=2))
    print(json.dumps({k:v for k,v in description.items() if k != 'files'}), flush=True)
    if a.inspect:
        return
    labels = ImageNetInfo().label_names()
    val_prov = json.loads((a.historical/'imagenet_provenance.json').read_text())
    assert val_prov['revision'] == REVISION and len(val_prov['images']) == 50000
    val_hashes = {r['sha256'] for r in val_prov['images']}
    cache = a.root/'download_cache'; cache.mkdir(exist_ok=True)
    records = a.root/'train_manifests'; records.mkdir(exist_ok=True)
    train = a.root/'train'; train.mkdir(exist_ok=True)
    for label in labels:
        (train/label).mkdir(exist_ok=True)
    def prepare(s):
        stem = Path(s.rfilename).stem
        marker = records/(stem+'.complete.json')
        if marker.exists():
            return json.loads(marker.read_text())
        target = cache/Path(s.rfilename).name
        url = f'https://hf-mirror.com/datasets/{REPO}/resolve/{REVISION}/{s.rfilename}'
        subprocess.run(['curl','-fsSL','--retry','8','--retry-delay','5','--connect-timeout','30',
                        '--speed-time','120','--speed-limit','1024','-C','-','-o',str(target),url], check=True)
        if target.stat().st_size != s.size or sha(target) != s.lfs.sha256:
            raise ValueError(f'Shard checksum mismatch: {target}')
        count = 0; overlap = []; class_counts = [0]*1000
        manifest = records/(stem+'.jsonl'); temp = manifest.with_suffix('.tmp')
        with temp.open('w') as out:
            for batch in pq.ParquetFile(target).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    label = int(row['label']); raw = row['image']['bytes']
                    original = Path(row['image'].get('path') or f'{stem}_{count:07d}.JPEG').name
                    if 'ILSVRC2012_val' in original:
                        raise ValueError('Validation filename in training shard')
                    # Shard prefix ensures unique paths even if source basenames repeat.
                    relative = f'{labels[label]}/{stem}_{original}'
                    dest = train/relative; digest = hashlib.sha256(raw).hexdigest()
                    if digest in val_hashes:
                        overlap.append(dict(path=relative, sha256=digest))
                    if not dest.exists():
                        dest.write_bytes(raw)
                    elif sha(dest) != digest:
                        raise ValueError(f'Image collision: {dest}')
                    out.write(json.dumps(dict(path=relative, label=label, bytes=len(raw), sha256=digest))+'\n')
                    count += 1; class_counts[label] += 1
        temp.replace(manifest)
        result = dict(shard=s.rfilename, sha256=s.lfs.sha256, images=count, class_counts=class_counts,
                      validation_content_overlap=overlap, manifest_sha256=sha(manifest))
        marker.write_text(json.dumps(result, indent=2))
        # Only our freshly downloaded, verified disposable shard cache is removed.
        assert target.resolve().parent == cache.resolve()
        target.unlink()
        print('PREPARED', stem, count, flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        summaries = list(pool.map(prepare, shards))
    count = sum(s['images'] for s in summaries)
    counts = [sum(s['class_counts'][i] for s in summaries) for i in range(1000)]
    overlap = [r for s in summaries for r in s['validation_content_overlap']]
    result = dict(repo=REPO, revision=REVISION, images=count, classes=1000, class_counts=counts,
                  validation_root=str(a.historical/'data/imagenet1k_val'), validation_content_overlap=overlap,
                  manifests_sha256={p.name:sha(p) for p in sorted(records.glob('*.jsonl'))}, completed_unix=time.time())
    (a.root/'data_audit.json').write_text(json.dumps(result, indent=2))
    assert count == 1281167 and min(counts) > 0
    if overlap:
        raise RuntimeError(f'{len(overlap)} train/validation byte duplicates: review data_audit before training')
    (a.root/'DATA_READY.json').write_text(json.dumps(result, indent=2))
    print('DATA_READY',count,flush=True)


if __name__ == '__main__':
    main()
