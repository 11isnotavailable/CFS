"""Pack verified bytes without image transcoding; preserve checksums and ordering."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import numpy as np

DTYPE=np.dtype([('label','<u2'),('path','S200'),('shard','<u2'),('offset','<u8'),('length','<u4')])


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def pack_parquet(root,files,records,split,shard,labels,stem=None):
    """Sequential parquet IO avoids the host's slow per-image metadata reads."""
    import pyarrow.parquet as pq
    marker=root/f'{split}_{shard:04d}.json'
    if marker.exists(): return
    expected={r['path']:r['sha256'] for r in records}; seen=set(); rows=[]; h=hashlib.sha256()
    final=root/f'{split}_{shard:04d}.images'; temp=final.with_suffix('.tmp')
    with temp.open('wb') as out:
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    label=int(row['label']); image=row['image']; name=Path(image['path']).name
                    relative=f'{labels[label]}/'+(f'{stem}_' if stem else '')+name
                    raw=image['bytes']; digest=hashlib.sha256(raw).hexdigest()
                    if expected.get(relative)!=digest or relative in seen:
                        raise ValueError(f'Packed image identity mismatch: {relative}')
                    seen.add(relative); rows.append((label,relative.encode(),shard,out.tell(),len(raw)))
                    out.write(raw); h.update(raw)
        out.flush(); os.fsync(out.fileno())
    assert len(seen)==len(expected)
    temp.replace(final); np.save(root/f'{split}_{shard:04d}.npy',np.array(rows,dtype=DTYPE))
    marker.write_text(json.dumps(dict(images=len(rows),sha256=h.hexdigest(),bytes=final.stat().st_size),indent=2))
    print('PACKED_PARQUET',split,shard,len(rows),flush=True)


def pack(root,source,records,split,shard,remove_staging=False):
    marker=root/f'{split}_{shard:04d}.json'
    if marker.exists(): return
    final=root/f'{split}_{shard:04d}.images'; temp=final.with_suffix('.tmp')
    rows=[]; h=hashlib.sha256()
    def read(record):
        path=source/record['path']; raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256']: raise ValueError(f'Image checksum mismatch: {path}')
        return raw
    with temp.open('wb') as out, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(records),128):
            batch=records[start:start+128]
            for record,raw in zip(batch,pool.map(read,batch)):
                if len(record['path'].encode())>200: raise ValueError('Index path too long')
                rows.append((record['label'],record['path'].encode(),shard,out.tell(),len(raw)))
                out.write(raw); h.update(raw)
        out.flush(); os.fsync(out.fileno())
    temp.replace(final)
    np.save(root/f'{split}_{shard:04d}.npy',np.array(rows,dtype=DTYPE))
    marker.write_text(json.dumps(dict(images=len(rows),sha256=h.hexdigest(),bytes=final.stat().st_size),indent=2))
    if remove_staging:
        # Only this experiment's new staging images, byte-verified and retained in packs.
        assert source.resolve().name=='train' and source.resolve().parent.name=='data'
        assert source.resolve().parent.parent.name=='cfs_controlled_vit_20261116'
        for record in records:
            path=(source/record['path']).resolve()
            if not path.is_relative_to(source.resolve()): raise ValueError('Unsafe staging path')
            path.unlink()
    print('PACKED',split,shard,len(rows),flush=True)


def merge(root,split,count):
    arrays=[np.load(root/f'{split}_{i:04d}.npy') for i in range(count)]
    values=np.concatenate(arrays); values.sort(order=['label','path'])
    np.save(root/f'{split}_index.npy',values)
    return len(values)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--historical',type=Path,default=Path('/data/cfs_rebuttal_20261114'))
    p.add_argument('--val-only',action='store_true'); a=p.parse_args()
    packed=a.root/'packed'; packed.mkdir(parents=True,exist_ok=True)
    from timm.data import ImageNetInfo
    labels=ImageNetInfo().label_names()
    (packed/'classes.json').write_text(json.dumps({label:i for i,label in enumerate(labels)},indent=2))
    provenance=json.loads((a.historical/'imagenet_provenance.json').read_text())
    valfiles=sorted((a.historical/'data/imagenet_parquet').rglob('*validation-*.parquet'))
    assert valfiles
    pack_parquet(packed,valfiles,provenance['images'],'val',0,labels)
    assert merge(packed,'val',1)==50000
    (packed/'VAL_READY.json').write_text(json.dumps(dict(images=50000)))
    if a.val_only: return
    plan=json.loads((a.root/'download_plan.json').read_text())
    source=(a.root/'train').resolve()
    assert source.parent.parent.name=='cfs_controlled_vit_20261116'
    parents={label:(source/label).resolve() for label in labels}
    assert all(parent.is_relative_to(source) for parent in parents.values())
    cleanup_pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
    cleanup_futures=[]
    def cleanup_shard(records):
        def remove_copy(record):
            parts=Path(record['path']).parts
            assert len(parts)==2 and parts[0] in parents and parts[1] not in ['.','..']
            (parents[parts[0]]/parts[1]).unlink(missing_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as cleanup:
            list(cleanup.map(remove_copy,records))
    def prepare(shard):
        stem=f'train-{shard:05d}-of-00310'; marker=a.root/'train_manifests'/(stem+'.complete.json')
        while not marker.exists(): time.sleep(10)
        records=[json.loads(line) for line in (marker.parent/(stem+'.jsonl')).read_text().splitlines()]
        if not (packed/f'train_{shard:04d}.json').exists():
            while shutil.disk_usage(a.root).free < 20*(1<<30):
                for future in cleanup_futures:
                    if future.done(): future.result()
                time.sleep(5)
            info=plan['files'][shard]; target=a.root/'download_cache'/Path(info['name']).name
            url=f"https://hf-mirror.com/datasets/{plan['repo']}/resolve/{plan['revision']}/{info['name']}"
            if not target.exists() or target.stat().st_size!=info['bytes']:
                subprocess.run(['curl','-fsSL','--retry','8','--retry-delay','3','-C','-','-o',str(target),url],check=True)
            assert target.stat().st_size==info['bytes'] and file_hash(target)==info['sha256']
            pack_parquet(packed,[target],records,'train',shard,labels,stem)
            assert target.resolve().parent==(a.root/'download_cache').resolve(); target.unlink()
        # Remove only new staging copies after byte-verified data packs exist.
        cleanup_futures.append(cleanup_pool.submit(cleanup_shard,records))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare,range(310)))
    assert merge(packed,'train',310)==1281167
    (packed/'PACKED_READY.json').write_text(json.dumps(dict(train_images=1281167,val_images=50000,
        transformation='byte concatenation only; no resize/reencode; sorted label/path ordering retained')))
    for future in cleanup_futures: future.result()
    cleanup_pool.shutdown()
    (packed/'STAGING_CLEANUP_COMPLETE.json').write_text(json.dumps(dict(staging_copies_removed=True)))


if __name__=='__main__': main()
