"""One-time, shared image preprocessing. Never cache random training augmentation.

Train: RGB, bicubic short edge256 (long edge capped1024), JPEG95/4:2:0.
Validation: exact original deterministic224 crop stored as CHW uint8 bytes.
Original packs and image hashes remain unchanged for provenance and CFS.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scripts.pack_controlled_imagenet import DTYPE
from src.controlled_vit_imagenet import MEAN, STD


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def resized_shape(width,height):
    if width<=height: nw,nh=256,int(height*256/width)
    else: nw,nh=int(width*256/height),256
    if max(nw,nh)>1024:
        ratio=1024/max(nw,nh); nw=max(1,int(nw*ratio)); nh=max(1,int(nh*ratio))
    return nw,nh


def process_shard(source,out,split,shard,workers):
    marker=out/f'{split}_{shard:04d}.json'
    if marker.exists(): return
    index=np.load(source/f'{split}_{shard:04d}.npy')
    fd=os.open(source/f'{split}_{shard:04d}.images',os.O_RDONLY)
    transform=None
    if split=='val':
        from timm.data import create_transform
        transform=create_transform(input_size=(3,224,224),is_training=False,use_prefetcher=True,
            mean=MEAN,std=STD,interpolation='bicubic',crop_pct=.875)
    def convert(row):
        raw=os.pread(fd,int(row['length']),int(row['offset']))
        with Image.open(io.BytesIO(raw)) as im: image=im.convert('RGB')
        original_size=image.size
        if split=='train':
            size=resized_shape(*image.size)
            if image.size!=size: image=image.resize(size,resample=Image.Resampling.BICUBIC)
            buffer=io.BytesIO(); image.save(buffer,format='JPEG',quality=95,subsampling=2,optimize=False,progressive=False)
            encoded=buffer.getvalue()
        else:
            pixels=np.asarray(transform(image)); assert pixels.shape==(3,224,224) and pixels.dtype==np.uint8
            encoded=np.ascontiguousarray(pixels).tobytes(); size=(224,224)
        return encoded,dict(path=row['path'].decode(),original_sha256=hashlib.sha256(raw).hexdigest(),
            processed_sha256=hashlib.sha256(encoded).hexdigest(),original_size=original_size,processed_size=size)
    final=out/f'{split}_{shard:04d}.images'; temp=final.with_suffix('.tmp')
    manifest=out/f'{split}_{shard:04d}_images.jsonl'; manifest_temp=manifest.with_suffix('.tmp')
    new=[]; whole=hashlib.sha256(); before=0; after=0; started=time.time()
    try:
        with temp.open('wb') as stream, manifest_temp.open('w') as records, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0,len(index),64):
                batch=index[start:start+64]
                for row,(encoded,record) in zip(batch,pool.map(convert,batch)):
                    new.append((int(row['label']),row['path'],shard,stream.tell(),len(encoded)))
                    stream.write(encoded); whole.update(encoded)
                    records.write(json.dumps(record)+'\n'); before+=int(row['length']); after+=len(encoded)
            stream.flush(); os.fsync(stream.fileno())
        temp.replace(final); manifest_temp.replace(manifest)
        np.save(out/f'{split}_{shard:04d}.npy',np.array(new,dtype=DTYPE))
        marker.write_text(json.dumps(dict(images=len(index),sha256=whole.hexdigest(),source_pack_sha256=sha_file(source/f'{split}_{shard:04d}.images'),
            original_bytes=before,processed_bytes=after,seconds=time.time()-started),indent=2))
        print('PREPROCESSED',split,shard,len(index),before,after,flush=True)
    finally: os.close(fd)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8); a=p.parse_args()
    source=a.data_root/'packed'; out=a.data_root/'preprocessed'; out.mkdir(exist_ok=True)
    while not (source/'VAL_READY.json').exists(): time.sleep(10)
    (out/'classes.json').write_bytes((source/'classes.json').read_bytes())
    (out/'storage.json').write_text(json.dumps(dict(train='jpeg',val='chw_uint8',benchmark='jpeg',benchmark_prefix='train',
        train_short_edge=256,train_max_long_edge=1024,jpeg_quality=95,jpeg_subsampling=2,val_shape=[3,224,224],
        augmentation_cached=False),indent=2))
    process_shard(source,out,'val',0,a.workers)
    values=np.load(out/'val_0000.npy'); values.sort(order=['label','path']); np.save(out/'val_index.npy',values)
    (out/'VAL_READY.json').write_text(json.dumps(dict(images=50000,format='CHW uint8 deterministic crop')))
    pending=set(range(310))
    while pending:
        for shard in sorted(pending):
            if not (source/f'train_{shard:04d}.json').exists(): continue
            process_shard(source,out,'train',shard,a.workers); pending.remove(shard)
            if not (out/'BENCHMARK_READY.json').exists() and all((out/f'train_{i:04d}.json').exists() for i in range(8)):
                small=np.concatenate([np.load(out/f'train_{i:04d}.npy') for i in range(8)])
                small.sort(order=['label','path']); np.save(out/'benchmark_index.npy',small)
                (out/'BENCHMARK_READY.json').write_text(json.dumps(dict(images=len(small),purpose='Throughput only; weights discarded')))
        if pending: time.sleep(10)
    while not (a.data_root/'DATA_READY.json').exists(): time.sleep(10)
    audit=json.loads((a.data_root/'DATA_READY.json').read_text()); excluded={r['path'].encode() for r in audit['excluded_records']}
    all_rows=np.concatenate([np.load(out/f'train_{i:04d}.npy') for i in range(310)])
    all_rows.sort(order=['label','path']); np.save(out/'train_full_index.npy',all_rows)
    clean=all_rows[np.array([path not in excluded for path in all_rows['path']])]
    assert len(clean)==audit['training_images_after_overlap_exclusion']; np.save(out/'train_index.npy',clean)
    total_before=total_after=0
    for i in range(310):
        row=json.loads((out/f'train_{i:04d}.json').read_text()); total_before+=row['original_bytes']; total_after+=row['processed_bytes']
    (out/'PREPROCESSED_READY.json').write_text(json.dumps(dict(train_images=len(clean),val_images=50000,
        original_train_bytes=total_before,processed_train_bytes=total_after,storage_ratio=total_after/total_before,
        train_index_sha256=sha_file(out/'train_index.npy'),val_index_sha256=sha_file(out/'val_index.npy'),
        note='Training preprocessing changed uniformly before formal training. Validation cached pixels preserve the original evaluation transform.'),indent=2))


if __name__=='__main__': main()
