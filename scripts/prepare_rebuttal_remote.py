"""Download public assets and freeze provenance for the authorized rerun."""
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HOME','/data/cfs_rebuttal_20261114/cache/huggingface')
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

ROOT=Path('/data/cfs_rebuttal_20261114')
API=HfApi()

def download_verified(repo, info, dest, dataset=False, only=None):
    """GET through mirror; verify LFS SHA256 or Git blob ID against pinned API."""
    prefix='datasets/' if dataset else ''
    for f in info.siblings:
        name=f.rfilename
        if only is not None:
            if name not in only: continue
        elif not name.endswith(('.json','.safetensors','.txt','.model')): continue
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True)
        expected_size=f.size
        if not target.exists() or target.stat().st_size!=expected_size:
            subprocess.run(['curl','-fL','--retry','5','--retry-delay','3','--connect-timeout','20','--speed-time','60','--speed-limit','1024','-C','-','-o',str(target),f'https://hf-mirror.com/{prefix}{repo}/resolve/main/{name}'],check=True)
        if target.stat().st_size!=expected_size: raise ValueError('Download size mismatch')
        digest=hashlib.sha256() if f.lfs else hashlib.sha1()
        if not f.lfs: digest.update(f'blob {expected_size}\0'.encode())
        with target.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
        expected=f.lfs.sha256 if f.lfs else f.blob_id
        if digest.hexdigest()!=expected: raise ValueError(f'Checksum mismatch: {target}')
        print('VERIFIED',name,flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['qwen','imagenet','inspect-imagenet']); a=p.parse_args()
    if a.mode=='inspect-imagenet':
        for repo in ['ILSVRC/imagenet-1k','benjamin-paine/imagenet-1k','satabios/imagenet_val']:
            info=API.dataset_info(repo)
            print(repo,'gated=',info.gated,'sha=',info.sha,flush=True)
            print([s.rfilename for s in info.siblings if 'val' in s.rfilename][:25],flush=True)
        return
    if a.mode=='imagenet':
        import pyarrow.parquet as pq
        from timm.data import ImageNetInfo
        labels=ImageNetInfo().label_names()
        repo='benjamin-paine/imagenet-1k'
        info=API.dataset_info(repo,files_metadata=True)
        names=sorted(f.rfilename for f in info.siblings if 'validation-' in f.rfilename and f.rfilename.endswith('.parquet'))
        dest=ROOT/'data'/'imagenet_parquet'
        download_verified(repo,info,dest,dataset=True,only=names)
        imgroot=ROOT/'data'/'imagenet1k_val'
        records=[]; count=0
        for name in names:
            pf=pq.ParquetFile(dest/name)
            for batch in pf.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    label=int(row['label']); image=row['image']
                    filename=Path(image.get('path') or f'val_{count:08d}.JPEG').name
                    target=imgroot/labels[label]/filename
                    raw=image['bytes']; digest=hashlib.sha256(raw).hexdigest()
                    if target.exists():
                        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest: raise ValueError('Image collision')
                    else:
                        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
                    records.append(dict(path=str(target.relative_to(imgroot)),sha256=digest,label=label))
                    count+=1
        if count!=50000 or len(set(r['path'] for r in records))!=50000: raise ValueError('Incomplete validation set')
        records.sort(key=lambda r:r['path'])
        (ROOT/'imagenet_provenance.json').write_text(json.dumps(dict(repo=repo,revision=info.sha,images=records),indent=2))
        print('IMAGENET_COMPLETE',count,flush=True)
        return
    prov={}
    for size in ['1.5B','3B','7B','14B']:
        repo='Qwen/Qwen2.5-'+size
        info=API.model_info(repo,files_metadata=True)
        dest=ROOT/'models'/('Qwen2.5-'+size)
        print('DOWNLOAD',repo,info.sha,flush=True)
        download_verified(repo,info,dest)
        prov[repo]=dict(revision=info.sha,path=str(dest))
        (ROOT/'model_provenance.json').write_text(json.dumps(prov,indent=2))
    repo='HuggingFaceFW/fineweb-edu'; info=API.dataset_info(repo,files_metadata=True)
    filename='sample/10BT/000_00000.parquet'
    download_verified(repo,info,ROOT/'data',dataset=True,only=[filename])
    path=str(ROOT/'data'/filename)
    prov['data']=dict(repo=repo,revision=info.sha,parquet=path)
    (ROOT/'model_provenance.json').write_text(json.dumps(prov,indent=2))
    contexts=ROOT/'data'/'fineweb_rebuttal_256x129.npz'
    if not contexts.exists():
        subprocess.run([sys.executable,'scripts/prepare_fineweb_eval.py',
                        '--model',str(ROOT/'models/Qwen2.5-1.5B'), '--parquet',path,
                        '--output',str(contexts),'--samples','256','--seed','20260802'],check=True)
    print('PREPARATION_COMPLETE',flush=True)

if __name__=='__main__': main()
