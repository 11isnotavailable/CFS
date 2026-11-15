"""Full raw data/coefficients/checkpoints/logs; no omitted teacher tensors."""
import argparse,hashlib,json,tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    path=root.parent/'cfs_coalitional_artifacts_20261116.tar.gz';assert not path.exists()
    with tarfile.open(path,'w:gz',compresslevel=3) as tar:
        for name in ('analysis','data','images','teacher','runs','pilot','benchmark','profile','src','scripts','tests'):
            tar.add(root/name,arcname='coalitional/'+name,filter=lambda t:None if '__pycache__' in t.name else t)
        for pattern in ('*.json','*.md','*.log','*COMPLETE'):
            for file in root.glob(pattern):
                if file.is_file():tar.add(file,arcname='coalitional/'+file.name)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    record=dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest(),remote_root=str(root),
        excludes='unchanged backbone (separate existing backup); earlier intermediate analysis presentations only')
    (root/'archive_manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
