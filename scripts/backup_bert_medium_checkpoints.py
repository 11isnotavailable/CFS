"""Download the28 immutable milestone weights and verify each remote SHA256."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path,PurePosixPath

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,default=20021)
    p.add_argument('--remote-root',default='/data/cfs_bert_medium_20261116');a=p.parse_args()
    entries=json.loads(a.manifest.read_text());assert len(entries)==28
    a.destination.mkdir(parents=True,exist_ok=True);verified=[];started=time.time()
    for entry in entries:
        rel=PurePosixPath(entry['path'])
        assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='runs' and rel.suffix=='.pt'
        dest=a.destination.joinpath(*rel.parts);dest.parent.mkdir(parents=True,exist_ok=True)
        if not (dest.exists() and dest.stat().st_size==entry['bytes'] and sha(dest)==entry['sha256']):
            temporary=dest.with_suffix('.download')
            subprocess.run(['scp','-q','-o','BatchMode=yes','-i',str(a.identity),'-P',str(a.port),
                            f'{a.host}:{a.remote_root}/{rel.as_posix()}',str(temporary)],check=True)
            assert temporary.stat().st_size==entry['bytes'] and sha(temporary)==entry['sha256'],str(rel)
            temporary.replace(dest)
        verified.append(entry)
        status=dict(verified_checkpoints=len(verified),total_checkpoints=28,verified_bytes=sum(x['bytes'] for x in verified),seconds=time.time()-started)
        (a.destination/'BACKUP_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    (a.destination/'CHECKPOINT_MANIFEST.json').write_text(json.dumps(verified,indent=2))
    (a.destination/'COMPLETE.json').write_text(json.dumps(dict(**status,all_sha256_verified=True),indent=2))
if __name__=='__main__':main()
