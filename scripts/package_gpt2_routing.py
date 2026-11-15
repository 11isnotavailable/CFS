"""Archive all router checkpoints/logs and raw teachers; keep full GPU caches remote."""
import argparse,json,shutil,tarfile
from pathlib import Path
import numpy as np
from verify_gpt2_routing import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    target=root/'artifact_export';target.mkdir(exist_ok=False)
    for name in ('data','src','scripts','tests','docs','analysis'):
        if (root/name).exists():shutil.copytree(root/name,target/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for pattern in ('*.json','*.log'):
        for file in root.glob(pattern):shutil.copy2(file,target/file.name)
    model=target/'model';model.mkdir()
    for file in (root/'model').iterdir():
        if file.is_file() and file.suffix not in ('.safetensors','.bin'):shutil.copy2(file,model/file.name)
    manifest=[]
    for layer in (11,8):
        d=root/f'layer{layer}';out=target/d.name
        shutil.copytree(d,out,ignore=shutil.ignore_patterns('pair','evaluation_initial_precision_check'))
        for split in ('train','tune'):
            src=d/f'pair/{split}';dst=out/f'pair/{split}';dst.mkdir(parents=True)
            for file in sorted(src.iterdir()):
                if not file.is_file():continue
                if file.suffix=='.npz':
                    manifest.append(dict(path=str(file.relative_to(root)),bytes=file.stat().st_size,sha256=sha(file),archived_full=split=='tune'))
                    if split=='train':
                        with np.load(file) as f:record={k:f[k] for k in f.files if k not in ('residual','heads','reference')}
                        np.savez_compressed(dst/file.name,**record)
                    else:shutil.copy2(file,dst/file.name)
                else:shutil.copy2(file,dst/file.name)
    (target/'full_cache_manifest.json').write_text(json.dumps(manifest,indent=2))
    (target/'ARCHIVE_README.md').write_text('''# GPT-2 routing artifact archive

Contains all pilot/bootstrap/final router checkpoint epochs and selected models,
all logs/configs, exact frozen documents/tokens, all pairwise drops/substitution/
alpha/features/baselines, all joint masks/KLs, full tune suffix caches, test
predictions and raw results, bootstrap draws, analysis and source code.

Only TRAIN pair-cache residual/heads/reference tensors are omitted here to save
space. They remain unchanged in the remote run root, with full file SHA256 in
full_cache_manifest.json. No scientific teacher target was omitted. Rebuilding
joint labels from compact train files requires the original full remote caches
or fresh prefix capture; existing joint labels need no recomputation.

The frozen pretrained backbone remains under the remote run root/model. Its
exact revision and every file SHA256 are in provenance.json. Download the pinned
model.safetensors from openai-community/gpt2 at that revision if reconstructing
the run from this archive; verify its hash. It was never trained. All trained
router weights ARE included in this archive, including every saved epoch.

Use a fresh run root to recompute from source. Completed experiment directories
must not be overwritten. Source dependencies are recorded in environment.json.
''',encoding='utf-8')
    archive=root.parent/'cfs_gpt2_routing_artifacts_20261115.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=3) as stream:stream.add(target,arcname='gpt2_routing')
    record=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),remote_root=str(root),
        public_backbone=json.loads(root.joinpath('provenance.json').read_text()),train_cache_manifest='artifact_export/full_cache_manifest.json')
    (root/'archive_manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='public_backbone'},indent=2))


if __name__=='__main__':main()
