"""Export small evidence files; retain training token caches and weights remotely."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    status = json.loads((root / 'STATUS.json').read_text())
    if status['stage'] != 'complete_2B':
        raise RuntimeError('Final export requires completed seven-model analysis')
    weights = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
                    sha256=sha256(p)) for p in sorted((root / 'runs').glob('B*/model_*.pt'))]
    assert len(weights) == 28, 'Expected four milestones for each of seven models'
    (root / 'CHECKPOINT_MANIFEST.json').write_text(json.dumps(weights, indent=2))
    selected = []
    for directory in ['analysis', 'audit', 'cfs', 'runs']:
        selected.extend(p for p in (root / directory).rglob('*')
                        if p.is_file() and p.suffix not in {'.pt', '.tmp'})
    for pattern in ['benchmark/*.csv', 'benchmark/*.json', 'benchmark/*.md',
                    'benchmark/measurements/*.json', 'benchmark/io_confirmation/*.json',
                    'data/READY.json', 'data/validation*.json', 'data/part_*.json',
                    'data/validation.npz', 'data/cfs_inputs.npz', 'smoke/*PASSED.json']:
        selected.extend(root.glob(pattern))
    selected.extend(root / name for name in ['STATUS.json', 'PROTOCOL_LOCK.json',
                                             'CHECKPOINT_MANIFEST.json', 'pipeline.log'])
    files = sorted(set(selected))
    manifest = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
                     sha256=sha256(p)) for p in files]
    (root / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    temporary = args.output.with_suffix('.partial')
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in files + [root / 'ARTIFACT_MANIFEST.json']:
            archive.add(path, arcname=path.relative_to(root), recursive=False)
    temporary.replace(args.output)
    print(json.dumps(dict(archive=str(args.output), sha256=sha256(args.output),
                          files=len(files), checkpoints_retained_remotely=len(weights))))


if __name__ == '__main__':
    main()
