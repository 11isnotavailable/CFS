"""Export existing teacher/features without re-running any backbone intervention."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.train_dynamic_cfs_router import load_joint_teacher, assert_joint_alignment
from src.learned_head_gating import assert_disjoint


def load_fields(directory, fields):
    parts = {key: [] for key in fields}
    for path in sorted(directory.glob('shard_*.npz')):
        with np.load(path) as archive:
            for key in fields:
                parts[key].append(archive[key])
    return {key: np.concatenate(value) for key, value in parts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.workspace / 'outputs'
    locations = {'train': out / 'vit_base_layer11_train4096_alpha16',
                 'tune': out / 'vit_base_layer11_router_final_split/tune',
                 'test': out / 'vit_base_layer11_router_final_split/test'}
    data = {split: load_fields(path, ['prehead', 'path', 'dataset_index', 'imagenet_target'])
            for split, path in locations.items()}
    assert_disjoint(*(v['path'] for v in data.values()))
    manifest = {'source_dirs': {k: str(v) for k, v in locations.items()}, 'counts': {},
                'teacher': 'Existing 48 hard masks per budget; no teacher recomputation',
                'heldout_teacher_exported': False}
    for split, values in data.items():
        export = {'feature': values['prehead'][:, 0].astype(np.float32),
                  'paths': values['path'], 'dataset_index': values['dataset_index'],
                  'target': values['imagenet_target']}
        if split != 'test':
            teacher = load_joint_teacher(out / ('vit_base_joint_teacher_train4096_c48'
                                                if split == 'train' else 'vit_base_joint_teacher_tune64_c48'))
            assert_joint_alignment({'dataset_index': values['dataset_index'], 'paths': values['path']}, teacher)
            export.update(masks=teacher['candidate_masks'], kl=teacher['joint_kl'])
        np.savez_compressed(args.output / f'{split}.npz', **export)
        manifest['counts'][split] = len(values['path'])
    for seed in (17, 23, 41):
        source = out / f'vit_base_joint_router_seed{seed}_w05'
        payload = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
        payload['args'] = {k: str(v) if isinstance(v, Path) else v for k, v in payload['args'].items()}
        torch.save(payload, args.output / f'r2_seed{seed}.pt')
        shutil.copy2(source / 'test_router_predictions.npz', args.output / f'r2_predictions_seed{seed}.npz')
        shutil.copy2(out / f'vit_base_joint_router_calibrated_execution_seed{seed}_w05/execution_samples.csv',
                     args.output / f'historical_seed{seed}.csv')
    image_root = args.output / 'images'
    for raw in data['test']['path'].astype(str):
        relative = raw.replace('\\', '/').split('imagenette2-160/')[-1]
        source = args.workspace / '.cache/data/imagenette2-160' / relative
        destination = image_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    checkpoint = args.workspace / 'assets/vit_scaling/base.pth'
    manifest['backbone_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in args.output.glob('*') if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    main()
