"""Uniform frozen-backbone evaluation with common dense references and compensation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from scripts.evaluate_dynamic_cfs_execution import FrozenPathDataset, dense_kl
from src.dynamic_cfs_router import DynamicCFSRouter, subset_masks, selected_masks, compose_scales
from src.learned_head_gating import LearnedHeadGating
from src.learned_head_gating import topk_mask
from src.head_intervention import make_intervenable, clear_interventions
from src.model_loading import create_timm_model
from src.runtime_cfs_router import prehead_summary


def predict(payload, features, kind, device):
    model = (LearnedHeadGating() if kind == 'gating' else DynamicCFSRouter(2304, 12)).to(device).eval()
    model.load_state_dict(payload['state_dict'])
    normalized = (features-payload['global_mean'])/payload['global_std']
    results = []
    with torch.inference_mode():
        for begin in range(0, len(features), 64):
            result = model(torch.from_numpy(normalized[begin:begin+64]).to(device))
            results.append([v.cpu() for v in result] if isinstance(result, tuple) else result.cpu())
    if kind == 'gating':
        scores = torch.cat(results)
        return {k: topk_mask(scores, k) for k in (3, 6, 9)}, model
    score, importance, alpha = [torch.cat([r[i] for r in results]) for i in range(3)]
    masks = {k: selected_masks(score, importance.softmax(-1), subset_masks(12, k))[0] for k in (3, 6, 9)}
    return (masks, score, importance, alpha), model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-samples', type=int)
    p.add_argument('--feature-file', type=Path, help='Optional separately audited live prehead features; no retraining')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = json.loads((args.data/'manifest.json').read_text())
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == manifest['backbone_sha256']
    data = dict(np.load(args.data/'test.npz'))
    count = len(data['paths']) if args.max_samples is None else args.max_samples
    features = data['feature'][:count]
    if args.feature_file:
        live = np.load(args.feature_file)
        np.testing.assert_array_equal(live['dataset_index'], data['dataset_index'])
        features = live['feature'][:count]
    scale_table, mask_archive, checks, complexity = {}, {}, {}, []
    for seed in (17, 23, 41):
        original = torch.load(args.data/f'r2_seed{seed}.pt', map_location='cpu', weights_only=False)
        historical = dict(np.load(args.data/f'r2_predictions_seed{seed}.npz'))
        assert np.array_equal(data['paths'], historical['paths'])
        (r2masks, score, importance, alpha), r2 = predict(original, features, 'r2', args.device)
        checks[f'r2_seed{seed}_score_max_error'] = float((score-torch.from_numpy(historical['predicted_scores'][:count])).abs().max())
        checks[f'r2_seed{seed}_mask_disagreements'] = sum(int((r2masks[k]!=torch.from_numpy(historical[f'keep{k}_dynamic_cfs_router'][:count])).any(-1).sum()) for k in (3, 6, 9))
        # Original historical decisions retained exactly; no new checkpoint choice.
        if args.feature_file is None:
            score = torch.from_numpy(historical['predicted_scores'][:count])
            alpha = torch.from_numpy(historical['predicted_alpha'][:count])
            r2masks = {k: torch.from_numpy(historical[f'keep{k}_dynamic_cfs_router'][:count]) for k in (3, 6, 9)}
        methods = {'R2': r2masks,
                   'Taylor': {k: torch.from_numpy(historical[f'keep{k}_taylor'][:count]) for k in (3, 6, 9)}}
        for kind, name in [('gating', 'Gating'), ('joint_only_r2', 'R2_joint_only')]:
            payload = torch.load(args.runs/f'{kind}_seed{seed}/model.pt', map_location='cpu', weights_only=False)
            np.testing.assert_array_equal(payload['global_mean'], original['global_mean'])
            np.testing.assert_array_equal(payload['global_std'], original['global_std'])
            predicted, router = predict(payload, features, kind, args.device)
            methods[name] = predicted if kind == 'gating' else predicted[0]
            complexity.append({'seed': seed, 'method': name, 'parameters': sum(v.numel() for v in router.parameters()),
                               'best_epoch': payload['best_epoch'], 'val_joint_regret': payload['val_joint_regret']})
        complexity.append({'seed': seed, 'method': 'R2', 'parameters': sum(v.numel() for v in r2.parameters()),
                           'best_epoch': None, 'val_joint_regret': None})
        for method, masks in methods.items():
            for keep, mask in masks.items():
                assert (mask.sum(-1)==keep).all()
                mask_archive[f'{seed}_{method}_{keep}'] = mask.numpy()
                scale_table[seed, method, keep, 'hard'] = mask.float()
                scale_table[seed, method, keep, 'shared_r2_merge'] = compose_scales(mask, score, alpha)[0]
    pd.DataFrame(complexity).to_csv(args.output/'complexity.csv', index=False)
    np.savez_compressed(args.output/'masks_and_scales.npz', **mask_archive,
                        **{'scale_'+'_'.join(map(str,key)): value.numpy() for key,value in scale_table.items()})
    model, _ = create_timm_model('vit_base_patch16_224.augreg_in21k_ft_in1k', args.checkpoint)
    model.eval().requires_grad_(False).to(args.device).half()
    wrappers = make_intervenable(model); wrapper = wrappers[11]
    wrapper.capture_input = True
    config = resolve_model_data_config(model)
    paths = [str(args.data/'images'/str(raw).replace('\\','/').split('imagenette2-160/')[-1]) for raw in data['paths'][:count]]
    loader = DataLoader(FrozenPathDataset(np.array(paths), data['target'][:count], create_transform(**config, is_training=False)),
                        batch_size=16, shuffle=False, num_workers=0)
    rows, dense_logits, dense_rows, feature_errors = [], [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch, (images, targets, indices) in enumerate(loader):
            images = images.to(args.device).half(); targets = targets.to(args.device)
            clear_interventions(wrappers)
            wrapper.capture_input = True
            reference = model(images); dense_prediction = reference.argmax(-1)
            dense_logits.append(reference.float().cpu().numpy())
            current_feature = prehead_summary(wrapper.last_input).half().float().cpu().numpy()
            feature_errors.append(float(np.max(np.abs(current_feature-features[indices.numpy()]))))
            for offset, idx in enumerate(indices.tolist()):
                dense_rows.append({'sample': idx, 'dataset_index': int(data['dataset_index'][idx]), 'path': str(data['paths'][idx]),
                                   'target': int(targets[offset]), 'prediction': int(dense_prediction[offset]),
                                   'correct': int(dense_prediction[offset]==targets[offset])})
            for (seed, method, keep, compensation), scales in scale_table.items():
                wrapper.set_head_scales(scales[indices].to(args.device).half())
                logits = model(images); pred = logits.argmax(-1)
                kl = dense_kl(reference, logits)
                assert torch.isfinite(kl).all()
                for offset, idx in enumerate(indices.tolist()):
                    rows.append({'sample': idx, 'dataset_index': int(data['dataset_index'][idx]),
                                 'seed': seed, 'method': method, 'keep': keep, 'compensation': compensation,
                                 'kl': float(kl[offset]), 'correct': int(pred[offset]==targets[offset]),
                                 'prediction': int(pred[offset]), 'dense_prediction': int(dense_prediction[offset]),
                                 'dense_correct': int(dense_prediction[offset]==targets[offset])})
            print(f'evaluation {min((batch+1)*16,count)}/{count} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'per_input.csv', index=False)
    frame.to_json(args.output/'per_input.json', orient='records')
    pd.DataFrame(dense_rows).to_csv(args.output/'dense_inputs.csv', index=False)
    np.savez_compressed(args.output/'dense_logits.npz', logits=np.concatenate(dense_logits), dataset_index=data['dataset_index'][:count])
    checks.update(dense_shared=True, feature_source=str(args.feature_file) if args.feature_file else 'identical historical prehead summaries for every router',
                  live_vs_cached_feature_max_error=max(feature_errors), backbone_sha256=manifest['backbone_sha256'],
                  preprocessing=config, samples=count, elapsed_seconds=time.perf_counter()-start)
    (args.output/'checks.json').write_text(json.dumps(checks, indent=2, default=str))
    # Historical consistency is diagnostic only, never a checkpoint/loss selector.
    comparisons = []
    for seed in (17,23,41):
        old = pd.read_csv(args.data/f'historical_seed{seed}.csv')
        for method, policy in [('R2','dynamic_cfs_router'), ('Taylor','taylor')]:
            for comp, suffix in [('hard','prune'),('shared_r2_merge','merge' if method=='R2' else 'router_merge')]:
                for keep in (3,6,9):
                    a = frame.query('seed==@seed and method==@method and compensation==@comp and keep==@keep').sort_values('sample')
                    b = old[(old.policy==f'{policy}_{suffix}') & (old.keep==keep)].sort_values('sample').iloc[:count]
                    assert np.array_equal(a.dataset_index, b.dataset_index)
                    delta = a.kl.to_numpy()-b.kl_to_dense.to_numpy()
                    comparisons.append({'seed':seed,'method':method,'compensation':comp,'keep':keep,
                                        'current_kl':a.kl.mean(),'historical_kl':b.kl_to_dense.mean(),
                                        'mean_delta':delta.mean(),'max_abs_delta':abs(delta).max(),
                                        'accuracy_delta':a.correct.mean()-b.correct.mean()})
    pd.DataFrame(comparisons).to_csv(args.output/'historical_comparison.csv',index=False)


if __name__ == '__main__':
    main()
