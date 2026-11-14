"""Predeclared head gating and joint-only R2 structure controls (no test loading)."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.dynamic_cfs_router import DynamicCFSRouter, candidate_subset_utilities
from src.learned_head_gating import (LearnedHeadGating, additive_utilities,
                                     joint_ranking_loss, assert_disjoint)


def utilities(model, feature, masks, kind):
    prediction = model(feature)
    if kind == 'gating':
        return torch.stack([additive_utilities(prediction, masks[:, k]) for k in range(3)], 1)
    scores, importance, _ = prediction
    return torch.stack([candidate_subset_utilities(scores, importance.softmax(-1), masks[:, k])
                        for k in range(3)], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['gating', 'joint_only_r2'], default='gating')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    train, tune = [dict(np.load(args.data / f'{split}.npz')) for split in ('train', 'tune')]
    assert_disjoint(train['paths'], tune['paths'])
    mean = train['feature'].mean(0); std = train['feature'].std(0).clip(min=1e-5)
    if args.smoke:
        train = {k: v[:64] for k, v in train.items()}
    def tensors(data):
        return (torch.from_numpy((data['feature'] - mean) / std),
                torch.from_numpy(data['masks'].astype(bool)),
                torch.from_numpy(data['kl'].astype(np.float32)))
    loader = DataLoader(TensorDataset(*tensors(train)), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    val = [v.to(args.device) for v in tensors(tune)]
    model = (LearnedHeadGating() if args.kind == 'gating' else DynamicCFSRouter(2304, 12)).to(args.device)
    if args.kind != 'gating':
        # Alpha is not used by hard subset ranking; do not count it as optimized.
        model.alpha_output.requires_grad_(False)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=3e-4, weight_decay=1e-4)
    config = {**vars(args), 'optimizer': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4,
              'batch': 64, 'joint_loss_weight': 0.5, 'target_temperature': 0.5,
              'prediction_temperature': 0.05, 'parameters': sum(v.numel() for v in model.parameters()),
              'optimized_parameters': sum(v.numel() for v in model.parameters() if v.requires_grad),
              'checkpoint_rule': 'minimum tune candidate-set regret; first epoch on ties',
              'test_loaded': False, 'hyperparameter_trials': 1, 'keep_counts': [3, 6, 9]}
    (args.output / 'config.json').write_text(json.dumps(config, default=str, indent=2))
    best, history, start = float('inf'), [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in loader:
            feature, masks, kl = [v.to(args.device) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * joint_ranking_loss(utilities(model, feature, masks, args.kind), kl)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            utility = utilities(model, val[0], val[1], args.kind)
            chosen = val[2].gather(-1, utility.argmax(-1, keepdim=True)).squeeze(-1)
            regret = float((chosen - val[2].min(-1).values).mean())
        history.append({'epoch': epoch, 'train_loss': np.mean(losses), 'val_joint_regret': regret})
        if regret < best:
            best = regret
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'global_mean': mean, 'global_std': std, 'config': config,
                        'best_epoch': epoch, 'val_joint_regret': best}, args.output / 'model.pt')
        pd.DataFrame(history).to_csv(args.output / 'training_history.csv', index=False)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f'{args.kind} seed={args.seed} epoch={epoch} loss={np.mean(losses):.6f} regret={regret:.6f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    (args.output / 'complete.json').write_text(json.dumps({'best_regret': best, 'seconds': time.perf_counter()-start}))


if __name__ == '__main__':
    main()
