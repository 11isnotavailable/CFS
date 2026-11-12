"""Execute learned CFS keep sets as real ViT head interventions.

This measures functional fidelity and accuracy at a logical head budget. The
current timm QKV projection is fused, so these gates do not yet imply wall-clock
speedup; they establish whether the learned routing decisions are usable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

from scripts.run_experiment0 import configure_cache
from scripts.train_dynamic_cfs_router import load_router_data
from src.dynamic_cfs_router import compose_scales
from src.head_intervention import clear_interventions, make_intervenable
from src.model_loading import create_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--router-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="vit_small_patch16_224.augreg_in1k")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--path-root",
        type=Path,
        help="Optional local replacement for the frozen dataset's val root.",
    )
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--keep-counts", default="2,3,4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class FrozenPathDataset(Dataset):
    def __init__(self, paths: np.ndarray, targets: np.ndarray, transform) -> None:
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(str(self.paths[index])) as image:
            value = self.transform(image.convert("RGB"))
        return value, int(self.targets[index]), index


def load_targets(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    paths, targets = [], []
    for file in sorted(directory.glob("shard_*.npz")):
        archive = np.load(file)
        paths.append(archive["path"])
        targets.append(archive["imagenet_target"])
    if not paths:
        raise FileNotFoundError(f"No shards in {directory}")
    return np.concatenate(paths), np.concatenate(targets).astype(np.int64)


def remap_paths(paths: np.ndarray, root: Path | None) -> np.ndarray:
    if root is None:
        return paths
    remapped = []
    for raw in paths.astype(str):
        normalized = raw.replace("\\", "/")
        marker = "/val/"
        if marker not in normalized:
            raise ValueError(f"Cannot locate /val/ in frozen path: {raw}")
        relative = normalized.split(marker, 1)[1]
        remapped.append(str(root / Path(relative)))
    return np.asarray(remapped)


def dense_kl(reference: torch.Tensor, variant: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(reference.float(), dim=-1)
    log_q = F.log_softmax(variant.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).clamp_min(0)


def random_masks(samples: int, heads: int, keep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + keep)
    masks = np.zeros((samples, heads), dtype=bool)
    for sample in range(samples):
        masks[sample, rng.choice(heads, keep, replace=False)] = True
    return masks


def main() -> None:
    args = parse_args()
    configure_cache()
    keep_counts = [int(value) for value in args.keep_counts.split(",")]
    oracle = load_router_data(args.oracle_dir, args.layer_index)
    paths, targets = load_targets(args.oracle_dir)
    prediction = np.load(args.router_predictions)
    if not np.array_equal(paths.astype(str), prediction["paths"].astype(str)):
        raise ValueError("Router predictions and oracle paths are not aligned")
    paths = remap_paths(paths, args.path_root)
    count = len(paths) if args.max_samples is None else min(args.max_samples, len(paths))
    paths, targets = paths[:count], targets[:count]
    true_scores = np.nan_to_num(oracle["scores"][:count], nan=-1.0).astype(np.float32)
    true_alpha = np.nan_to_num(oracle["alpha"][:count], nan=1.0).astype(np.float32)
    predicted_scores = prediction["predicted_scores"][:count].astype(np.float32)
    predicted_alpha = prediction["predicted_alpha"][:count].astype(np.float32)
    heads = true_scores.shape[-1]

    model, _ = create_timm_model(args.model, args.checkpoint)
    model.eval().to(args.device)
    if args.fp16:
        model.half()
    wrappers = make_intervenable(model)
    wrapper = wrappers[args.layer]
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    dataset = FrozenPathDataset(paths, targets, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )

    scale_table: dict[tuple[int, str], np.ndarray] = {}
    mask_table: dict[tuple[int, str], np.ndarray] = {}
    for keep in keep_counts:
        policies = (
            "static_cfs",
            "current_l2",
            "magnitude",
            "taylor",
            "deletion_oracle",
            "dynamic_cfs_router",
            "pairwise_cfs_oracle",
        )
        mask_table[(keep, "random")] = random_masks(
            count, heads, keep, args.seed
        )
        for policy in policies:
            mask_table[(keep, policy)] = prediction[
                f"keep{keep}_{policy}"
            ][:count].astype(bool)
        for policy in ("random",) + policies:
            scale_table[(keep, f"{policy}_prune")] = mask_table[
                (keep, policy)
            ].astype(np.float32)
        dynamic_scale, _ = compose_scales(
            torch.from_numpy(mask_table[(keep, "dynamic_cfs_router")]),
            torch.from_numpy(predicted_scores),
            torch.from_numpy(predicted_alpha),
        )
        oracle_scale, _ = compose_scales(
            torch.from_numpy(mask_table[(keep, "pairwise_cfs_oracle")]),
            torch.from_numpy(true_scores),
            torch.from_numpy(true_alpha),
        )
        scale_table[(keep, "dynamic_cfs_router_merge")] = dynamic_scale.numpy()
        scale_table[(keep, "pairwise_cfs_oracle_merge")] = oracle_scale.numpy()
        router_oracle_scale, _ = compose_scales(
            torch.from_numpy(mask_table[(keep, "dynamic_cfs_router")]),
            torch.from_numpy(true_scores),
            torch.from_numpy(true_alpha),
        )
        taylor_router_scale, _ = compose_scales(
            torch.from_numpy(mask_table[(keep, "taylor")]),
            torch.from_numpy(predicted_scores),
            torch.from_numpy(predicted_alpha),
        )
        taylor_oracle_scale, _ = compose_scales(
            torch.from_numpy(mask_table[(keep, "taylor")]),
            torch.from_numpy(true_scores),
            torch.from_numpy(true_alpha),
        )
        scale_table[(keep, "dynamic_cfs_router_oracle_merge")] = (
            router_oracle_scale.numpy()
        )
        scale_table[(keep, "taylor_router_merge")] = taylor_router_scale.numpy()
        scale_table[(keep, "taylor_oracle_merge")] = taylor_oracle_scale.numpy()

    policy_keys = sorted(scale_table, key=lambda item: (item[0], item[1]))
    totals = {
        key: {"correct": 0, "agreement": 0, "kl": 0.0, "samples": 0}
        for key in policy_keys
    }
    dense_correct = 0
    sample_rows = []
    started = time.perf_counter()
    dtype = torch.float16 if args.fp16 else torch.float32
    for batch_number, (images, batch_targets, indices) in enumerate(loader, start=1):
        images = images.to(args.device, dtype=dtype, non_blocking=True)
        batch_targets = batch_targets.to(args.device)
        clear_interventions(wrappers)
        with torch.inference_mode():
            reference = model(images)
        dense_prediction = reference.argmax(dim=-1)
        dense_correct += int((dense_prediction == batch_targets).sum())
        begin, end = int(indices[0]), int(indices[-1]) + 1
        for keep, policy in policy_keys:
            scales = torch.from_numpy(scale_table[(keep, policy)][begin:end]).to(
                args.device, dtype=dtype
            )
            wrapper.set_head_scales(scales)
            with torch.inference_mode():
                logits = model(images)
            prediction_value = logits.argmax(dim=-1)
            kl = dense_kl(reference, logits)
            local_correct = prediction_value == batch_targets
            local_agreement = prediction_value == dense_prediction
            total = totals[(keep, policy)]
            total["correct"] += int(local_correct.sum())
            total["agreement"] += int(local_agreement.sum())
            total["kl"] += float(kl.sum())
            total["samples"] += len(images)
            for offset, sample_index in enumerate(indices.tolist()):
                sample_rows.append(
                    {
                        "sample": sample_index,
                        "dataset_index": int(oracle["dataset_index"][sample_index]),
                        "keep": keep,
                        "policy": policy,
                        "correct": int(local_correct[offset]),
                        "dense_agreement": int(local_agreement[offset]),
                        "kl_to_dense": float(kl[offset]),
                    }
                )
        clear_interventions(wrappers)
        print(
            f"batch={batch_number}/{len(loader)} samples={end}/{count} "
            f"elapsed={(time.perf_counter() - started) / 60:.1f}m"
        )

    rows = [
        {
            "keep": heads,
            "keep_fraction": 1.0,
            "policy": "dense",
            "accuracy": dense_correct / count,
            "dense_agreement": 1.0,
            "mean_kl_to_dense": 0.0,
        }
    ]
    for (keep, policy), total in totals.items():
        rows.append(
            {
                "keep": keep,
                "keep_fraction": keep / heads,
                "policy": policy,
                "accuracy": total["correct"] / total["samples"],
                "dense_agreement": total["agreement"] / total["samples"],
                "mean_kl_to_dense": total["kl"] / total["samples"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "execution_summary.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(
        args.output_dir / "execution_samples.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "layer": args.layer,
                "samples": count,
                "keep_counts": keep_counts,
                "runtime_input": "pre-head residual; predictions frozen before execution",
                "logical_budget_only": True,
                "wall_clock_speedup_claimed": False,
                "merge_rule": "1 + sum_i(alpha_ij - 1), clipped to [0,6]",
                "elapsed_minutes": (time.perf_counter() - started) / 60,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
