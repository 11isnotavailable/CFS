"""Collect true joint-subset KL labels around CFS and pruning policies."""

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
import torch
from timm.data import create_transform, resolve_model_data_config

from scripts.evaluate_dynamic_cfs_execution import FrozenPathDataset, load_targets, remap_paths
from scripts.run_experiment0 import configure_cache, run_variants
from scripts.train_dynamic_cfs_router import (
    choose_top,
    load_router_data,
    similarity_scores,
    static_masks_for_train,
)
from src.dynamic_cfs_router import DynamicCFSRouter, selected_masks, subset_masks
from src.head_intervention import clear_interventions, make_intervenable
from src.model_loading import create_timm_model


ANCHORS = (
    "pairwise_cfs_oracle",
    "dynamic_cfs_router",
    "deletion_oracle",
    "taylor",
    "magnitude",
    "current_l2",
    "static_cfs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--keep-counts", default="3,6,9")
    parser.add_argument("--candidates-per-budget", type=int, default=48)
    parser.add_argument("--variant-batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_router_predictions(path: Path, features: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    model = DynamicCFSRouter(
        global_dim=int(payload["global_dim"]),
        heads=int(payload["heads"]),
        hidden_dim=int(payload["hidden_dim"]),
        embedding_dim=int(payload["embedding_dim"]),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    normalized = (features - payload["global_mean"]) / payload["global_std"]
    score_parts, importance_parts = [], []
    with torch.inference_mode():
        for begin in range(0, len(normalized), 512):
            scores, importance, _ = model(
                torch.from_numpy(normalized[begin : begin + 512]).float()
            )
            score_parts.append(scores)
            importance_parts.append(importance.softmax(dim=-1))
    return torch.cat(score_parts), torch.cat(importance_parts)


def select_in_chunks(
    scores: torch.Tensor,
    importance: torch.Tensor,
    masks: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    pieces = []
    with torch.inference_mode():
        for begin in range(0, len(scores), chunk):
            selected, _ = selected_masks(
                scores[begin : begin + chunk],
                importance[begin : begin + chunk],
                masks,
            )
            pieces.append(selected)
    return torch.cat(pieces)


def build_anchor_masks(
    data: dict[str, np.ndarray],
    router_scores: torch.Tensor,
    router_importance: torch.Tensor,
    keep_counts: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    true_scores = torch.from_numpy(np.nan_to_num(data["scores"], nan=-1.0)).float()
    drops = torch.from_numpy(data["drops"]).float()
    heads = true_scores.shape[-1]
    l2_scores = similarity_scores(data["l2"])
    uniform = torch.ones_like(drops)
    static = static_masks_for_train(data, keep_counts)
    result = {}
    for keep in keep_counts:
        masks = subset_masks(heads, keep)
        result[keep] = {
            "pairwise_cfs_oracle": select_in_chunks(true_scores, drops, masks),
            "dynamic_cfs_router": select_in_chunks(
                router_scores, router_importance, masks
            ),
            "deletion_oracle": choose_top(drops, keep),
            "taylor": choose_top(torch.from_numpy(data["taylor"]), keep),
            "magnitude": choose_top(torch.from_numpy(data["norm"]), keep),
            "current_l2": select_in_chunks(l2_scores, uniform, masks),
            "static_cfs": static[keep][None].expand(len(drops), -1),
        }
    return result


def candidate_pool(
    anchors: dict[str, torch.Tensor],
    sample: int,
    keep: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1009 * sample + 37 * keep)
    heads = anchors[ANCHORS[0]].shape[-1]
    unique: list[tuple[int, ...]] = []
    lookup: dict[tuple[int, ...], int] = {}

    def add(mask) -> int:
        key = tuple(int(value) for value in np.asarray(mask, dtype=np.uint8))
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(key)
        return lookup[key]

    anchor_indices = []
    for name in ANCHORS:
        anchor_indices.append(add(anchors[name][sample].numpy()))
    mutations = []
    for name in ("pairwise_cfs_oracle", "dynamic_cfs_router"):
        base = anchors[name][sample].numpy().astype(bool)
        kept, dropped = np.where(base)[0], np.where(~base)[0]
        for remove in kept:
            for insert in dropped:
                value = base.copy()
                value[remove] = False
                value[insert] = True
                mutations.append(value)
    rng.shuffle(mutations)
    for value in mutations:
        if len(unique) >= count:
            break
        add(value)
    while len(unique) < count:
        value = np.zeros(heads, dtype=np.uint8)
        value[rng.choice(heads, keep, replace=False)] = 1
        add(value)
    return np.asarray(unique[:count], dtype=np.uint8), np.asarray(anchor_indices)


def save_shard(path: Path, records: list[dict]) -> None:
    payload = {key: np.stack([record[key] for record in records]) for key in records[0]}
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    configure_cache()
    keep_counts = [int(value) for value in args.keep_counts.split(",")]
    data = load_router_data(args.oracle_dir, args.layer_index)
    frozen_paths, targets = load_targets(args.oracle_dir)
    if not np.array_equal(frozen_paths.astype(str), data["paths"].astype(str)):
        raise ValueError("Oracle path ordering mismatch")
    count = len(frozen_paths) if args.max_samples is None else min(args.max_samples, len(frozen_paths))
    for key, value in list(data.items()):
        if isinstance(value, np.ndarray) and len(value) == len(frozen_paths):
            data[key] = value[:count]
    frozen_paths, targets = frozen_paths[:count], targets[:count]
    paths = remap_paths(frozen_paths, args.path_root)
    router_scores, router_importance = load_router_predictions(
        args.router_checkpoint, data["global"]
    )
    anchors_by_budget = build_anchor_masks(
        data, router_scores, router_importance, keep_counts
    )

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32
    model, _ = create_timm_model(args.model, args.checkpoint)
    model.eval().to(device=device, dtype=dtype).requires_grad_(False)
    wrappers = make_intervenable(model)
    wrapper = wrappers[args.layer]
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    dataset = FrozenPathDataset(paths, targets, transform)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "oracle_dir": str(args.oracle_dir.resolve()),
                "router_checkpoint": str(args.router_checkpoint.resolve()),
                "model": args.model,
                "layer": args.layer,
                "keep_counts": keep_counts,
                "candidates_per_budget": args.candidates_per_budget,
                "anchors": ANCHORS,
                "samples": count,
                "label": "true downstream KL under simultaneous hard head gating",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    records, started = [], time.perf_counter()
    for sample in range(count):
        shard = sample // args.shard_size
        shard_path = args.output_dir / f"shard_{shard:05d}.npz"
        if shard_path.exists():
            if sample % args.shard_size == 0:
                print(f"skip existing {shard_path.name}")
            continue
        image, target, _ = dataset[sample]
        image = image[None].to(device=device, dtype=dtype)
        clear_interventions(wrappers)
        with torch.inference_mode():
            reference = model(image)
        masks_all, errors_all, anchor_indices_all = [], [], []
        for keep in keep_counts:
            masks, anchor_indices = candidate_pool(
                anchors_by_budget[keep], sample, keep,
                args.candidates_per_budget, args.seed,
            )
            _, errors = run_variants(
                model, wrapper, image, torch.from_numpy(masks).float(),
                reference, args.variant_batch_size, 1.0,
            )
            masks_all.append(masks)
            errors_all.append(errors.numpy().astype(np.float32))
            anchor_indices_all.append(anchor_indices)
        records.append(
            {
                "dataset_index": np.asarray(data["dataset_index"][sample]),
                "path": np.asarray(frozen_paths[sample]),
                "target": np.asarray(target),
                "dense_prediction": np.asarray(int(reference.argmax(dim=-1))),
                "candidate_masks": np.stack(masks_all),
                "joint_kl": np.stack(errors_all),
                "anchor_indices": np.stack(anchor_indices_all),
            }
        )
        if len(records) == args.shard_size or sample + 1 == count:
            save_shard(shard_path, records)
            records = []
        elapsed = time.perf_counter() - started
        eta = elapsed / (sample + 1) * (count - sample - 1)
        if sample == 0 or sample + 1 == count or (sample + 1) % args.log_every == 0:
            print(
                f"sample={sample + 1}/{count} elapsed={elapsed/60:.1f}m "
                f"eta={eta/60:.1f}m"
            )


if __name__ == "__main__":
    main()
