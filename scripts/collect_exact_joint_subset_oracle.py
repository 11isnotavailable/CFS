"""Enumerate the exact simultaneous hard-gating oracle for ViT heads."""

from __future__ import annotations

import argparse
import itertools
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
from timm.data import create_transform, resolve_model_data_config

from scripts.evaluate_dynamic_cfs_execution import FrozenPathDataset, load_targets, remap_paths
from scripts.run_experiment0 import configure_cache, run_variants
from scripts.train_dynamic_cfs_router import load_router_data
from src.head_intervention import clear_interventions, make_intervenable
from src.model_loading import create_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--keep-counts", default="3,6,9")
    parser.add_argument("--variant-batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def masks_for_budget(heads: int, keep: int) -> np.ndarray:
    values = np.zeros((len(list(itertools.combinations(range(heads), keep))), heads), dtype=np.float32)
    for row, subset in enumerate(itertools.combinations(range(heads), keep)):
        values[row, list(subset)] = 1.0
    return values


def summarize(output_dir: Path, heads: int, keep_counts: list[int]) -> None:
    rows = []
    for path in sorted(output_dir.glob("sample_*.npz")):
        archive = np.load(path)
        for index, keep in enumerate(keep_counts):
            rows.append(
                {
                    "sample": int(archive["sample"]),
                    "dataset_index": int(archive["dataset_index"]),
                    "keep": keep,
                    "dense_correct": int(archive["dense_correct"]),
                    "oracle_kl": float(archive["oracle_kl"][index]),
                    "oracle_correct": int(archive["oracle_correct"][index]),
                    "oracle_dense_agreement": int(archive["oracle_dense_agreement"][index]),
                    "taylor_kl": float(archive["taylor_kl"][index]),
                    "taylor_correct": int(archive["taylor_correct"][index]),
                    "taylor_dense_agreement": int(archive["taylor_dense_agreement"][index]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "exact_joint_oracle_samples.csv", index=False)
    summary = (
        frame.groupby("keep")
        .agg(
            samples=("sample", "size"),
            dense_accuracy=("dense_correct", "mean"),
            oracle_mean_kl=("oracle_kl", "mean"),
            oracle_median_kl=("oracle_kl", "median"),
            oracle_accuracy=("oracle_correct", "mean"),
            oracle_dense_fidelity=("oracle_dense_agreement", "mean"),
            taylor_mean_kl=("taylor_kl", "mean"),
            taylor_median_kl=("taylor_kl", "median"),
            taylor_accuracy=("taylor_correct", "mean"),
            taylor_dense_fidelity=("taylor_dense_agreement", "mean"),
        )
        .reset_index()
    )
    summary["oracle_kl_reduction_vs_taylor"] = 1.0 - summary.oracle_mean_kl / summary.taylor_mean_kl
    summary.to_csv(output_dir / "exact_joint_oracle_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    configure_cache()
    keep_counts = [int(value) for value in args.keep_counts.split(",")]
    oracle = load_router_data(args.oracle_dir, args.layer_index)
    frozen_paths, targets = load_targets(args.oracle_dir)
    paths = remap_paths(frozen_paths, args.path_root)
    count = len(paths) if args.max_samples is None else min(args.max_samples, len(paths))
    paths, targets = paths[:count], targets[:count]
    for key, value in list(oracle.items()):
        if isinstance(value, np.ndarray) and len(value) == len(frozen_paths):
            oracle[key] = value[:count]
    heads = int(oracle["drops"].shape[-1])
    masks_by_keep = {keep: masks_for_budget(heads, keep) for keep in keep_counts}

    model, _ = create_timm_model(args.model, args.checkpoint)
    dtype = torch.float16 if args.fp16 else torch.float32
    model.eval().to(device=args.device, dtype=dtype).requires_grad_(False)
    wrappers = make_intervenable(model)
    wrapper = wrappers[args.layer]
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    dataset = FrozenPathDataset(paths, targets, transform)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "oracle_dir": str(args.oracle_dir.resolve()),
                "model": args.model,
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "layer": args.layer,
                "heads": heads,
                "keep_counts": keep_counts,
                "subset_counts": {str(k): len(v) for k, v in masks_by_keep.items()},
                "samples": count,
                "intervention": "simultaneous hard head gating",
                "selection": "exact exhaustive minimum KL to dense output",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    started = time.perf_counter()
    for sample in range(count):
        output = args.output_dir / f"sample_{sample:05d}.npz"
        if output.exists():
            continue
        image, target, _ = dataset[sample]
        image = image[None].to(device=args.device, dtype=dtype)
        clear_interventions(wrappers)
        with torch.inference_mode():
            reference = model(image)
        dense_prediction = int(reference.argmax(dim=-1))
        oracle_kl, oracle_correct, oracle_agreement, oracle_masks = [], [], [], []
        taylor_kl, taylor_correct, taylor_agreement = [], [], []
        for keep in keep_counts:
            masks = masks_by_keep[keep]
            logits, errors = run_variants(
                model,
                wrapper,
                image,
                torch.from_numpy(masks),
                reference,
                args.variant_batch_size,
                1.0,
            )
            best = int(errors.argmin())
            best_prediction = int(logits[best].argmax())
            oracle_kl.append(float(errors[best]))
            oracle_correct.append(int(best_prediction == target))
            oracle_agreement.append(int(best_prediction == dense_prediction))
            oracle_masks.append(masks[best].astype(np.uint8))

            taylor_heads = np.argsort(oracle["taylor"][sample])[-keep:]
            taylor_mask = np.zeros(heads, dtype=np.float32)
            taylor_mask[taylor_heads] = 1.0
            matches = np.where(np.all(masks == taylor_mask[None], axis=1))[0]
            taylor_index = int(matches[0])
            taylor_prediction = int(logits[taylor_index].argmax())
            taylor_kl.append(float(errors[taylor_index]))
            taylor_correct.append(int(taylor_prediction == target))
            taylor_agreement.append(int(taylor_prediction == dense_prediction))
        np.savez_compressed(
            output,
            sample=np.asarray(sample),
            dataset_index=np.asarray(oracle["dataset_index"][sample]),
            dense_correct=np.asarray(int(dense_prediction == target)),
            oracle_kl=np.asarray(oracle_kl, dtype=np.float32),
            oracle_correct=np.asarray(oracle_correct, dtype=np.uint8),
            oracle_dense_agreement=np.asarray(oracle_agreement, dtype=np.uint8),
            oracle_masks=np.asarray(oracle_masks, dtype=np.uint8),
            taylor_kl=np.asarray(taylor_kl, dtype=np.float32),
            taylor_correct=np.asarray(taylor_correct, dtype=np.uint8),
            taylor_dense_agreement=np.asarray(taylor_agreement, dtype=np.uint8),
        )
        elapsed = time.perf_counter() - started
        eta = elapsed / (sample + 1) * (count - sample - 1)
        print(f"sample={sample + 1}/{count} elapsed={elapsed / 60:.2f}m eta={eta / 60:.2f}m", flush=True)
    clear_interventions(wrappers)
    summarize(args.output_dir, heads, keep_counts)


if __name__ == "__main__":
    main()
