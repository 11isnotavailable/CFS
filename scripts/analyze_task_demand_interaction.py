"""Analyze the preregistered task-by-scale CFS interaction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_functional_capacity import (
    exact_minimum_cover,
    spectral_effective_rank,
    subset_table,
)
from scripts.analyze_scaling_cfs import expected_best_of_k, shard_field
from scripts.train_representative_predictor import load_shards


SIZES = ("tiny", "small", "base", "large")
TASKS = ("coarse", "fine")
PRIMARY_METRICS = (
    "oracle_recovery",
    "matched_best_of_2",
    "kl_weighted_recovery",
    "mutual_density",
    "cover_ratio",
    "cover_to_valid_ratio",
    "spectral_rank_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def analyze_entry(
    task: str,
    size: str,
    directory: Path,
    minimum_drop: float,
    threshold: float,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    archive = load_shards(directory)
    substitution = archive["substitution"][:, 0].astype(np.float64)
    drops = archive["single_drop"][:, 0].astype(np.float64)
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[-1]
    diagonal = np.arange(heads)
    scores[:, diagonal, diagonal] = np.nan
    valid = drops > minimum_drop
    oracle = np.nanmax(scores, axis=-1)

    substitution_without_self = substitution.copy()
    substitution_without_self[:, diagonal, diagonal] = np.nan
    best_substitution = np.nanmin(substitution_without_self, axis=-1)
    recovered_kl = drops - best_substitution
    weighted = np.divide(
        recovered_kl.sum(axis=1), drops.sum(axis=1),
        out=np.full(len(drops), np.nan), where=drops.sum(axis=1) > 0,
    )

    matched_by_source = np.full(valid.shape, np.nan, dtype=np.float64)
    candidate_rows = scores[valid]
    finite = candidate_rows[np.isfinite(candidate_rows)].reshape(
        int(valid.sum()), heads - 1
    )
    matched_by_source[valid] = expected_best_of_k(finite, min(2, heads - 1))

    selections, cardinality = subset_table(heads)
    cover = exact_minimum_cover(
        scores, valid, threshold, selections, cardinality
    )
    rank = spectral_effective_rank(scores, valid)
    valid_counts = valid.sum(axis=1)
    adjacency = (scores > threshold) & valid[:, :, None]
    mutual = adjacency & adjacency.transpose(0, 2, 1)
    valid_pairs = valid[:, :, None] & valid[:, None, :]
    valid_pairs[:, diagonal, diagonal] = False

    rows = []
    for sample in range(len(scores)):
        pair_mask = valid_pairs[sample]
        rows.append({
            "task": task,
            "model_size": size,
            "sample": sample,
            "heads": heads,
            "accuracy": np.nan,
            "valid_source_ratio": float(valid[sample].mean()),
            "oracle_recovery": safe_mean(oracle[sample, valid[sample]]),
            "matched_best_of_2": safe_mean(matched_by_source[sample]),
            "kl_weighted_recovery": float(weighted[sample]),
            "directed_density": safe_mean(adjacency[sample][pair_mask]),
            "mutual_density": safe_mean(mutual[sample][pair_mask]),
            "mean_minimum_cover": float(cover[sample]),
            "cover_ratio": float(cover[sample] / heads),
            "cover_to_valid_ratio": (
                float(cover[sample] / valid_counts[sample])
                if valid_counts[sample] else np.nan
            ),
            "spectral_rank": float(rank[sample]),
            "spectral_rank_ratio": float(rank[sample] / heads),
        })
    frame = pd.DataFrame(rows)
    prediction = shard_field(directory, "prediction")
    target = shard_field(directory, "imagenet_target")
    frame["accuracy"] = (prediction == target).astype(float)
    summary = {
        "task": task,
        "model_size": size,
        "model": metadata["model"],
        "heads": heads,
        "samples": len(frame),
        **{
            column: float(frame[column].mean())
            for column in frame.columns
            if column not in {"task", "model_size", "sample", "heads"}
        },
    }
    return summary, frame, np.asarray(metadata["sample_indices"], dtype=np.int64)


def bootstrap_interaction(
    samples: pd.DataFrame, metric: str, repeats: int, seed: int
) -> dict:
    pivot = samples.pivot(index="sample", columns=("task", "model_size"), values=metric)
    required = [(task, size) for task in TASKS for size in ("base", "large")]
    pivot = pivot.dropna(subset=required)
    values = (
        (pivot[("coarse", "large")] - pivot[("coarse", "base")])
        - (pivot[("fine", "large")] - pivot[("fine", "base")])
    ).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        draws[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "metric": metric,
        "samples": len(values),
        "interaction": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    args = parse_args()
    summaries = []
    sample_frames = []
    reference_indices = None
    for task in TASKS:
        for size in SIZES:
            directory = args.root / f"{task}_{size}"
            summary, frame, indices = analyze_entry(
                task, size, directory, args.minimum_drop_kl, args.threshold
            )
            if reference_indices is None:
                reference_indices = indices
            elif not np.array_equal(indices, reference_indices):
                raise RuntimeError(f"Sample mismatch for {task}/{size}")
            summaries.append(summary)
            sample_frames.append(frame)
    summary_frame = pd.DataFrame(summaries)
    samples = pd.concat(sample_frames, ignore_index=True)
    interactions = pd.DataFrame([
        bootstrap_interaction(samples, metric, args.bootstrap, args.seed)
        for metric in PRIMARY_METRICS
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.output_dir / "task_demand_summary.csv", index=False)
    samples.to_csv(args.output_dir / "task_demand_samples.csv", index=False)
    interactions.to_csv(args.output_dir / "task_by_scale_interactions.csv", index=False)
    print(summary_frame.to_string(index=False))
    print(interactions.to_string(index=False))


if __name__ == "__main__":
    main()
