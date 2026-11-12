"""Paired cross-scale statistics for completed random-position Qwen CFS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_qwen_cfs import expected_best_two, load_shards, minimum_cover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cover-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    records: list[dict[str, float | int | str]] = []
    schedule_hash = None
    for model_order, directory in enumerate(args.input_dirs):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        current_hash = metadata["prediction_position_schedule_sha256"]
        if schedule_hash is None:
            schedule_hash = current_hash
        elif current_hash != schedule_hash:
            raise ValueError(f"Position schedule mismatch in {directory}")
        data = load_shards(directory)
        drops = data["single_drop"].astype(np.float64)
        substitution = data["substitution"].astype(np.float64)
        scores = 1 - substitution / np.maximum(drops[..., None], 1e-12)
        heads = int(metadata["heads"])
        diagonal = np.arange(heads)
        scores[..., diagonal, diagonal] = np.nan
        substitution[..., diagonal, diagonal] = np.nan
        for sample in range(len(drops)):
            for layer_offset, layer in enumerate(metadata["layers"]):
                local_scores = scores[sample, layer_offset]
                local_drops = drops[sample, layer_offset]
                best_substitution = np.nanmin(substitution[sample, layer_offset], axis=-1)
                best_scores = np.nanmax(local_scores, axis=-1)
                recovered = np.maximum(local_drops - best_substitution, 0)
                matched = [
                    expected_best_two(row[np.isfinite(row)]) for row in local_scores
                ]
                cover = minimum_cover(
                    local_scores, np.ones(heads, dtype=bool), args.cover_threshold
                )
                records.append({
                    "model": metadata["model"],
                    "model_order": model_order,
                    "parameters": metadata["parameters"],
                    "heads": heads,
                    "sample": sample,
                    "layer": layer,
                    "depth_fraction": (layer + 1) / metadata["depth"],
                    "raw_oracle_all": float(np.mean(best_scores)),
                    "matched_best_of_2_all": float(np.mean(matched)),
                    "weighted_recovery": float(recovered.sum() / max(local_drops.sum(), 1e-30)),
                    "cover_per_head_all": cover / heads,
                    "mean_drop_kl": float(np.mean(local_drops)),
                })
    frame = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "sample_layer_metrics.csv", index=False)

    metrics = [
        "raw_oracle_all", "matched_best_of_2_all", "weighted_recovery",
        "cover_per_head_all", "mean_drop_kl",
    ]
    rng = np.random.default_rng(args.seed)
    aggregate_rows = []
    context_frame = frame.groupby(
        ["model_order", "model", "parameters", "sample"], as_index=False
    )[metrics].mean()
    for keys, group in context_frame.groupby(["model_order", "model", "parameters"]):
        order, model, parameters = keys
        for metric in metrics:
            values = group[metric].to_numpy()
            indices = rng.integers(0, len(values), size=(args.bootstrap, len(values)))
            boot = values[indices].mean(axis=1)
            low, high = percentile_interval(boot)
            aggregate_rows.append({
                "model_order": order, "model": model, "parameters": parameters,
                "metric": metric, "mean": float(values.mean()),
                "ci95_low": low, "ci95_high": high, "contexts": len(values),
            })
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["metric", "model_order"])
    aggregate.to_csv(args.output_dir / "aggregate_bootstrap.csv", index=False)

    contrast_rows = []
    orders = sorted(context_frame["model_order"].unique())
    for left_order, right_order in zip(orders[:-1], orders[1:]):
        left = context_frame[context_frame.model_order == left_order].sort_values("sample")
        right = context_frame[context_frame.model_order == right_order].sort_values("sample")
        if not np.array_equal(left["sample"].to_numpy(), right["sample"].to_numpy()):
            raise ValueError("Context indices do not match for paired contrast")
        for metric in metrics:
            delta = right[metric].to_numpy() - left[metric].to_numpy()
            indices = rng.integers(0, len(delta), size=(args.bootstrap, len(delta)))
            boot = delta[indices].mean(axis=1)
            low, high = percentile_interval(boot)
            contrast_rows.append({
                "left_model": left.iloc[0]["model"],
                "right_model": right.iloc[0]["model"],
                "metric": metric, "mean_delta_right_minus_left": float(delta.mean()),
                "ci95_low": low, "ci95_high": high,
                "probability_delta_gt_zero": float(np.mean(boot > 0)),
                "contexts": len(delta),
            })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    print(aggregate.to_string(index=False))
    print("\nPAIRED CONTRASTS\n", contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
