"""Paired endpoint-sensitivity analysis for one Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_qwen_cfs import expected_best_two, load_shards, minimum_cover


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="label=directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    rows = []
    for item in args.runs:
        label, raw_directory = item.split("=", 1)
        directory = Path(raw_directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        data = load_shards(directory)
        drops = data["single_drop"].astype(np.float64)
        substitution = data["substitution"].astype(np.float64)
        scores = 1 - substitution / np.maximum(drops[..., None], 1e-12)
        heads = int(metadata["heads"])
        diagonal = np.arange(heads)
        scores[..., diagonal, diagonal] = np.nan
        substitution[..., diagonal, diagonal] = np.nan
        for sample in range(len(drops)):
            metrics = []
            for layer_offset in range(len(metadata["layers"])):
                local = scores[sample, layer_offset]
                local_drop = drops[sample, layer_offset]
                best_sub = np.nanmin(substitution[sample, layer_offset], axis=-1)
                metrics.append({
                    "raw_oracle": float(np.nanmax(local, axis=-1).mean()),
                    "matched_2": float(np.mean([
                        expected_best_two(row[np.isfinite(row)]) for row in local
                    ])),
                    "weighted_recovery": float(
                        np.maximum(local_drop - best_sub, 0).sum() / local_drop.sum()
                    ),
                    "cover_per_head": minimum_cover(
                        local, np.ones(heads, dtype=bool), 0.5
                    ) / heads,
                    "mean_drop_kl": float(local_drop.mean()),
                })
            averaged = pd.DataFrame(metrics).mean().to_dict()
            rows.append({"endpoint": label, "sample": sample, **averaged})
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "sample_metrics.csv", index=False)
    metric_names = [
        "raw_oracle", "matched_2", "weighted_recovery", "cover_per_head", "mean_drop_kl"
    ]
    rng = np.random.default_rng(args.seed)
    summary_rows = []
    for endpoint, group in frame.groupby("endpoint"):
        for metric in metric_names:
            values = group.sort_values("sample")[metric].to_numpy()
            indices = rng.integers(0, len(values), size=(args.bootstrap, len(values)))
            boot = values[indices].mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            summary_rows.append({
                "endpoint": endpoint, "metric": metric, "mean": values.mean(),
                "ci95_low": low, "ci95_high": high,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
