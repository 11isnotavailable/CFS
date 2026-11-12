"""Multi-seed aggregation and paired-image bootstrap for CFS execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    summaries, samples = [], []
    for seed_index, directory in enumerate(args.input_dirs):
        summary = pd.read_csv(directory / "execution_summary.csv")
        summary["seed_index"] = seed_index
        summaries.append(summary)
        sample = pd.read_csv(directory / "execution_samples.csv")
        sample["seed_index"] = seed_index
        samples.append(sample)
    summary = pd.concat(summaries, ignore_index=True)
    sample = pd.concat(samples, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = summary.groupby(
        ["keep", "keep_fraction", "policy"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_seed_std=("accuracy", "std"),
        agreement_mean=("dense_agreement", "mean"),
        agreement_seed_std=("dense_agreement", "std"),
        kl_mean=("mean_kl_to_dense", "mean"),
        kl_seed_std=("mean_kl_to_dense", "std"),
    )
    aggregate.to_csv(args.output_dir / "aggregate_multiseed.csv", index=False)

    router_name = "dynamic_cfs_router_prune"
    baselines = ("taylor_prune", "magnitude_prune", "current_l2_prune", "static_cfs_prune")
    metrics = ("correct", "dense_agreement", "kl_to_dense")
    rng = np.random.default_rng(args.seed)
    rows = []
    for keep in sorted(sample.keep.unique()):
        local = sample[sample.keep == keep]
        router = local[local.policy == router_name].groupby("sample")[list(metrics)].mean()
        for baseline_name in baselines:
            # Baselines are deterministic and identical in every router seed.
            baseline = local[
                (local.policy == baseline_name) & (local.seed_index == 0)
            ].set_index("sample")
            for metric in metrics:
                delta = (router[metric] - baseline[metric]).to_numpy()
                draws = rng.integers(
                    0, len(delta), size=(args.bootstrap, len(delta))
                )
                boot = delta[draws].mean(axis=1)
                low, high = interval(boot)
                rows.append(
                    {
                        "keep": keep,
                        "router": router_name,
                        "baseline": baseline_name,
                        "metric": metric,
                        "mean_delta_router_minus_baseline": float(delta.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "probability_delta_gt_zero": float(np.mean(boot > 0)),
                        "images": len(delta),
                        "router_seeds": len(args.input_dirs),
                    }
                )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    print(aggregate.to_string(index=False))
    print("\nPAIRED ROUTER CONTRASTS\n", contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
