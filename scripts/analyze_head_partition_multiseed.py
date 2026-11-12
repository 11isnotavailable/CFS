"""Aggregate trained fixed-width head controls without pseudo-replication."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORE_METRICS = [
    "accuracy", "valid_source_ratio", "oracle_recovery", "matched_best_of_2",
    "top_source_oracle_recovery", "top_source_matched_best_of_2",
    "kl_weighted_recovery", "spectral_rank", "spectral_rank_per_head",
]
SAMPLE_METRICS = [
    "valid_source_ratio", "oracle_recovery", "top_source_oracle_recovery",
    "kl_weighted_recovery", "mean_drop_kl",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry", action="append", nargs=3,
        metavar=("SEED", "SUMMARY_CSV", "SAMPLES_CSV"), required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries, samples = [], []
    for raw_seed, summary_path, samples_path in args.entry:
        seed = int(raw_seed)
        summary = pd.read_csv(summary_path); summary["seed"] = seed
        sample = pd.read_csv(samples_path); sample["seed"] = seed
        summaries.append(summary); samples.append(sample)
    summary = pd.concat(summaries, ignore_index=True)
    sample = pd.concat(samples, ignore_index=True)
    aggregate = summary.groupby("heads")[CORE_METRICS].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    # Treat each trained seed as one replicate for endpoint contrasts.
    endpoint_rows = []
    for seed, group in summary.groupby("seed"):
        indexed = group.set_index("heads")
        for metric in CORE_METRICS:
            endpoint_rows.append({
                "seed": seed, "metric": metric,
                "h12_minus_h3": indexed.loc[12, metric] - indexed.loc[3, metric],
            })
    endpoints = pd.DataFrame(endpoint_rows)
    endpoint_summary = endpoints.groupby("metric")["h12_minus_h3"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    # Correctness is intersected within each seed, then metrics are averaged per
    # seed. Images are not counted as independent training replicates.
    shared_rows = []
    for seed, group in sample.groupby("seed"):
        correct = group.pivot(index="sample", columns="heads", values="accuracy")
        common = correct.eq(1).all(axis=1)
        for heads, head_group in group[group["sample"].isin(common[common].index)].groupby("heads"):
            for metric in SAMPLE_METRICS:
                shared_rows.append({
                    "seed": seed, "heads": heads, "metric": metric,
                    "mean": head_group[metric].mean(),
                    "shared_correct_images": int(common.sum()),
                })
    shared = pd.DataFrame(shared_rows)
    shared_aggregate = shared.groupby(["heads", "metric"])["mean"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "per_seed_summary.csv", index=False)
    sample.to_csv(args.output_dir / "per_seed_samples.csv", index=False)
    aggregate.to_csv(args.output_dir / "seed_aggregate.csv", index=False)
    endpoints.to_csv(args.output_dir / "seed_endpoint_contrasts.csv", index=False)
    endpoint_summary.to_csv(args.output_dir / "seed_endpoint_summary.csv", index=False)
    shared.to_csv(args.output_dir / "shared_correct_per_seed.csv", index=False)
    shared_aggregate.to_csv(args.output_dir / "shared_correct_seed_aggregate.csv", index=False)
    print(aggregate[[
        "heads", "accuracy_mean", "accuracy_std", "oracle_recovery_mean",
        "oracle_recovery_std", "matched_best_of_2_mean",
        "kl_weighted_recovery_mean", "kl_weighted_recovery_std",
        "spectral_rank_mean", "spectral_rank_per_head_mean",
    ]].to_string(index=False))
    print(endpoint_summary[endpoint_summary.metric.isin([
        "oracle_recovery", "matched_best_of_2", "kl_weighted_recovery",
        "spectral_rank", "spectral_rank_per_head", "valid_source_ratio",
    ])].to_string(index=False))


if __name__ == "__main__":
    main()
