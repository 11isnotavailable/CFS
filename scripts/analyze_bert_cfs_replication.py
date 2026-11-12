"""Aggregate the frozen BERT/SST-2 CFS replication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.train_representative_predictor import load_shards


ENDPOINTS = {
    "level0": "Residual",
    "level2": "Q/K",
    "contribution": "Post-WO",
    "gradient": "Gradient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert_cfs_analysis"))
    return parser.parse_args()


def threshold_rows(layer: int) -> list[dict]:
    archive = load_shards(Path(f"outputs/bert_layer{layer}_test512_alpha31"))
    drop = archive["single_drop"][:, 0]
    substitution = archive["substitution"][:, 0]
    scores = 1.0 - substitution / np.maximum(drop[:, :, None], 1e-12)
    heads = scores.shape[-1]
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    random_expected = np.nanmean(scores, axis=-1)
    oracle = np.nanmax(scores, axis=-1)
    rows = []
    for threshold in (1e-7, 1e-6, 1e-5, 1e-4):
        valid = drop > threshold
        rows.append(
            {
                "layer": layer,
                "minimum_drop_kl": threshold,
                "valid_sources": int(valid.sum()),
                "total_sources": int(valid.size),
                "coverage": float(valid.mean()),
                "random_calibrated_recovery": float(random_expected[valid].mean()),
                "oracle_recovery": float(oracle[valid].mean()),
                "oracle_minus_random": float(
                    (oracle[valid] - random_expected[valid]).mean()
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    baseline_rows = []
    for layer in (6, 10):
        for tag, endpoint in ENDPOINTS.items():
            for seed in (17, 23, 41):
                table = pd.read_csv(
                    f"outputs/bert_layer{layer}_{tag}_seed{seed}/policy_summary.csv"
                )
                row = table.loc[table["policy"] == "pair_score_predictor"].iloc[0]
                seed_rows.append(
                    {
                        "layer": layer,
                        "endpoint": endpoint,
                        "seed": seed,
                        "mean_recovery": row["mean_recovery"],
                        "median_recovery": row["median_recovery"],
                        "representative_accuracy": row["representative_accuracy"],
                        "pair_spearman_mean": row["pair_spearman_mean"],
                    }
                )
        baseline = pd.read_csv(
            f"outputs/bert_layer{layer}_gradient_seed17/policy_summary.csv"
        )
        for policy in (
            "random_expected",
            "current_l2",
            "highest_taylor_importance",
            "static_train_representative",
            "functional_oracle",
        ):
            row = baseline.loc[baseline["policy"] == policy].iloc[0]
            baseline_rows.append(
                {
                    "layer": layer,
                    "policy": policy,
                    "mean_recovery": row["mean_recovery"],
                    "median_recovery": row["median_recovery"],
                    "representative_accuracy": row["representative_accuracy"],
                }
            )

    seeds = pd.DataFrame(seed_rows)
    aggregate = (
        seeds.groupby(["layer", "endpoint"], sort=False)
        .agg(
            mean_recovery=("mean_recovery", "mean"),
            seed_std=("mean_recovery", "std"),
            representative_accuracy=("representative_accuracy", "mean"),
            pair_spearman=("pair_spearman_mean", "mean"),
        )
        .reset_index()
    )
    baselines = pd.DataFrame(baseline_rows)
    thresholds = pd.DataFrame(
        threshold_rows(6) + threshold_rows(10)
    )
    seeds.to_csv(args.output_dir / "endpoint_seed_results.csv", index=False)
    aggregate.to_csv(args.output_dir / "endpoint_summary.csv", index=False)
    baselines.to_csv(args.output_dir / "baseline_summary.csv", index=False)
    thresholds.to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)
    print(aggregate.to_string(index=False))
    print(baselines.to_string(index=False))
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
