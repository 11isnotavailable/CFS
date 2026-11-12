"""Aggregate the frozen BERT/MNLI CFS replication."""

from __future__ import annotations

import argparse
import json
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
    "contribution_only": "Post-WO",
    "gradient_only": "Gradient",
}
SEEDS = (17, 23, 41)


def shard_field(directory: Path, name: str) -> np.ndarray:
    values = []
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as archive:
            values.append(archive[name])
    if not values:
        raise FileNotFoundError(f"No oracle shards found in {directory}")
    return np.concatenate(values, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/bert_mnli_cfs_layer10"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bert_mnli_cfs_analysis"),
    )
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict] = []
    for tag, endpoint in ENDPOINTS.items():
        for seed in SEEDS:
            table = pd.read_csv(
                args.root / f"predictor_{tag}_seed{seed}" / "policy_summary.csv"
            )
            row = table.loc[table["policy"] == "pair_score_predictor"].iloc[0]
            seed_rows.append(
                {
                    "endpoint": endpoint,
                    "seed": seed,
                    "mean_recovery": row["mean_recovery"],
                    "median_recovery": row["median_recovery"],
                    "representative_accuracy": row["representative_accuracy"],
                    "pair_spearman_mean": row["pair_spearman_mean"],
                }
            )

    seeds = pd.DataFrame(seed_rows)
    endpoint_summary = (
        seeds.groupby("endpoint", sort=False)
        .agg(
            mean_recovery=("mean_recovery", "mean"),
            seed_std=("mean_recovery", "std"),
            representative_accuracy=("representative_accuracy", "mean"),
            pair_spearman=("pair_spearman_mean", "mean"),
        )
        .reset_index()
    )

    reference = pd.read_csv(
        args.root / "predictor_gradient_only_seed17" / "policy_summary.csv"
    )
    policies = (
        "random_expected",
        "current_l2",
        "highest_taylor_importance",
        "signed_ce_first_order",
        "signed_logit_first_order",
        "static_train_representative",
        "functional_oracle",
    )
    baselines = reference.loc[reference["policy"].isin(policies)].copy()

    oracle = load_shards(args.root / "test_oracle")
    drops = oracle["single_drop"][:, 0]
    substitution = oracle["substitution"][:, 0]
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[-1]
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    valid = drops > args.minimum_drop_kl
    predictions = shard_field(args.root / "test_oracle", "prediction")
    labels = shard_field(args.root / "test_oracle", "dataset_label")
    accuracy = float(np.mean(predictions == labels))
    diagnostics = {
        "test_examples": int(len(drops)),
        "heads": int(heads),
        "test_accuracy": accuracy,
        "minimum_drop_kl": args.minimum_drop_kl,
        "valid_sources": int(valid.sum()),
        "total_sources": int(valid.size),
        "source_coverage": float(valid.mean()),
        "oracle_recovery_direct": float(np.nanmax(scores, axis=-1)[valid].mean()),
        "random_recovery_direct": float(np.nanmean(scores, axis=-1)[valid].mean()),
    }

    seeds.to_csv(args.output_dir / "endpoint_seed_results.csv", index=False)
    endpoint_summary.to_csv(args.output_dir / "endpoint_summary.csv", index=False)
    baselines.to_csv(args.output_dir / "baseline_summary.csv", index=False)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(endpoint_summary.to_string(index=False))
    print(baselines.to_string(index=False))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
