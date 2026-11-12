"""Evaluate attention-map, magnitude, entropy, and Taylor baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = np.load(args.experiment_dir / "matrix_records.npz")
    features = np.load(args.features)
    metadata = json.loads(
        (args.analysis_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    test_samples = set(metadata["test_samples"])

    curves = archive["substitution_curves"].astype(np.float64)
    substitution = archive["substitution"].astype(np.float64)
    drops = archive["single_drop"].astype(np.float64)
    current_l2 = archive["current_l2"].astype(np.float64)
    samples = archive["sample"].astype(int)
    layers = archive["layer"].astype(int)
    heads = curves.shape[1]

    # Baseline features are stored sample-major and then layer-major, matching
    # Experiment 0's unit ordering.
    attention_distance = features["attention_map_distance"].reshape(
        -1, heads, heads
    )
    entropy = features["attention_entropy"].reshape(-1, heads)
    norm = features["contribution_norm"].reshape(-1, heads)
    taylor = features["taylor"].reshape(-1, heads)
    if len(attention_distance) != len(curves):
        raise ValueError("Baseline features do not align with oracle units.")

    rows = []
    rank_rows = []
    source_rows = []
    for unit, (sample, layer) in enumerate(zip(samples, layers)):
        if int(sample) not in test_samples:
            continue
        for source in range(heads):
            if drops[unit, source] <= 1e-5:
                continue
            representatives = np.asarray(
                [head for head in range(heads) if head != source]
            )
            oracle_errors = np.nanmin(
                curves[unit, source, representatives, :],
                axis=-1,
            )
            oracle_rep = int(representatives[np.argmin(oracle_errors)])

            selections = {
                "current_l2": int(
                    representatives[
                        np.argmin(current_l2[unit, source, representatives])
                    ]
                ),
                "attention_map_similarity": int(
                    representatives[
                        np.argmin(
                            attention_distance[unit, source, representatives]
                        )
                    ]
                ),
                "largest_contribution_norm": int(
                    representatives[np.argmax(norm[unit, representatives])]
                ),
                "lowest_attention_entropy": int(
                    representatives[np.argmin(entropy[unit, representatives])]
                ),
                "highest_taylor_importance": int(
                    representatives[np.argmax(taylor[unit, representatives])]
                ),
            }
            for policy, representative in selections.items():
                error = float(
                    np.nanmin(curves[unit, source, representative, :])
                )
                rows.append(
                    {
                        "sample": int(sample),
                        "layer": int(layer),
                        "source": source,
                        "policy": policy,
                        "representative": representative,
                        "oracle_representative": oracle_rep,
                        "representative_match": int(representative == oracle_rep),
                        "recovery_score": 1.0 - error / drops[unit, source],
                        "error_kl": error,
                    }
                )

            functional_score = 1.0 - oracle_errors / drops[unit, source]
            feature_vectors = {
                "current_l2": -current_l2[unit, source, representatives],
                "attention_map_similarity": -attention_distance[
                    unit, source, representatives
                ],
                "contribution_norm": norm[unit, representatives],
                "negative_attention_entropy": -entropy[unit, representatives],
                "taylor_importance": taylor[unit, representatives],
            }
            for feature_name, values in feature_vectors.items():
                rank_rows.append(
                    {
                        "sample": int(sample),
                        "layer": int(layer),
                        "source": source,
                        "feature": feature_name,
                        "spearman_with_S": float(
                            spearmanr(values, functional_score).statistic
                        ),
                    }
                )

        # Source-head pruning relation: which scalar ranks direct-drop damage?
        source_features = {
            "contribution_norm": norm[unit],
            "negative_attention_entropy": -entropy[unit],
            "taylor_importance": taylor[unit],
        }
        for feature_name, values in source_features.items():
            source_rows.append(
                {
                    "sample": int(sample),
                    "layer": int(layer),
                    "feature": feature_name,
                    "spearman_with_drop_kl": float(
                        spearmanr(values, drops[unit]).statistic
                    ),
                    "selected_drop_kl": float(
                        drops[unit, int(np.argmin(values))]
                    ),
                    "oracle_drop_kl": float(np.min(drops[unit])),
                    "random_mean_drop_kl": float(np.mean(drops[unit])),
                }
            )

    policy_frame = pd.DataFrame(rows)
    rank_frame = pd.DataFrame(rank_rows)
    source_frame = pd.DataFrame(source_rows)
    policy_frame.to_csv(args.analysis_dir / "strong_baseline_policies.csv", index=False)
    rank_frame.to_csv(args.analysis_dir / "strong_baseline_ranks.csv", index=False)
    source_frame.to_csv(
        args.analysis_dir / "source_importance_baselines.csv", index=False
    )

    policy_summary = (
        policy_frame.groupby("policy")
        .agg(
            mean_recovery=("recovery_score", "mean"),
            median_recovery=("recovery_score", "median"),
            representative_accuracy=("representative_match", "mean"),
            mean_kl=("error_kl", "mean"),
        )
        .sort_values("mean_recovery", ascending=False)
    )
    rank_summary = (
        rank_frame.groupby("feature")["spearman_with_S"]
        .agg(["mean", "median"])
        .sort_values("mean", ascending=False)
    )
    source_summary = (
        source_frame.groupby("feature")
        .agg(
            mean_rank_correlation=("spearman_with_drop_kl", "mean"),
            median_rank_correlation=("spearman_with_drop_kl", "median"),
            mean_selected_drop_kl=("selected_drop_kl", "mean"),
            mean_oracle_drop_kl=("oracle_drop_kl", "mean"),
            mean_random_drop_kl=("random_mean_drop_kl", "mean"),
        )
        .sort_values("mean_selected_drop_kl")
    )
    policy_summary.to_csv(args.analysis_dir / "strong_baseline_summary.csv")
    rank_summary.to_csv(args.analysis_dir / "strong_baseline_rank_summary.csv")
    source_summary.to_csv(args.analysis_dir / "source_importance_summary.csv")
    print("REPRESENTATIVE POLICIES")
    print(policy_summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nFEATURE RANKING OF S")
    print(rank_summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nSOURCE IMPORTANCE")
    print(source_summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
