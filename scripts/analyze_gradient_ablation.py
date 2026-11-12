"""Aggregate frozen gradient-information ablations at layers 2 and 6."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_full_ladder import bootstrap, realized
from scripts.train_representative_predictor import load_shards


LAYERS = (2, 6)
SEEDS = (17, 23, 41)
LEARNED = {
    "|CE gradient| pair scorer": "gradient_abs_ce_only",
    "signed CE pair scorer": "gradient_ce_only",
    "predicted-logit pair scorer": "gradient_logit_only",
    "full gradient pair scorer": "gradient_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def run_dir(root: Path, layer: int, slug: str, seed: int) -> Path:
    if slug == "gradient_only":
        return root / f"deit_layer{layer}_laddertest_{slug}_seed{seed}"
    return root / f"deit_layer{layer}_ablation_{slug}_seed{seed}"


def oracle_layer(
    oracle: dict[str, np.ndarray],
    layer_index: int,
    minimum_drop: float = 1e-5,
) -> dict[str, np.ndarray]:
    substitution = oracle["substitution"][:, layer_index].astype(np.float32)
    drop = oracle["single_drop"][:, layer_index].astype(np.float32)
    scores = 1.0 - substitution / np.maximum(drop[:, :, None], 1e-12)
    heads = scores.shape[-1]
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    sample, source = np.where(drop > minimum_drop)
    valid = np.arange(heads)[None, :] != source[:, None]
    taylor = oracle["taylor"][sample, layer_index].astype(np.float32)
    representative = np.where(valid, taylor, -np.inf).argmax(axis=1)
    return {
        "sample": sample,
        "taylor_values": scores[sample, source, representative],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle = load_shards(args.oracle_dir)

    rows = []
    contrasts = []
    for layer_index, layer in enumerate(LAYERS):
        layer_oracle = oracle_layer(oracle, layer_index)
        predictions: dict[tuple[str, int], np.ndarray] = {}
        for label, slug in LEARNED.items():
            for seed in SEEDS:
                directory = run_dir(
                    args.outputs_root,
                    layer,
                    slug,
                    seed,
                )
                summary = pd.read_csv(
                    directory / "policy_summary.csv"
                ).set_index("policy")
                row = summary.loc["pair_score_predictor"]
                rows.append(
                    {
                        "layer": layer,
                        "feature": label,
                        "slug": slug,
                        "seed": seed,
                        "mean_recovery": row["mean_recovery"],
                        "representative_accuracy": row[
                            "representative_accuracy"
                        ],
                        "pair_spearman_mean": row["pair_spearman_mean"],
                    }
                )
                values, sample, _ = realized(directory)
                if not np.array_equal(sample, layer_oracle["sample"]):
                    raise ValueError(
                        f"Sample order mismatch at layer {layer}, {slug}."
                    )
                predictions[(slug, seed)] = values

        for seed in SEEDS:
            comparisons = (
                (
                    "abs-CE-pair-minus-Taylor",
                    predictions[("gradient_abs_ce_only", seed)],
                    layer_oracle["taylor_values"],
                ),
                (
                    "signed-CE-minus-abs-CE",
                    predictions[("gradient_ce_only", seed)],
                    predictions[("gradient_abs_ce_only", seed)],
                ),
                (
                    "full-minus-abs-CE",
                    predictions[("gradient_only", seed)],
                    predictions[("gradient_abs_ce_only", seed)],
                ),
                (
                    "full-minus-Taylor",
                    predictions[("gradient_only", seed)],
                    layer_oracle["taylor_values"],
                ),
            )
            for comparison, left, right in comparisons:
                contrasts.append(
                    {
                        "layer": layer,
                        "seed": seed,
                        "comparison": comparison,
                        **bootstrap(
                            layer_oracle["sample"],
                            left - right,
                            args.bootstrap,
                            args.seed + 100 * layer + seed,
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "gradient_ablation_seeds.csv", index=False)
    aggregate = (
        frame.groupby(["layer", "feature", "slug"], sort=False)
        .agg(
            mean_recovery=("mean_recovery", "mean"),
            recovery_seed_std=("mean_recovery", "std"),
            mean_accuracy=("representative_accuracy", "mean"),
            mean_pair_spearman=("pair_spearman_mean", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(
        args.output_dir / "gradient_ablation_summary.csv",
        index=False,
    )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(
        args.output_dir / "gradient_ablation_bootstrap.csv",
        index=False,
    )
    print(aggregate.to_string(index=False))
    print(contrast_frame.to_string(index=False))


if __name__ == "__main__":
    main()
