"""Aggregate the untouched layer-10 gradient-ablation test."""

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


SEEDS = (17, 23, 41)
LEVELS = {
    "|CE gradient| pair": "gradient_abs_ce_only",
    "signed CE pair": "gradient_ce_only",
    "predicted-logit pair": "gradient_logit_only",
    "full gradient pair": "gradient_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def directory(root: Path, slug: str, seed: int) -> Path:
    return root / f"deit_layer10_gradtest_{slug}_seed{seed}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle = load_shards(args.oracle_dir)
    substitution = oracle["substitution"][:, 0].astype(np.float32)
    drop = oracle["single_drop"][:, 0].astype(np.float32)
    scores = 1.0 - substitution / np.maximum(drop[:, :, None], 1e-12)
    heads = scores.shape[-1]
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    sample, source = np.where(drop > 1e-5)
    valid = np.arange(heads)[None, :] != source[:, None]
    taylor = oracle["taylor"][sample, 0].astype(np.float32)
    taylor_rep = np.where(valid, taylor, -np.inf).argmax(axis=1)
    taylor_values = scores[sample, source, taylor_rep]

    rows = []
    values: dict[tuple[str, int], np.ndarray] = {}
    for label, slug in LEVELS.items():
        for seed in SEEDS:
            run = directory(args.outputs_root, slug, seed)
            summary = pd.read_csv(run / "policy_summary.csv").set_index(
                "policy"
            )
            predictor = summary.loc["pair_score_predictor"]
            rows.append(
                {
                    "feature": label,
                    "slug": slug,
                    "seed": seed,
                    "mean_recovery": predictor["mean_recovery"],
                    "representative_accuracy": predictor[
                        "representative_accuracy"
                    ],
                    "pair_spearman_mean": predictor["pair_spearman_mean"],
                }
            )
            realized_values, prediction_sample, _ = realized(run)
            if not np.array_equal(sample, prediction_sample):
                raise ValueError(f"Sample order mismatch for {slug}.")
            values[(slug, seed)] = realized_values

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "late_ablation_seeds.csv", index=False)
    aggregate = (
        frame.groupby(["feature", "slug"], sort=False)
        .agg(
            mean_recovery=("mean_recovery", "mean"),
            recovery_seed_std=("mean_recovery", "std"),
            mean_accuracy=("representative_accuracy", "mean"),
            mean_pair_spearman=("pair_spearman_mean", "mean"),
        )
        .reset_index()
    )
    aggregate["taylor_recovery"] = float(taylor_values.mean())
    aggregate["oracle_recovery"] = float(np.nanmax(scores, axis=-1).mean())
    aggregate.to_csv(args.output_dir / "late_ablation_summary.csv", index=False)

    contrasts = []
    for seed in SEEDS:
        full = values[("gradient_only", seed)]
        for comparison, reference in (
            (
                "full-minus-abs-CE",
                values[("gradient_abs_ce_only", seed)],
            ),
            (
                "full-minus-signed-CE",
                values[("gradient_ce_only", seed)],
            ),
            ("full-minus-Taylor", taylor_values),
        ):
            contrasts.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    **bootstrap(
                        sample,
                        full - reference,
                        args.bootstrap,
                        args.seed + seed,
                    ),
                }
            )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(
        args.output_dir / "late_ablation_bootstrap.csv",
        index=False,
    )
    print(aggregate.to_string(index=False))
    print(contrast_frame.to_string(index=False))


if __name__ == "__main__":
    main()
