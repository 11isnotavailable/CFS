"""Aggregate a frozen three-endpoint observability replication.

The replication intentionally evaluates only the pre-registered endpoints:
residual (L0), the best forward probe selected on the first model (L2), and
the downstream-sensitivity probe (L7).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_full_ladder import bootstrap, realized
from scripts.train_representative_predictor import flatten_examples, load_shards


LEVELS = {
    "L0 residual": "level0",
    "L2 Q/K moments": "level2",
    "L7 gradient": "gradient_only",
}
SEEDS = (17, 23, 41)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-prefix", type=str, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def run_dir(root: Path, prefix: str, slug: str, seed: int) -> Path:
    return root / f"{prefix}_{slug}_seed{seed}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oracle = flatten_examples(load_shards(args.oracle_dir), 1e-5)
    heads = int(oracle["heads_count"])
    valid = np.arange(heads)[None, :] != oracle["source"][:, None]
    taylor_rep = np.where(valid, oracle["taylor"], -np.inf).argmax(axis=1)
    taylor_values = oracle["scores"][
        np.arange(len(taylor_rep)),
        taylor_rep,
    ]
    oracle_values = np.nanmax(oracle["scores"], axis=1)

    rows: list[dict[str, float | int | str]] = []
    for label, slug in LEVELS.items():
        for seed in SEEDS:
            directory = run_dir(
                args.outputs_root,
                args.run_prefix,
                slug,
                seed,
            )
            summary = pd.read_csv(
                directory / "policy_summary.csv"
            ).set_index("policy")
            predictor = summary.loc["pair_score_predictor"]
            rows.append(
                {
                    "level": label,
                    "slug": slug,
                    "seed": seed,
                    "mean_recovery": predictor["mean_recovery"],
                    "representative_accuracy": predictor[
                        "representative_accuracy"
                    ],
                    "pair_spearman_mean": predictor["pair_spearman_mean"],
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "seed_results.csv", index=False)
    aggregate = (
        frame.groupby(["level", "slug"], sort=False)
        .agg(
            mean_recovery=("mean_recovery", "mean"),
            recovery_seed_std=("mean_recovery", "std"),
            mean_accuracy=("representative_accuracy", "mean"),
            accuracy_seed_std=("representative_accuracy", "std"),
            mean_pair_spearman=("pair_spearman_mean", "mean"),
            spearman_seed_std=("pair_spearman_mean", "std"),
        )
        .reset_index()
    )
    aggregate["taylor_recovery"] = float(taylor_values.mean())
    aggregate["oracle_recovery"] = float(oracle_values.mean())
    aggregate.to_csv(args.output_dir / "summary.csv", index=False)

    contrasts = []
    for seed in SEEDS:
        gradient, sample, _ = realized(
            run_dir(
                args.outputs_root,
                args.run_prefix,
                "gradient_only",
                seed,
            )
        )
        level2, level2_sample, _ = realized(
            run_dir(
                args.outputs_root,
                args.run_prefix,
                "level2",
                seed,
            )
        )
        if not np.array_equal(sample, level2_sample):
            raise ValueError("Prediction sample order mismatch.")
        for comparison, reference in (
            ("gradient-minus-L2", level2),
            ("gradient-minus-Taylor", taylor_values),
        ):
            contrasts.append(
                {
                    "seed": seed,
                    "comparison": comparison,
                    **bootstrap(
                        sample,
                        gradient - reference,
                        args.bootstrap,
                        args.seed + seed,
                    ),
                }
            )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(
        args.output_dir / "bootstrap_contrasts.csv",
        index=False,
    )

    print(aggregate.to_string(index=False))
    print(contrast_frame.to_string(index=False))


if __name__ == "__main__":
    main()
