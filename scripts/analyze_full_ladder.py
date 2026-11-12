"""Aggregate the frozen L0-L7 ladder and image-level bootstrap contrasts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.train_representative_predictor import flatten_examples, load_shards


LEVELS = {
    "L0 residual": "level0",
    "L2 Q/K moments": "level2",
    "L3 attention logits": "level3",
    "L4 V content": "value_only",
    "L5 A@V output": "aggregated_only",
    "L6 post-WO": "contribution_only",
    "L7 gradient": "gradient_only",
}
SEEDS = (17, 23, 41)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def output_dir(root: Path, slug: str, seed: int) -> Path:
    return root / f"laddertest_{slug}_seed{seed}"


def bootstrap(
    sample: np.ndarray,
    difference: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    unique = np.unique(sample)
    per_image = np.asarray(
        [difference[sample == item].mean() for item in unique]
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        per_image,
        size=(repeats, len(per_image)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean_difference": float(per_image.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0)),
    }


def realized(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(directory / "predictions.npz")
    prediction = archive["prediction"]
    scores = archive["scores"]
    values = scores[np.arange(len(prediction)), prediction]
    return values, archive["sample"], prediction


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, slug in LEVELS.items():
        for seed in SEEDS:
            summary = pd.read_csv(
                output_dir(args.outputs_root, slug, seed)
                / "policy_summary.csv"
            ).set_index("policy")
            predictor = summary.loc["pair_score_predictor"]
            rows.append(
                {
                    "level": label,
                    "slug": slug,
                    "seed": seed,
                    "mean_recovery": predictor["mean_recovery"],
                    "median_recovery": predictor["median_recovery"],
                    "representative_accuracy": predictor[
                        "representative_accuracy"
                    ],
                    "pair_spearman_mean": predictor["pair_spearman_mean"],
                    "pair_spearman_median": predictor[
                        "pair_spearman_median"
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "ladder_seed_results.csv", index=False)
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
    aggregate.to_csv(args.output_dir / "ladder_summary.csv", index=False)

    oracle = flatten_examples(load_shards(args.oracle_dir), 1e-5)
    heads = int(oracle["heads_count"])
    valid = np.arange(heads)[None, :] != oracle["source"][:, None]
    taylor_rep = np.where(valid, oracle["taylor"], -np.inf).argmax(axis=1)
    taylor_values = oracle["scores"][
        np.arange(len(taylor_rep)),
        taylor_rep,
    ]

    contrasts = []
    for seed in SEEDS:
        gradient, sample, _ = realized(
            output_dir(args.outputs_root, "gradient_only", seed)
        )
        level2, level2_sample, _ = realized(
            output_dir(args.outputs_root, "level2", seed)
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
    pd.DataFrame(contrasts).to_csv(
        args.output_dir / "bootstrap_contrasts.csv",
        index=False,
    )

    plt.figure(figsize=(10, 5.5))
    positions = np.arange(len(aggregate))
    plt.errorbar(
        positions,
        aggregate["mean_recovery"],
        yerr=aggregate["recovery_seed_std"],
        fmt="o-",
        capsize=4,
        color="#28536b",
    )
    plt.axhline(
        float(taylor_values.mean()),
        color="#d95f02",
        linestyle="--",
        label="Taylor representative",
    )
    plt.axhline(
        float(np.nanmax(oracle["scores"], axis=1).mean()),
        color="#1b9e77",
        linestyle=":",
        label="Functional oracle",
    )
    plt.xticks(positions, aggregate["level"], rotation=25, ha="right")
    plt.ylabel("Mean recovery score S")
    plt.title("Where does functional substitution become observable?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "observability_ladder.png", dpi=180)
    plt.close()

    print(aggregate.to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__ == "__main__":
    main()
