"""Aggregate frozen early/middle/late observability replications."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analyze_full_ladder import bootstrap, realized
from scripts.train_representative_predictor import load_shards


LAYERS = (2, 6, 10)
LEVELS = {
    "L0 residual": "level0",
    "L2 Q/K": "level2",
    "L7 gradient": "gradient_only",
}
SEEDS = (17, 23, 41)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--multilayer-oracle",
        type=Path,
        required=True,
    )
    parser.add_argument("--layer10-oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def prediction_dir(root: Path, layer: int, slug: str, seed: int) -> Path:
    if layer == 10:
        return root / f"laddertest_{slug}_seed{seed}"
    return root / f"deit_layer{layer}_laddertest_{slug}_seed{seed}"


def oracle_for_layer(
    multilayer: dict[str, np.ndarray],
    layer10: dict[str, np.ndarray],
    layer: int,
) -> tuple[dict[str, np.ndarray], int]:
    if layer == 2:
        return multilayer, 0
    if layer == 6:
        return multilayer, 1
    return layer10, 0


def flatten_layer(
    oracle: dict[str, np.ndarray],
    layer_index: int,
    minimum_drop: float = 1e-5,
) -> dict[str, np.ndarray | int]:
    substitution = oracle["substitution"][:, layer_index].astype(np.float32)
    drop = oracle["single_drop"][:, layer_index].astype(np.float32)
    scores = 1.0 - substitution / np.maximum(drop[:, :, None], 1e-12)
    heads = scores.shape[-1]
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    sample, source = np.where(drop > minimum_drop)
    return {
        "scores": scores[sample, source],
        "sample": sample.astype(np.int64),
        "source": source.astype(np.int64),
        "taylor": oracle["taylor"][sample, layer_index].astype(np.float32),
        "heads": heads,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multilayer = load_shards(args.multilayer_oracle)
    layer10_oracle = load_shards(args.layer10_oracle)

    rows = []
    contrasts = []
    baseline_rows = []
    for layer in LAYERS:
        raw_oracle, layer_index = oracle_for_layer(
            multilayer,
            layer10_oracle,
            layer,
        )
        oracle = flatten_layer(raw_oracle, layer_index)
        heads = int(oracle["heads"])
        valid = np.arange(heads)[None, :] != oracle["source"][:, None]
        taylor_rep = np.where(
            valid,
            oracle["taylor"],
            -np.inf,
        ).argmax(axis=1)
        taylor = oracle["scores"][
            np.arange(len(taylor_rep)),
            taylor_rep,
        ]
        functional_oracle = np.nanmax(oracle["scores"], axis=1)
        baseline_rows.append(
            {
                "layer": layer,
                "taylor_recovery": float(taylor.mean()),
                "oracle_recovery": float(functional_oracle.mean()),
                "valid_source_units": len(taylor),
            }
        )

        for level, slug in LEVELS.items():
            for seed in SEEDS:
                directory = prediction_dir(
                    args.outputs_root,
                    layer,
                    slug,
                    seed,
                )
                summary = pd.read_csv(
                    directory / "policy_summary.csv"
                ).set_index("policy")
                predictor = summary.loc["pair_score_predictor"]
                rows.append(
                    {
                        "layer": layer,
                        "level": level,
                        "slug": slug,
                        "seed": seed,
                        "mean_recovery": predictor["mean_recovery"],
                        "representative_accuracy": predictor[
                            "representative_accuracy"
                        ],
                        "pair_spearman_mean": predictor[
                            "pair_spearman_mean"
                        ],
                    }
                )

        for seed in SEEDS:
            gradient, sample, _ = realized(
                prediction_dir(
                    args.outputs_root,
                    layer,
                    "gradient_only",
                    seed,
                )
            )
            level2, level2_sample, _ = realized(
                prediction_dir(
                    args.outputs_root,
                    layer,
                    "level2",
                    seed,
                )
            )
            if not np.array_equal(sample, level2_sample):
                raise ValueError(
                    f"Prediction sample order mismatch at layer {layer}."
                )
            if not np.array_equal(sample, oracle["sample"]):
                raise ValueError(
                    f"Oracle/prediction sample order mismatch at layer {layer}."
                )
            for comparison, reference in (
                ("gradient-minus-L2", level2),
                ("gradient-minus-Taylor", taylor),
            ):
                contrasts.append(
                    {
                        "layer": layer,
                        "seed": seed,
                        "comparison": comparison,
                        **bootstrap(
                            sample,
                            gradient - reference,
                            args.bootstrap,
                            args.seed + 100 * layer + seed,
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "depth_seed_results.csv", index=False)
    aggregate = (
        frame.groupby(["layer", "level", "slug"], sort=False)
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
    aggregate.to_csv(args.output_dir / "depth_summary.csv", index=False)
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(args.output_dir / "depth_baselines.csv", index=False)
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(
        args.output_dir / "depth_bootstrap_contrasts.csv",
        index=False,
    )

    figure, axes = plt.subplots(1, len(LAYERS), figsize=(13, 4.3))
    colors = ["#7a9cc6", "#4f7cac", "#d1495b"]
    for axis, layer in zip(axes, LAYERS, strict=True):
        subset = aggregate[aggregate["layer"] == layer]
        subset = subset.set_index("slug").loc[list(LEVELS.values())]
        base = baseline.set_index("layer").loc[layer]
        positions = np.arange(len(LEVELS))
        axis.bar(
            positions,
            subset["mean_recovery"],
            yerr=subset["recovery_seed_std"],
            capsize=4,
            color=colors,
        )
        axis.axhline(
            base["taylor_recovery"],
            color="#e07a1f",
            linestyle="--",
            label="Taylor",
        )
        axis.axhline(
            base["oracle_recovery"],
            color="#2a9d6f",
            linestyle=":",
            label="Oracle",
        )
        axis.set_xticks(
            positions,
            ["L0", "L2", "L7"],
        )
        axis.set_title(f"Layer {layer}")
        axis.set_ylabel("Mean recovery score S")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    figure.suptitle("Substitution observability across model depth")
    figure.tight_layout()
    figure.savefig(args.output_dir / "depth_observability.png", dpi=200)
    plt.close(figure)

    print(aggregate.to_string(index=False))
    print(baseline.to_string(index=False))
    print(contrast_frame.to_string(index=False))


if __name__ == "__main__":
    main()
