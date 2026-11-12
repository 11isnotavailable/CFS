"""Plot frozen observability endpoints across model/training replications."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deit-summary",
        type=Path,
        default=Path("outputs/full_ladder_analysis/ladder_summary.csv"),
    )
    parser.add_argument(
        "--vit-summary",
        type=Path,
        default=Path("outputs/vitaug_replication_analysis/summary.csv"),
    )
    parser.add_argument(
        "--dino-summary",
        type=Path,
        default=Path("outputs/dino_clean_replication_analysis/summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/cross_model_replication/"
            "observability_replication.png"
        ),
    )
    return parser.parse_args()


def endpoint(frame: pd.DataFrame, slug: str) -> tuple[float, float]:
    row = frame.loc[frame["slug"] == slug]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {slug}, found {len(row)}.")
    return (
        float(row.iloc[0]["mean_recovery"]),
        float(row.iloc[0]["recovery_seed_std"]),
    )


def main() -> None:
    args = parse_args()
    deit = pd.read_csv(args.deit_summary)
    vit = pd.read_csv(args.vit_summary)
    dino = pd.read_csv(args.dino_summary)
    frames = [deit, vit, dino]
    models = ["DeiT-S", "ViT-S AugReg", "DINO ViT-S + linear probe"]
    slugs = ["level0", "level2", "gradient_only"]
    labels = ["L0 residual", "L2 Q/K", "L7 gradient"]

    means = np.asarray(
        [[endpoint(frame, slug)[0] for slug in slugs] for frame in frames]
    )
    errors = np.asarray(
        [[endpoint(frame, slug)[1] for slug in slugs] for frame in frames]
    )
    taylors = np.asarray(
        [
            0.231833,
            float(vit.iloc[0]["taylor_recovery"]),
            float(dino.iloc[0]["taylor_recovery"]),
        ]
    )
    oracles = np.asarray(
        [
            0.323060,
            float(vit.iloc[0]["oracle_recovery"]),
            float(dino.iloc[0]["oracle_recovery"]),
        ]
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    colors = ["#7a9cc6", "#4f7cac", "#d1495b"]
    positions = np.arange(len(slugs))
    for index, axis in enumerate(axes):
        axis.bar(
            positions,
            means[index],
            yerr=errors[index],
            capsize=4,
            color=colors,
            width=0.68,
        )
        axis.axhline(
            taylors[index],
            color="#e07a1f",
            linestyle="--",
            linewidth=1.5,
            label="Taylor",
        )
        axis.axhline(
            oracles[index],
            color="#2a9d6f",
            linestyle=":",
            linewidth=1.8,
            label="Functional oracle",
        )
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.set_ylabel("Mean recovery score S")
        axis.set_title(models[index])
        axis.legend(loc="upper left")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Downstream sensitivity improves substitution observability "
        "across models"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"figure={args.output.resolve()}")


if __name__ == "__main__":
    main()
