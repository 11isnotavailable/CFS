"""Main fixed-width granularity figure using trained seeds as replicates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def series(axis, frame, metric, label, color, marker="o"):
    axis.errorbar(
        frame.heads, frame[f"{metric}_mean"], yerr=frame[f"{metric}_std"],
        marker=marker, capsize=3, label=label, color=color,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--community", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.aggregate)
    frame = frame[frame.heads <= 12].sort_values("heads")
    community = pd.read_csv(args.community)
    mutual = community[
        (community.heads <= 12) & (community.threshold == 0.5)
        & (community.control == "matched_h3")
        & (community.metric == "mutual_edge_density")
    ].sort_values("heads")
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.2), constrained_layout=True)
    series(axes[0, 0], frame, "accuracy", "Evaluation accuracy", "#4c78a8")
    axes[0, 0].set_ylabel("Evaluation accuracy")
    series(axes[0, 1], frame, "oracle_recovery", "Raw oracle", "#f58518")
    series(axes[0, 1], frame, "matched_best_of_2", "Matched best-of-2", "#b279a2", "s")
    axes[0, 1].set_ylabel("Conditional recovery"); axes[0, 1].legend(frameon=False, fontsize=8)
    series(axes[0, 2], frame, "kl_weighted_recovery", "KL-weighted", "#e45756")
    axes[0, 2].set_ylabel("Threshold-free KL-weighted recovery")
    series(axes[1, 0], frame, "valid_source_ratio", "Valid sources", "#72b7b2")
    axes[1, 0].set_ylabel("Valid-source ratio")
    series(axes[1, 1], frame, "spectral_rank", "Effective rank", "#54a24b")
    series(axes[1, 1], frame, "spectral_rank_per_head", "Rank / H", "#ff9da6", "s")
    axes[1, 1].set_ylabel("CFS spectral proxy"); axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 2].errorbar(
        mutual.heads, mutual["mean"], yerr=mutual["std"],
        marker="o", capsize=3, color="#59a14f",
    )
    axes[1, 2].set_ylabel(r"Matched 3-head mutual density ($\tau=0.5$)")
    heads = frame.heads.to_numpy()
    for axis in axes.flat:
        axis.set_xscale("log", base=2); axis.set_xticks(heads, [str(value) for value in heads])
        axis.set_xlabel("Heads (fixed $d_{model}=384$)"); axis.grid(alpha=0.22)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
