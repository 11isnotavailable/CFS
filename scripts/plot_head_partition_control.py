"""Plot the trained fixed-width head-granularity control."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def errorbar(axis, heads, bootstrap, metric, label, color):
    part = bootstrap[bootstrap.metric == metric].set_index("heads").loc[heads]
    axis.errorbar(
        heads, part["mean"],
        yerr=np.vstack([part["mean"] - part["ci_low"], part["ci_high"] - part["mean"]]),
        marker="o", capsize=2.5, label=label, color=color,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--community-bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.summary).sort_values("heads")
    thresholds = pd.read_csv(args.thresholds)
    bootstrap = pd.read_csv(args.bootstrap)
    community = pd.read_csv(args.community_bootstrap)
    heads = summary.heads.to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    axes[0, 0].plot(heads, 100 * summary.accuracy, marker="o", color="#4c78a8")
    axes[0, 0].set_ylabel("Held-out evaluation accuracy (%)")
    errorbar(axes[0, 1], heads, bootstrap, "oracle_recovery", "Raw oracle", "#f58518")
    axes[0, 1].plot(heads, summary.matched_best_of_2, marker="s", label="Matched best-of-2", color="#b279a2")
    axes[0, 1].set_ylabel("Conditional recovery")
    axes[0, 1].legend(frameon=False, fontsize=8)
    errorbar(axes[0, 2], heads, bootstrap, "kl_weighted_recovery", "KL-weighted", "#e45756")
    axes[0, 2].set_ylabel("Threshold-free KL-weighted recovery")
    errorbar(axes[1, 0], heads, bootstrap, "valid_source_ratio", "Valid sources", "#72b7b2")
    axes[1, 0].set_ylabel("Valid-source ratio")
    axes[1, 1].plot(heads, summary.spectral_rank, marker="o", label="Effective rank", color="#54a24b")
    axes[1, 1].plot(heads, summary.spectral_rank_per_head, marker="s", label="Rank / H", color="#ff9da6")
    axes[1, 1].set_ylabel("CFS spectral proxy"); axes[1, 1].legend(frameon=False, fontsize=8)
    part = community[(community.threshold == 0.5) & (community.control == "matched_h3") & (community.metric == "mutual_edge_density")]
    part = part.assign(heads=part.model_size.str.replace("H", "", regex=False).astype(int)).set_index("heads").loc[heads]
    axes[1, 2].errorbar(
        heads, part["mean"],
        yerr=np.vstack([part["mean"] - part["ci_low"], part["ci_high"] - part["mean"]]),
        marker="o", capsize=2.5, color="#59a14f",
    )
    axes[1, 2].set_ylabel(r"Matched 3-head mutual density ($\tau=0.5$)")
    for axis in axes.flat:
        axis.set_xscale("log", base=2); axis.set_xticks(heads, [str(value) for value in heads])
        axis.set_xlabel("Heads (fixed $d_{model}=384$)"); axis.grid(alpha=0.22)
        axis.axvspan(13, 35, color="#bdbdbd", alpha=0.10, lw=0)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
