"""Descriptive external-capability/internal-topology comparison.

Four cross-sectional checkpoints do not support correlation, causation, or a
scaling-law fit.  The figure intentionally shows raw quantities rather than
min-max-normalizing them into visually similar curves.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling-summary", type=Path, required=True)
    parser.add_argument("--capacity-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    scaling = pd.read_csv(args.scaling_summary)
    capacity = pd.read_csv(args.capacity_summary)
    capacity = capacity[
        np.isclose(capacity["depth_fraction"], 1.0)
        & np.isclose(capacity["threshold"], args.threshold)
    ]
    frame = scaling.merge(
        capacity[[
            "model_size", "parameters", "minimum_cover_mean", "cover_ratio_mean",
            "cover_to_valid_ratio_mean", "valid_head_ratio",
        ]], on="model_size", validate="one_to_one",
    )
    order = [name for name in ("tiny", "small", "base", "large") if name in set(frame.model_size)]
    frame = frame.set_index("model_size").loc[order].reset_index()
    x = frame["parameters"] / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), constrained_layout=True)
    axes[0].plot(x, 100 * frame["classification_accuracy"], marker="o", color="#4c78a8")
    axes[0].set_ylabel("Imagenette accuracy (%)")
    axes[1].plot(x, frame["minimum_cover_mean"], marker="o", color="#f58518")
    axes[1].set_ylabel(fr"Minimum representative cover ($\tau={args.threshold:g}$)")
    axes[2].plot(x, frame["cover_ratio_mean"], marker="o", label="cover / total H", color="#54a24b")
    axes[2].plot(x, frame["cover_to_valid_ratio_mean"], marker="s", label="cover / valid H", color="#e45756")
    axes[2].set_ylabel("Graph-derived cover ratio")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xscale("log"); axis.set_xlabel("Parameters (millions, log scale)"); axis.grid(alpha=0.22)
        axis.set_xticks(x, [name.title() for name in order])
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
