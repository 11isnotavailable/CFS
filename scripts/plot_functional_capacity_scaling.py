"""Create paper-ready scaling-topology and capacity-proxy figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ("tiny", "small", "base", "large")
LABELS = {"tiny": "Tiny", "small": "Small", "base": "Base", "large": "Large"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling-summary", type=Path, required=True)
    parser.add_argument("--capacity-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    present = [name for name in ORDER if name in set(frame["model_size"])]
    return frame.set_index("model_size").loc[present].reset_index()


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def topology_figure(scaling: pd.DataFrame, output_dir: Path) -> None:
    frame = ordered(scaling)
    labels = [LABELS[name] for name in frame["model_size"]]
    panels = (
        ("oracle_recovery", "Raw oracle recovery", "CFS recovery"),
        ("matched_best_of_2_recovery", "Matched best-of-2", "CFS recovery"),
        ("best_representative_entropy", "Representative entropy", "Normalized entropy"),
        ("best_representative_concentration", "Representative concentration", "Top representative share"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for axis, (column, title, ylabel) in zip(axes.flat, panels):
        values = frame[column].to_numpy()
        axis.plot(labels, values, marker="o", linewidth=2, color="#2f5aa8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.annotate(f"{value:.3f}", (index, value), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=8)
    save(fig, output_dir, "cfs_scaling_topology")


def capacity_figure(
    capacity: pd.DataFrame,
    threshold: float,
    output_dir: Path,
) -> None:
    selected = capacity[np.isclose(capacity["threshold"], threshold)].copy()
    last = ordered(selected[np.isclose(selected["depth_fraction"], 1.0)])
    labels = [LABELS[name] for name in last["model_size"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=True)
    axes = axes.flat

    axes[0].plot(labels, last["minimum_cover_mean"], marker="o", label="Min. cover")
    axes[0].plot(labels, last["spectral_effective_rank_mean"], marker="s",
                 label="Spectral rank")
    axes[0].set_title("Absolute capacity proxies")
    axes[0].set_ylabel("Effective heads")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(labels, last["cover_ratio_mean"], marker="o", label="Cover / heads")
    axes[1].plot(labels, last["spectral_rank_ratio_mean"], marker="s",
                 label="Rank / heads")
    axes[1].set_title("Capacity per head")
    axes[1].set_ylabel("Normalized proxy")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        labels, last["cover_to_valid_ratio_mean"], marker="o",
        label="Cover / valid heads",
    )
    axes[2].plot(
        labels, last["spectral_to_valid_ratio_mean"], marker="s",
        label="Rank / valid heads",
    )
    axes[2].set_title("Conditional on valid sources")
    axes[2].set_ylabel("Normalized proxy")
    axes[2].legend(frameon=False, fontsize=8)

    for name in ORDER:
        model = selected[selected["model_size"] == name].sort_values("depth_fraction")
        if len(model) < 2:
            continue
        axes[3].plot(
            model["depth_fraction"], model["cover_ratio_mean"], marker="o",
            label=LABELS[name],
        )
    axes[3].set_title(f"Depth profile ($\\tau={threshold:g}$)")
    axes[3].set_xlabel("Relative layer depth")
    axes[3].set_ylabel("Minimum cover / heads")
    axes[3].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    save(fig, output_dir, "cfs_functional_capacity")


def main() -> None:
    args = parse_args()
    scaling = pd.read_csv(args.scaling_summary)
    capacity = pd.read_csv(args.capacity_summary)
    topology_figure(scaling, args.output_dir)
    capacity_figure(capacity, args.threshold, args.output_dir)


if __name__ == "__main__":
    main()
