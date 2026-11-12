"""Repeat CFS community summaries on images every compared model classifies correctly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_cfs_communities import paired_contrasts, summarize_samples
from scripts.analyze_scaling_cfs import shard_field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--community-samples", type=Path, required=True)
    parser.add_argument(
        "--entry", action="append", nargs=2, metavar=("NAME", "ORACLE_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    correctness = []
    selected_models = []
    for name, directory in args.entry:
        selected_models.append(name)
        path = Path(directory)
        correctness.append(
            shard_field(path, "prediction") == shard_field(path, "imagenet_target")
        )
    shared = np.logical_and.reduce(correctness)
    frame = pd.read_csv(args.community_samples)
    frame = frame[frame["model_size"].isin(selected_models)]
    filtered = frame[frame["sample"].isin(np.flatnonzero(shared))].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output_dir / "shared_correct_community_samples.csv", index=False)
    summary = summarize_samples(filtered, args.bootstrap_repetitions, args.seed)
    contrasts = paired_contrasts(filtered, args.bootstrap_repetitions, args.seed + 1)
    summary.to_csv(args.output_dir / "shared_correct_community_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "shared_correct_paired_contrasts.csv", index=False)
    pd.DataFrame({
        "sample": np.arange(len(shared)), "all_models_correct": shared.astype(int)
    }).to_csv(args.output_dir / "shared_correct_mask.csv", index=False)
    print(f"shared_correct={shared.sum()}/{len(shared)}")
    print(summary[
        (summary.threshold == 0.5) & (summary.control == "full")
        & summary.metric.isin(["directed_edge_density", "mutual_edge_density", "mutual_component_ratio", "mutual_largest_ratio"])
    ].to_string(index=False))


if __name__ == "__main__":
    main()
