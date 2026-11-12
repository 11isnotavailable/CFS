"""Aggregate community metrics across independently trained head partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry", action="append", nargs=2, metavar=("SEED", "SUMMARY_CSV"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for seed, path in args.entry:
        frame = pd.read_csv(path)
        frame["seed"] = int(seed)
        frame["heads"] = frame.model_size.str.replace("H", "", regex=False).astype(int)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    # Seed-level means are already computed over the same 128 frozen images.
    aggregate = data.groupby(
        ["heads", "threshold", "control", "metric"]
    )["mean"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "community_per_seed.csv", index=False)
    aggregate.to_csv(args.output_dir / "community_seed_aggregate.csv", index=False)
    print(aggregate[
        (aggregate.threshold == 0.5) & (aggregate.control == "matched_h3")
        & aggregate.metric.isin(["directed_edge_density", "mutual_edge_density"])
    ].to_string(index=False))


if __name__ == "__main__":
    main()
