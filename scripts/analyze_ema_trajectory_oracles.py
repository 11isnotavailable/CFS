"""Analyze matched online/EMA functional trajectories at fixed epochs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_cfs_developmental_oracles import analyze_layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = (3, 7, 11)
    rows = []
    for epoch in (50, 75, 100):
        for track in ("online", "ema"):
            label = f"{track}_e{epoch:03d}"
            directory = args.root / f"cfs_{label}_final512"
            for position, layer in enumerate(layers):
                row = analyze_layer(
                    label, directory, position, layer, args.minimum_drop_kl
                )
                row["track"] = track
                row["epoch"] = epoch
                rows.append(row)

    by_layer = pd.DataFrame(rows)
    metrics = {
        "valid_source_fraction": "mean",
        "mean_drop_kl": "mean",
        "mean_best_substitution_kl": "mean",
        "raw_oracle": "mean",
        "matched_best_of_2": "mean",
        "source_fraction_s_gt_025": "mean",
        "representative_concentration": "mean",
        "representative_entropy": "mean",
        "slice_accuracy": "first",
    }
    macro = by_layer.groupby(["epoch", "track"], as_index=False).agg(metrics)
    by_layer.to_csv(args.output_dir / "ema_trajectory_by_layer.csv", index=False)
    macro.to_csv(args.output_dir / "ema_trajectory_macro.csv", index=False)
    print("BY_LAYER")
    print(by_layer.to_string(index=False))
    print("MACRO")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
