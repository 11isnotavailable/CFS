"""Matched-image bootstrap contrasts for layer-wise capacity proxies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "oracle_recovery",
    "cover_ratio",
    "cover_to_valid_ratio",
    "spectral_rank_ratio",
    "spectral_to_valid_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.samples)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for (model, threshold), group in frame.groupby(["model_size", "threshold"]):
        layers = sorted(group["layer"].unique())
        if len(layers) < 2:
            continue
        early, late = layers[0], layers[-1]
        early_frame = group[group["layer"] == early].set_index("sample")
        late_frame = group[group["layer"] == late].set_index("sample")
        common = early_frame.index.intersection(late_frame.index)
        for metric in METRICS:
            difference = (
                late_frame.loc[common, metric].to_numpy(dtype=np.float64)
                - early_frame.loc[common, metric].to_numpy(dtype=np.float64)
            )
            difference = difference[np.isfinite(difference)]
            draws = rng.integers(0, len(difference), size=(args.bootstrap, len(difference)))
            bootstrap = difference[draws].mean(axis=1)
            rows.append(
                {
                    "model_size": model,
                    "threshold": threshold,
                    "early_layer": early,
                    "late_layer": late,
                    "metric": metric,
                    "samples": len(difference),
                    "late_minus_early": difference.mean(),
                    "ci95_low": np.quantile(bootstrap, 0.025),
                    "ci95_high": np.quantile(bootstrap, 0.975),
                }
            )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
