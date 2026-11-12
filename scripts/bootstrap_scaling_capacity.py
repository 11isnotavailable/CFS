"""Matched-image adjacent-size contrasts for final-layer capacity proxies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ORDER = ("tiny", "small", "base", "large")
METRICS = (
    "oracle_recovery",
    "valid_head_ratio",
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
    frame = frame[np.isclose(frame["depth_fraction"], 1.0)]
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    present = [name for name in ORDER if name in set(frame["model_size"])]
    for threshold, group in frame.groupby("threshold"):
        for smaller, larger in zip(present[:-1], present[1:]):
            left = group[group["model_size"] == smaller].set_index("sample")
            right = group[group["model_size"] == larger].set_index("sample")
            common = left.index.intersection(right.index)
            for metric in METRICS:
                difference = (
                    right.loc[common, metric].to_numpy(dtype=np.float64)
                    - left.loc[common, metric].to_numpy(dtype=np.float64)
                )
                difference = difference[np.isfinite(difference)]
                draws = rng.integers(
                    0, len(difference), size=(args.bootstrap, len(difference))
                )
                bootstrap = difference[draws].mean(axis=1)
                rows.append(
                    {
                        "threshold": threshold,
                        "smaller": smaller,
                        "larger": larger,
                        "metric": metric,
                        "samples": len(difference),
                        "larger_minus_smaller": difference.mean(),
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
