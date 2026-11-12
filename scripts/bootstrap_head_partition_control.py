"""Bootstrap uncertainty for the fixed-width head-granularity control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "valid_source_ratio", "oracle_recovery", "top_source_oracle_recovery",
    "kl_weighted_recovery", "positive_kl_weighted_recovery", "mean_drop_kl",
]


def bootstrap_mean(values: np.ndarray, repetitions: int, rng) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    return tuple(np.quantile(values[draws].mean(axis=1), [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--main-heads", default="3,4,6,8,12")
    args = parser.parse_args()
    frame = pd.read_csv(args.samples)
    heads = sorted(frame.heads.unique())
    rng = np.random.default_rng(args.seed)
    summary_rows = []
    for head in heads:
        group = frame[frame.heads == head]
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            low, high = bootstrap_mean(values, args.bootstrap_repetitions, rng)
            summary_rows.append({
                "heads": head, "metric": metric, "mean": values.mean(),
                "ci_low": low, "ci_high": high, "samples": len(values),
            })
    pairs = list(zip(heads[:-1], heads[1:]))
    main_heads = [int(value) for value in args.main_heads.split(",")]
    pairs.extend([(main_heads[0], main_heads[-1]), (heads[0], heads[-1])])
    contrast_rows = []
    for left_head, right_head in dict.fromkeys(pairs):
        left = frame[frame.heads == left_head].set_index("sample")
        right = frame[frame.heads == right_head].set_index("sample")
        common = left.index.intersection(right.index)
        for metric in METRICS:
            difference = (
                right.loc[common, metric].to_numpy(float)
                - left.loc[common, metric].to_numpy(float)
            )
            difference = difference[np.isfinite(difference)]
            low, high = bootstrap_mean(difference, args.bootstrap_repetitions, rng)
            contrast_rows.append({
                "contrast": f"H{right_head}-H{left_head}", "metric": metric,
                "mean_difference": difference.mean(), "ci_low": low,
                "ci_high": high, "paired_samples": len(difference),
            })
    # Shared-correctness sensitivity within the high-performance main range.
    main = frame[frame.heads.isin(main_heads)]
    correct_table = main.pivot(index="sample", columns="heads", values="accuracy")
    shared_correct = correct_table[main_heads].eq(1).all(axis=1)
    correct_rows = []
    for head in main_heads:
        group = frame[(frame.heads == head) & frame["sample"].isin(shared_correct[shared_correct].index)]
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            low, high = bootstrap_mean(values, args.bootstrap_repetitions, rng)
            correct_rows.append({
                "heads": head, "metric": metric, "mean": values.mean(),
                "ci_low": low, "ci_high": high, "samples": len(values),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "bootstrap_summary.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    pd.DataFrame(correct_rows).to_csv(args.output_dir / "shared_correct_summary.csv", index=False)
    print(f"shared_correct_main={shared_correct.sum()}/{len(shared_correct)}")
    print(pd.DataFrame(contrast_rows).query(
        "contrast == 'H12-H3' and metric in ['oracle_recovery', 'kl_weighted_recovery', 'valid_source_ratio']"
    ).to_string(index=False))


if __name__ == "__main__":
    main()
