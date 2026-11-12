"""Bootstrap summaries for the exact simultaneous-gating subset oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def interval(values: np.ndarray) -> tuple[float, float, float]:
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.samples)
    rng = np.random.default_rng(args.seed)
    rows = []
    for keep, group in frame.groupby("keep", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        indices = rng.integers(0, n, size=(args.bootstrap, n))
        arrays = {
            "oracle_kl": group.oracle_kl.to_numpy(),
            "taylor_kl": group.taylor_kl.to_numpy(),
            "oracle_accuracy": group.oracle_correct.to_numpy(),
            "taylor_accuracy": group.taylor_correct.to_numpy(),
            "oracle_dense_fidelity": group.oracle_dense_agreement.to_numpy(),
            "taylor_dense_fidelity": group.taylor_dense_agreement.to_numpy(),
        }
        boot = {name: values[indices].mean(axis=1) for name, values in arrays.items()}
        boot["oracle_minus_taylor_kl"] = boot["oracle_kl"] - boot["taylor_kl"]
        boot["oracle_minus_taylor_accuracy"] = boot["oracle_accuracy"] - boot["taylor_accuracy"]
        boot["oracle_minus_taylor_fidelity"] = boot["oracle_dense_fidelity"] - boot["taylor_dense_fidelity"]
        for metric, values in boot.items():
            mean, low, high = interval(values)
            rows.append(
                {
                    "keep": int(keep),
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "samples": n,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "exact_joint_oracle_bootstrap.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
