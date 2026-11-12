"""Fit descriptive capacity exponents with matched-image bootstrap intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), deg=1)[0])


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.analysis_dir / "functional_capacity_summary.csv")
    samples = pd.read_csv(args.analysis_dir / "functional_capacity_samples.csv")
    rng = np.random.default_rng(args.seed)
    rows = []
    for (depth, threshold), group in summary.groupby(
        ["depth_fraction", "threshold"], sort=True
    ):
        group = group.sort_values("parameters")
        if len(group) < 3:
            continue
        names = group["model_size"].tolist()
        parameters = group["parameters"].to_numpy(dtype=np.float64)
        heads = group["heads"].to_numpy(dtype=np.float64)
        detail = samples[
            np.isclose(samples["depth_fraction"], depth)
            & np.isclose(samples["threshold"], threshold)
            & samples["model_size"].isin(names)
        ]
        cover = detail.pivot(
            index="sample", columns="model_size", values="minimum_cover"
        )[names].to_numpy(dtype=np.float64)
        spectral = detail.pivot(
            index="sample", columns="model_size", values="spectral_effective_rank"
        )[names].to_numpy(dtype=np.float64)
        if np.any(cover.mean(axis=0) <= 0) or np.any(spectral.mean(axis=0) <= 0):
            continue
        for metric, values in (
            ("minimum_cover", cover),
            ("spectral_effective_rank", spectral),
        ):
            estimates = {
                "parameters": slope(parameters, values.mean(axis=0)),
                "heads": slope(heads, values.mean(axis=0)),
            }
            boot = {"parameters": [], "heads": []}
            for _ in range(args.bootstrap):
                indices = rng.integers(0, len(values), size=len(values))
                means = values[indices].mean(axis=0)
                boot["parameters"].append(slope(parameters, means))
                boot["heads"].append(slope(heads, means))
            for x_name in ("parameters", "heads"):
                interval = np.quantile(boot[x_name], [0.025, 0.975])
                rows.append(
                    {
                        "depth_fraction": depth,
                        "threshold": threshold,
                        "models": ",".join(names),
                        "metric": metric,
                        "x_axis": x_name,
                        "exponent": estimates[x_name],
                        "bootstrap_ci_low": float(interval[0]),
                        "bootstrap_ci_high": float(interval[1]),
                        "bootstrap_unit": "matched input image",
                        "bootstrap_replicates": args.bootstrap,
                    }
                )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
