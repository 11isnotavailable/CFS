"""Summarize source-head deletion KL before applying a validity threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+", type=Path)
    parser.add_argument("--thresholds", default="1e-7,1e-6,1e-5,1e-4")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    for directory in args.input_dirs:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        shards = []
        for path in sorted(directory.glob("shard_*.npz")):
            with np.load(path) as archive:
                shards.append(archive["single_drop"].astype(np.float64))
        drops = np.concatenate(shards)
        for offset, layer in enumerate(metadata["layers"]):
            values = drops[:, offset].reshape(-1)
            quantiles = np.quantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
            rates = {f"rate_gt_{threshold:g}": float(np.mean(values > threshold)) for threshold in thresholds}
            print(json.dumps({
                "model": metadata["model"],
                "directory": str(directory),
                "layer": layer,
                "samples": len(drops),
                "heads": metadata["heads"],
                "drop_kl_quantiles": {
                    name: float(value) for name, value in zip(
                        ["min", "p10", "p25", "median", "p75", "p90", "max"], quantiles
                    )
                },
                **rates,
            }))


if __name__ == "__main__":
    main()
