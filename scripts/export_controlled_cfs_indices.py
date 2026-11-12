"""Export a frozen controlled-scaling test subset for the CFS collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument(
        "--split", choices=("train", "tune", "test"), default="test"
    )
    args = parser.parse_args()
    archive = np.load(args.split_manifest)
    population = archive[args.split].astype(np.int64)
    count = min(args.count, len(population))
    offsets = np.linspace(0, len(population) - 1, count, dtype=int)
    selected = population[offsets]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "sample_indices": selected.tolist(),
                "source_split": str(args.split_manifest.resolve()),
                "population": f"frozen stratified {args.split} only",
                "requested_samples": count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={args.output} samples={count}")


if __name__ == "__main__":
    main()
