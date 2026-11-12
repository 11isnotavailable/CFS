"""Create deterministic, image-disjoint views of a frozen oracle dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sizes", default="320,64,128")
    parser.add_argument("--names", default="train,tune,test")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--shard-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    names = args.names.split(",")
    if len(sizes) != len(names):
        raise ValueError("--sizes and --names must have equal length")
    files = sorted(args.source.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"No oracle shards found in {args.source}")
    keys = np.load(files[0]).files
    row_keys = [key for key in keys if key != "alpha_grid"]
    parts = {key: [] for key in row_keys}
    for file in files:
        with np.load(file) as archive:
            for key in row_keys:
                parts[key].append(archive[key])
    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    count = len(next(iter(arrays.values())))
    if sum(sizes) > count:
        raise ValueError(f"Requested {sum(sizes)} rows from only {count}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(count)
    source_metadata = json.loads((args.source / "metadata.json").read_text("utf-8"))
    alpha_grid = np.load(files[0])["alpha_grid"]
    cursor = 0
    manifest = {}
    for name, size in zip(names, sizes):
        indices = order[cursor : cursor + size]
        cursor += size
        destination = args.output_root / name
        destination.mkdir(parents=True, exist_ok=True)
        for shard, begin in enumerate(range(0, size, args.shard_size)):
            chosen = indices[begin : begin + args.shard_size]
            payload = {key: value[chosen] for key, value in arrays.items()}
            payload["alpha_grid"] = alpha_grid
            np.savez_compressed(destination / f"shard_{shard:05d}.npz", **payload)
        metadata = dict(source_metadata)
        metadata.update(
            {
                "split_parent": str(args.source.resolve()),
                "split_name": name,
                "split_seed": args.seed,
                "split_samples": size,
                "split_protocol": "deterministic image-disjoint pilot split",
            }
        )
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        manifest[name] = {
            "size": size,
            "dataset_indices": arrays["dataset_index"][indices].astype(int).tolist(),
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "split_manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "seed": args.seed,
                "unused_rows": count - sum(sizes),
                "splits": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({key: value["size"] for key, value in manifest.items()}))


if __name__ == "__main__":
    main()
