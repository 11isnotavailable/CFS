"""Restore Hugging Face Imagenette image parquet into ImageFolder layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--class-root", type=Path, required=True,
        help="Existing ImageFolder split whose sorted synset directories define labels.",
    )
    args = parser.parse_args()
    classes = sorted(path.name for path in args.class_root.iterdir() if path.is_dir())
    if len(classes) != 10:
        raise ValueError(f"Expected 10 Imagenette classes, found {classes}")
    for name in classes:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    table = pq.ParquetFile(args.parquet)
    count = 0
    for batch in table.iter_batches(columns=["image", "label"], batch_size=512):
        for record in batch.to_pylist():
            label = int(record["label"])
            image = record["image"]
            name = Path(image["path"]).name
            payload = image["bytes"]
            if payload is None:
                raise ValueError(f"Missing embedded bytes for {name}")
            (args.output / classes[label] / name).write_bytes(payload)
            count += 1
    print(f"images={count} classes={classes} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
