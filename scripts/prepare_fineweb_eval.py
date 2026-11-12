"""Freeze reproducible token contexts from the FineWeb-Edu 10BT sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--parquet", type=Path,
        help="Optional local FineWeb-Edu parquet shard (avoids Hub API listing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rng = np.random.default_rng(args.seed)
    contexts, document_ids = [], []
    needed = args.sequence_length + 1
    if args.parquet is None:
        rows = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        ).shuffle(seed=args.seed, buffer_size=args.buffer_size)
    else:
        parquet = pq.ParquetFile(args.parquet)
        groups = rng.permutation(parquet.num_row_groups)

        def local_rows():
            for group in groups:
                table = parquet.read_row_group(int(group), columns=["text", "id"])
                order = rng.permutation(len(table))
                texts = table.column("text")
                ids = table.column("id")
                for index in order:
                    yield {"text": texts[int(index)].as_py(), "id": ids[int(index)].as_py()}

        rows = local_rows()
    for row in rows:
        tokens = tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"]
        if len(tokens) < needed:
            continue
        start = int(rng.integers(0, len(tokens) - needed + 1))
        contexts.append(tokens[start : start + needed])
        document_ids.append(str(row.get("id", "")))
        if len(contexts) >= args.samples:
            break
    if len(contexts) != args.samples:
        raise RuntimeError(f"Collected only {len(contexts)} contexts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        input_ids=np.asarray(contexts, dtype=np.int32),
        document_id=np.asarray(document_ids),
    )
    metadata = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "source_split": "train",
        "role": "frozen evaluation slice (not an official validation split)",
        "model_tokenizer": args.model,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "buffer_size": args.buffer_size,
        "local_parquet": str(args.parquet) if args.parquet is not None else None,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"saved={args.output} shape={np.asarray(contexts).shape}")


if __name__ == "__main__":
    main()
