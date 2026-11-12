"""Create one frozen, non-overlapping 32x129 validation context slice."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--tokens", type=int, default=129)
    p.add_argument("--seed", type=int, default=20260802)
    args = p.parse_args()
    tokens = np.memmap(args.validation, dtype="<u2", mode="r")
    blocks = (len(tokens) - 1) // args.tokens
    rng = np.random.default_rng(args.seed)
    block_ids = np.sort(rng.choice(blocks, size=args.samples, replace=False))
    contexts = np.stack([
        np.asarray(tokens[index * args.tokens:(index + 1) * args.tokens], dtype=np.int64)
        for index in block_ids
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, input_ids=contexts, block_ids=block_ids)
    metadata = {
        "source": str(args.validation), "samples": args.samples,
        "tokens_per_context": args.tokens, "seed": args.seed,
        "block_ids": block_ids.tolist(),
        "input_ids_sha256": hashlib.sha256(contexts.tobytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
