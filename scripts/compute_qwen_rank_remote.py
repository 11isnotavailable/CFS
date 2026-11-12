"""Compute unified CFS spectral rank/head from saved Qwen raw shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


RUNS = {
    "Qwen/Qwen2.5-0.5B": "qwen05b_random8_32_alpha16",
    "Qwen/Qwen2.5-1.5B": "qwen15b_random8_32_alpha16",
    "Qwen/Qwen2.5-3B": "qwen3b_random8_32_alpha16",
    "Qwen/Qwen2.5-7B": "qwen7b_random8_32_alpha16",
}


def effective_rank(scores: np.ndarray, valid: np.ndarray) -> float:
    """Directed entropy effective rank with unit self-affinity for valid heads."""
    affinity = np.clip(np.nan_to_num(scores, nan=0.0), 0.0, 1.0)
    affinity[~valid] = 0.0
    diagonal = np.arange(len(valid))
    affinity[diagonal, diagonal] = valid.astype(np.float64)
    singular = np.linalg.svd(affinity, compute_uv=False)
    total = singular.sum()
    if total <= 0:
        return 0.0
    probability = singular[singular > 0] / total
    return math.exp(float(-(probability * np.log(probability)).sum()))


def load_run(directory: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    drops, substitutions = [], []
    shards = sorted(directory.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shards in {directory}")
    for shard in shards:
        with np.load(shard) as archive:
            drops.append(archive["single_drop"].astype(np.float64))
            substitutions.append(archive["substitution"].astype(np.float64))
    return metadata, np.concatenate(drops), np.concatenate(substitutions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    args = parser.parse_args()

    rows = []
    for model, run in RUNS.items():
        metadata, drop, substitution = load_run(args.outputs_root / run)
        samples, layer_count, heads = drop.shape
        selected_layers = metadata.get("selected_layers", metadata.get("layers"))
        if not isinstance(selected_layers, list):
            selected_layers = list(range(layer_count))
        for layer_offset in range(layer_count):
            layer_ranks = []
            layer_valid = []
            for sample in range(samples):
                valid = drop[sample, layer_offset] > args.minimum_drop_kl
                scores = 1 - substitution[sample, layer_offset] / np.maximum(
                    drop[sample, layer_offset, :, None], 1e-12
                )
                np.fill_diagonal(scores, np.nan)
                layer_ranks.append(effective_rank(scores, valid))
                layer_valid.append(valid.mean())
            rows.append(
                {
                    "model": model,
                    "run": run,
                    "samples": samples,
                    "layer": int(selected_layers[layer_offset]),
                    "layer_offset": layer_offset,
                    "heads": heads,
                    "minimum_drop_kl": args.minimum_drop_kl,
                    "spectral_effective_rank": float(np.mean(layer_ranks)),
                    "rank_per_head": float(np.mean(layer_ranks) / heads),
                    "valid_source_rate": float(np.mean(layer_valid)),
                }
            )

    frame = pd.DataFrame(rows)
    overall = (
        frame.groupby(["model", "run", "samples", "heads", "minimum_drop_kl"], as_index=False)
        .agg(
            spectral_effective_rank=("spectral_effective_rank", "mean"),
            rank_per_head=("rank_per_head", "mean"),
            valid_source_rate=("valid_source_rate", "mean"),
        )
    )
    frame["aggregation"] = "per_layer"
    overall["aggregation"] = "three_layer_mean"
    overall["layer"] = -1
    overall["layer_offset"] = -1
    columns = [
        "model", "run", "aggregation", "samples", "layer", "layer_offset",
        "heads", "minimum_drop_kl", "spectral_effective_rank", "rank_per_head",
        "valid_source_rate",
    ]
    result = pd.concat([frame[columns], overall[columns]], ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
