"""Analyze Qwen next-token CFS shards with exact set-cover optimization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--thresholds", default="0.25,0.5,0.75")
    return parser.parse_args()


def load_shards(directory: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[np.ndarray]] = {}
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as archive:
            for name in archive.files:
                fields.setdefault(name, []).append(archive[name])
    if not fields:
        raise FileNotFoundError(f"No shards in {directory}")
    return {name: np.concatenate(parts) for name, parts in fields.items()}


def expected_best_two(values: np.ndarray) -> float:
    ordered = np.sort(values)
    count = len(ordered)
    if count <= 2:
        return float(ordered[-1])
    denominator = math.comb(count, 2)
    return float(sum(math.comb(rank, 1) * ordered[rank] for rank in range(1, count)) / denominator)


def minimum_cover(scores: np.ndarray, valid: np.ndarray, threshold: float) -> int:
    heads = len(valid)
    active = np.flatnonzero(valid)
    if len(active) == 0:
        return 0
    coverage = np.zeros((len(active), heads), dtype=np.float64)
    for row, source in enumerate(active):
        coverage[row, source] = 1
        coverage[row, np.flatnonzero(scores[source] > threshold)] = 1
    result = milp(
        c=np.ones(heads), integrality=np.ones(heads), bounds=Bounds(0, 1),
        constraints=LinearConstraint(coverage, 1, np.inf),
        options={"time_limit": 30.0},
    )
    if not result.success:
        raise RuntimeError(f"Set-cover MILP failed: {result.message}")
    return int(round(result.fun))


def main():
    args = parse_args()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    data = load_shards(args.input_dir)
    drops = data["single_drop"].astype(np.float64)
    substitution = data["substitution"].astype(np.float64)
    scores = 1 - substitution / np.maximum(drops[..., None], 1e-12)
    heads = int(metadata["heads"])
    diagonal = np.arange(heads)
    scores[..., diagonal, diagonal] = np.nan
    valid = drops > args.minimum_drop_kl
    thresholds = [float(value) for value in args.thresholds.split(",")]
    loss = float(data["loss_sum"].sum() / data["token_count"].sum())
    rows = []
    for layer_offset, layer in enumerate(metadata["layers"]):
        local_scores = scores[:, layer_offset]
        local_valid = valid[:, layer_offset]
        oracle_values, matched_values, entropy_values = [], [], []
        covers = {threshold: [] for threshold in thresholds}
        for sample in range(len(local_scores)):
            choices = []
            for source in np.flatnonzero(local_valid[sample]):
                candidates = local_scores[sample, source]
                finite = candidates[np.isfinite(candidates)]
                oracle_values.append(float(np.max(finite)))
                matched_values.append(expected_best_two(finite))
                choices.append(int(np.nanargmax(candidates)))
            if choices:
                counts = np.bincount(choices, minlength=heads).astype(float)
                probability = counts[counts > 0] / counts.sum()
                entropy_values.append(float(-(probability * np.log(probability)).sum() / np.log(heads)))
            for threshold in thresholds:
                covers[threshold].append(minimum_cover(local_scores[sample], local_valid[sample], threshold))
        for threshold in thresholds:
            cover = np.asarray(covers[threshold], dtype=float)
            rows.append({
                "model": metadata["model"], "parameters": metadata["parameters"],
                "heads": heads, "kv_heads": metadata["kv_heads"], "depth": metadata["depth"],
                "layer": layer, "depth_fraction": (layer + 1) / metadata["depth"],
                "samples": len(local_scores), "validation_loss": loss,
                "perplexity": math.exp(min(loss, 30)), "threshold": threshold,
                "valid_source_rate": float(local_valid.mean()),
                "raw_oracle": float(np.mean(oracle_values)),
                "matched_best_of_2": float(np.mean(matched_values)),
                "minimum_cover": float(cover.mean()), "cover_per_head": float(cover.mean() / heads),
                "representative_entropy": float(np.mean(entropy_values)),
                "cover_solver": "scipy.optimize.milp exact binary set cover",
            })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
