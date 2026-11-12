"""Analyze controlled GPT-2 CFS shards without optional solver dependencies."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except ImportError:
    Bounds = LinearConstraint = milp = None


def load_shards(directory):
    fields = {}
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as archive:
            for name in archive.files:
                fields.setdefault(name, []).append(archive[name])
    if not fields:
        raise FileNotFoundError(directory)
    return {name: np.concatenate(parts) for name, parts in fields.items()}


def expected_best_two(values):
    ordered = np.sort(values)
    n = len(ordered)
    if n <= 2:
        return float(ordered[-1])
    denominator = math.comb(n, 2)
    return float(sum(math.comb(rank, 1) * ordered[rank] for rank in range(1, n)) / denominator)


def exact_cover(scores, valid, threshold):
    active = np.flatnonzero(valid)
    if len(active) == 0:
        return 0
    if milp is not None:
        heads = len(valid)
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
            raise RuntimeError(f"Exact set-cover MILP failed: {result.message}")
        return int(round(result.fun))
    full = (1 << len(active)) - 1
    masks = []
    for representative in range(len(valid)):
        mask = 0
        for bit, source in enumerate(active):
            if representative == source or scores[source, representative] > threshold:
                mask |= 1 << bit
        masks.append(mask)
    for size in range(1, len(valid) + 1):
        for selected in itertools.combinations(range(len(valid)), size):
            union = 0
            for representative in selected:
                union |= masks[representative]
                if union == full:
                    return size
    raise RuntimeError("Exact set cover failed")


def effective_rank(scores, valid):
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


def summarize(scores, drops, heads, threshold):
    valid = drops > 1e-5
    oracle, matched, covers, ranks = [], [], [], []
    for sample in range(len(scores)):
        for source in np.flatnonzero(valid[sample]):
            candidates = scores[sample, source]
            finite = candidates[np.isfinite(candidates)]
            oracle.append(float(np.max(finite)))
            matched.append(expected_best_two(finite))
        covers.append(exact_cover(scores[sample], valid[sample], threshold))
        ranks.append(effective_rank(scores[sample], valid[sample]))
    return {
        "valid_source_rate": float(valid.mean()),
        "raw_oracle": float(np.mean(oracle)),
        "matched_2": float(np.mean(matched)),
        "minimum_cover": float(np.mean(covers)),
        "cover_per_head": float(np.mean(covers) / heads),
        "spectral_rank": float(np.mean(ranks)),
        "rank_per_head": float(np.mean(ranks) / heads),
        "mean_deletion_kl": float(drops[valid].mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    data = load_shards(args.input_dir)
    drops = data["single_drop"].astype(np.float64)
    substitution = data["substitution"].astype(np.float64)
    scores = 1 - substitution / np.maximum(drops[..., None], 1e-12)
    diagonal = np.arange(metadata["heads"])
    scores[..., diagonal, diagonal] = np.nan
    common = {
        "model": metadata["name"], "parameters": metadata["parameters"],
        "depth": metadata["depth"], "width": metadata["hidden_size"],
        "heads": metadata["heads"], "head_dim": metadata["head_dim"],
        "samples": metadata["samples"], "minimum_drop_kl": 1e-5,
        "cover_threshold": args.threshold,
        "training_validation_loss": metadata["training_final_validation_loss"],
        "slice_loss": float(data["loss_sum"].sum() / data["token_count"].sum()),
    }
    rows = []
    for offset, layer in enumerate(metadata["layers"]):
        row = dict(common)
        row.update({"aggregation": "layer", "layer": layer,
                    "depth_fraction": (layer + 1) / metadata["depth"]})
        row.update(summarize(scores[:, offset], drops[:, offset], metadata["heads"], args.threshold))
        rows.append(row)
    flattened_scores = scores.reshape(-1, metadata["heads"], metadata["heads"])
    flattened_drops = drops.reshape(-1, metadata["heads"])
    row = dict(common)
    row.update({"aggregation": "three_layer_mean", "layer": -1, "depth_fraction": -1})
    row.update(summarize(flattened_scores, flattened_drops, metadata["heads"], args.threshold))
    rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output.with_suffix(".json")).write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
