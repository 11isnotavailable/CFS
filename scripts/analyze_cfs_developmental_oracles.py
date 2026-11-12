"""Mechanism audit for CFS-guided developmental ViT-S checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_representative_predictor import load_shards


def expected_best_of_k(values: np.ndarray, k: int) -> np.ndarray:
    ordered = np.sort(values, axis=-1)
    count = ordered.shape[-1]
    weights = np.zeros(count, dtype=np.float64)
    denominator = math.comb(count, k)
    for rank in range(k - 1, count):
        weights[rank] = math.comb(rank, k - 1) / denominator
    return (ordered * weights).sum(axis=-1)


def shard_field(directory: Path, name: str) -> np.ndarray:
    values = []
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as archive:
            values.append(archive[name])
    return np.concatenate(values, axis=0)


def graph_structure(scores: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    heads = scores.shape[-1]
    best = np.nanargmax(scores, axis=-1)
    concentrations, entropies = [], []
    for sample in range(len(scores)):
        choices = best[sample, valid[sample]]
        if not len(choices):
            continue
        counts = np.bincount(choices, minlength=heads).astype(np.float64)
        probability = counts[counts > 0] / counts.sum()
        concentrations.append(counts.max() / counts.sum())
        entropy = -(probability * np.log(probability)).sum()
        entropies.append(entropy / np.log(heads))
    return float(np.mean(concentrations)), float(np.mean(entropies))


def analyze_layer(
    policy: str,
    directory: Path,
    layer_position: int,
    layer_index: int,
    minimum_drop: float,
) -> dict[str, float | int | str]:
    archive = load_shards(directory)
    substitution = archive["substitution"][:, layer_position].astype(np.float64)
    drops = archive["single_drop"][:, layer_position].astype(np.float64)
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[-1]
    diagonal = np.arange(heads)
    scores[:, diagonal, diagonal] = np.nan
    substitution[:, diagonal, diagonal] = np.nan
    valid = drops > minimum_drop
    oracle = np.nanmax(scores, axis=-1)
    best_substitution = np.nanmin(substitution, axis=-1)
    finite_candidates = scores[valid].reshape(-1, heads)
    finite_candidates = finite_candidates[np.isfinite(finite_candidates)].reshape(
        int(valid.sum()), heads - 1
    )
    matched = expected_best_of_k(finite_candidates, 2)
    concentration, entropy = graph_structure(scores, valid)
    prediction = shard_field(directory, "prediction")
    label = shard_field(directory, "imagenet_target")
    return {
        "policy": policy,
        "layer": layer_index,
        "samples": len(scores),
        "heads": heads,
        "valid_source_fraction": float(valid.mean()),
        "mean_drop_kl": float(drops[valid].mean()),
        "mean_best_substitution_kl": float(best_substitution[valid].mean()),
        "raw_oracle": float(oracle[valid].mean()),
        "matched_best_of_2": float(matched.mean()),
        "source_fraction_s_gt_025": float((oracle[valid] > 0.25).mean()),
        "representative_concentration": concentration,
        "representative_entropy": entropy,
        "slice_accuracy": float((prediction == label).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = (
        "ce_baseline", "safe_adversarial", "soft_all", "coverage_cfs",
        "sensitivity_cfs", "head_exchange",
    )
    layers = (3, 7, 11)
    rows = []
    for policy in policies:
        directory = args.root / f"cfs_multilayer_{policy}_final512"
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("selected_layers") not in (None, list(layers)):
            raise ValueError(f"Unexpected layers for {policy}: {metadata['selected_layers']}")
        for position, layer in enumerate(layers):
            rows.append(analyze_layer(
                policy, directory, position, layer, args.minimum_drop_kl
            ))
    frame = pd.DataFrame(rows)
    macro = frame.groupby("policy", as_index=False).agg({
        "valid_source_fraction": "mean",
        "mean_drop_kl": "mean",
        "mean_best_substitution_kl": "mean",
        "raw_oracle": "mean",
        "matched_best_of_2": "mean",
        "source_fraction_s_gt_025": "mean",
        "representative_concentration": "mean",
        "representative_entropy": "mean",
        "slice_accuracy": "first",
    })
    frame.to_csv(args.output_dir / "developmental_oracle_by_layer.csv", index=False)
    macro.to_csv(args.output_dir / "developmental_oracle_macro.csv", index=False)
    print("BY_LAYER")
    print(frame.to_string(index=False))
    print("MACRO")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
