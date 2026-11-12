"""Analyze diagnostic head repartitions of one frozen ViT checkpoint.

The resulting trajectory is a sensitivity diagnostic, not controlled training:
changing the QKV partition changes the computation without retraining weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_functional_capacity import spectral_effective_rank
from scripts.analyze_scaling_cfs import (
    expected_best_of_k,
    per_input_representative_structure,
    shard_field,
)
from scripts.train_representative_predictor import load_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry", action="append", nargs=2, metavar=("HEADS", "ORACLE_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--top-source-fraction", type=float, default=0.75)
    return parser.parse_args()


def top_fraction_mask(drops: np.ndarray, fraction: float) -> np.ndarray:
    heads = drops.shape[-1]
    count = min(heads, max(1, int(round(fraction * heads))))
    order = np.argsort(drops, axis=-1)
    mask = np.zeros_like(drops, dtype=bool)
    rows = np.arange(len(drops))[:, None]
    mask[rows, order[:, -count:]] = True
    return mask


def analyze(
    head_label: str, directory: Path, minimum_drop: float,
    top_source_fraction: float,
) -> tuple[dict, list[dict], list[dict]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    archive = load_shards(directory)
    substitution = archive["substitution"][:, 0].astype(np.float64)
    drops = archive["single_drop"][:, 0].astype(np.float64)
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[-1]
    diagonal = np.arange(heads)
    scores[:, diagonal, diagonal] = np.nan
    valid = drops > minimum_drop
    oracle = np.nanmax(scores, axis=-1)
    top_sources = top_fraction_mask(drops, top_source_fraction)
    substitution_without_self = substitution.copy()
    substitution_without_self[:, diagonal, diagonal] = np.nan
    best_substitution = np.nanmin(substitution_without_self, axis=-1)
    recovered_kl = drops - best_substitution
    weighted_recovery_by_input = np.divide(
        recovered_kl.sum(axis=1), drops.sum(axis=1),
        out=np.full(len(drops), np.nan), where=drops.sum(axis=1) > 0,
    )
    positive_weighted_recovery_by_input = np.divide(
        np.maximum(recovered_kl, 0).sum(axis=1), drops.sum(axis=1),
        out=np.full(len(drops), np.nan), where=drops.sum(axis=1) > 0,
    )
    candidate_rows = scores[valid]
    finite_candidates = candidate_rows[np.isfinite(candidate_rows)].reshape(
        int(valid.sum()), heads - 1
    )
    matched_k = min(2, heads - 1)
    matched = expected_best_of_k(finite_candidates, matched_k)
    top_candidate_rows = scores[top_sources]
    top_finite_candidates = top_candidate_rows[
        np.isfinite(top_candidate_rows)
    ].reshape(int(top_sources.sum()), heads - 1)
    top_matched = expected_best_of_k(top_finite_candidates, matched_k)
    concentration, entropy = per_input_representative_structure(scores, valid)
    rank = spectral_effective_rank(scores, valid)
    prediction = shard_field(directory, "prediction")
    target = shard_field(directory, "imagenet_target")
    summary = {
        "heads": heads,
        "head_label": head_label,
        "samples": len(scores),
        "accuracy": float((prediction == target).mean()),
        "valid_source_ratio": float(valid.mean()),
        "oracle_recovery": float(oracle[valid].mean()),
        "matched_best_of_2": float(matched.mean()),
        "top_source_fraction": float(top_sources.mean()),
        "top_source_oracle_recovery": float(oracle[top_sources].mean()),
        "top_source_matched_best_of_2": float(top_matched.mean()),
        "mean_drop_kl": float(drops.mean()),
        "kl_weighted_recovery": float(np.nanmean(weighted_recovery_by_input)),
        "positive_kl_weighted_recovery": float(
            np.nanmean(positive_weighted_recovery_by_input)
        ),
        "representative_concentration": float(np.nanmean(concentration)),
        "representative_entropy": float(np.nanmean(entropy)),
        "spectral_rank": float(rank.mean()),
        "spectral_rank_per_head": float((rank / heads).mean()),
        "alpha_steps": len(metadata["alpha_grid"]),
        "diagnostic_repartition": bool(metadata.get("head_repartition_diagnostic")),
        "original_checkpoint_heads": metadata.get("original_checkpoint_heads"),
    }
    threshold_rows = []
    for threshold in (0.25, 0.5, 0.75):
        adjacency = (scores > threshold) & valid[:, :, None]
        valid_pairs = valid[:, :, None] & valid[:, None, :]
        valid_pairs[:, diagonal, diagonal] = False
        mutual = adjacency & adjacency.transpose(0, 2, 1)
        threshold_rows.append({
            "heads": heads, "threshold": threshold,
            "substitutable_source_fraction": float((oracle[valid] > threshold).mean()),
            "top_source_substitutable_fraction": float(
                (oracle[top_sources] > threshold).mean()
            ),
            "directed_edge_density": float(adjacency[valid_pairs].mean()),
            "mutual_edge_density": float(mutual[valid_pairs].mean()),
        })
    sample_rows = []
    for sample in range(len(scores)):
        sample_rows.append({
            "heads": heads, "head_label": head_label, "sample": sample,
            "accuracy": int(prediction[sample] == target[sample]),
            "valid_source_ratio": float(valid[sample].mean()),
            "oracle_recovery": float(oracle[sample, valid[sample]].mean())
            if valid[sample].any() else np.nan,
            "top_source_oracle_recovery": float(
                oracle[sample, top_sources[sample]].mean()
            ),
            "kl_weighted_recovery": float(weighted_recovery_by_input[sample]),
            "positive_kl_weighted_recovery": float(
                positive_weighted_recovery_by_input[sample]
            ),
            "mean_drop_kl": float(drops[sample].mean()),
        })
    return summary, threshold_rows, sample_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, thresholds, samples = [], [], []
    for heads, directory in args.entry:
        summary, rows, detail = analyze(
            heads, Path(directory), args.minimum_drop_kl, args.top_source_fraction
        )
        summaries.append(summary); thresholds.extend(rows); samples.extend(detail)
    summary_frame = pd.DataFrame(summaries).sort_values("heads")
    threshold_frame = pd.DataFrame(thresholds).sort_values(["threshold", "heads"])
    summary_frame.to_csv(args.output_dir / "head_repartition_summary.csv", index=False)
    threshold_frame.to_csv(args.output_dir / "head_repartition_thresholds.csv", index=False)
    pd.DataFrame(samples).to_csv(
        args.output_dir / "head_repartition_samples.csv", index=False
    )
    print(summary_frame.to_string(index=False))
    print(threshold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
