"""Analyze CFS oracle matrices across a controlled ViT scaling family."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.train_representative_predictor import load_shards


def shard_field(directory: Path, name: str) -> np.ndarray:
    values = []
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as archive:
            values.append(archive[name])
    if not values:
        raise FileNotFoundError(f"No oracle shards found in {directory}")
    return np.concatenate(values, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", type=Path, required=True)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--large", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument(
        "--max-samples", type=int,
        help="Optional matched prefix for partial-run diagnostics.",
    )
    return parser.parse_args()


def expected_best_of_k(values: np.ndarray, k: int) -> np.ndarray:
    """Exact expected maximum of a uniform size-k subset, row-wise."""
    ordered = np.sort(values, axis=-1)
    count = ordered.shape[-1]
    if k > count:
        raise ValueError(f"Cannot draw {k} from {count} candidates.")
    weights = np.zeros(count, dtype=np.float64)
    denominator = math.comb(count, k)
    for rank in range(k - 1, count):
        weights[rank] = math.comb(rank, k - 1) / denominator
    return (ordered * weights).sum(axis=-1)


def per_input_representative_structure(
    scores: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    heads = scores.shape[-1]
    best = np.nanargmax(scores, axis=-1)
    concentration = np.full(len(scores), np.nan, dtype=np.float64)
    entropy = np.full(len(scores), np.nan, dtype=np.float64)
    for sample in range(len(scores)):
        choices = best[sample, valid[sample]]
        if len(choices) == 0:
            continue
        counts = np.bincount(choices, minlength=heads).astype(np.float64)
        probability = counts[counts > 0] / counts.sum()
        concentration[sample] = float(counts.max() / counts.sum())
        value = -(probability * np.log(probability)).sum()
        entropy[sample] = float(value / np.log(heads)) if heads > 1 else 0.0
    return concentration, entropy


def analyze(
    name: str,
    directory: Path,
    minimum_drop: float,
    max_samples: int | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    archive = load_shards(directory)
    limit = slice(None, max_samples)
    substitution = archive["substitution"][limit, 0].astype(np.float64)
    drops = archive["single_drop"][limit, 0].astype(np.float64)
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[-1]
    diagonal = np.arange(heads)
    scores[:, diagonal, diagonal] = np.nan
    valid = drops > minimum_drop
    oracle = np.nanmax(scores, axis=-1)
    candidate_values = scores[valid].reshape(-1, heads)
    candidate_values = candidate_values[np.isfinite(candidate_values)].reshape(
        int(valid.sum()), heads - 1
    )
    best_of_two = expected_best_of_k(candidate_values, 2)
    concentration, representative_entropy = per_input_representative_structure(
        scores, valid
    )
    matched_by_source = np.full(valid.shape, np.nan, dtype=np.float64)
    matched_by_source[valid] = best_of_two
    oracle_by_input = np.asarray(
        [oracle[index, valid[index]].mean() for index in range(len(scores))]
    )
    matched_by_input = np.nanmean(matched_by_source, axis=1)
    prediction = shard_field(directory, "prediction")[limit]
    label = shard_field(directory, "imagenet_target")[limit]
    summary = {
        "model_size": name,
        "model": metadata["model"],
        "heads": heads,
        "samples": len(scores),
        "valid_sources": int(valid.sum()),
        "total_sources": int(valid.size),
        "valid_source_coverage": float(valid.mean()),
        "classification_accuracy": float((prediction == label).mean()),
        "oracle_recovery": float(oracle[valid].mean()),
        "matched_best_of_2_recovery": float(best_of_two.mean()),
        "oracle_candidate_advantage": float(
            oracle[valid].mean() - best_of_two.mean()
        ),
        "best_representative_concentration": float(np.nanmean(concentration)),
        "best_representative_entropy": float(np.nanmean(representative_entropy)),
    }
    sample_rows = [
        {
            "model_size": name,
            "sample": index,
            "valid_source_coverage": float(valid[index].mean()),
            "classification_correct": int(prediction[index] == label[index]),
            "oracle_recovery": float(oracle_by_input[index]),
            "matched_best_of_2_recovery": float(matched_by_input[index]),
            "best_representative_concentration": float(concentration[index]),
            "best_representative_entropy": float(representative_entropy[index]),
        }
        for index in range(len(scores))
    ]
    threshold_rows = []
    for threshold in (0.25, 0.5, 0.75, 0.9):
        candidate_count = np.sum(candidate_values > threshold, axis=-1)
        threshold_rows.append(
            {
                "model_size": name,
                "heads": heads,
                "threshold": threshold,
                "substitutable_source_fraction": float(
                    (oracle[valid] > threshold).mean()
                ),
                "effective_non_substitutable_ratio": float(
                    (oracle[valid] <= threshold).mean()
                ),
                "mean_substitute_fraction": float(
                    (candidate_count / (heads - 1)).mean()
                ),
                "at_least_two_substitutes": float(
                    (candidate_count >= 2).mean()
                ),
            }
        )
    return summary, threshold_rows, sample_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    thresholds = []
    samples = []
    entries = [
        ("tiny", args.tiny),
        ("small", args.small),
        ("base", args.base),
    ]
    if args.large is not None:
        entries.append(("large", args.large))
    for name, directory in entries:
        summary, rows, detail = analyze(
            name, directory, args.minimum_drop_kl, args.max_samples
        )
        summaries.append(summary)
        thresholds.extend(rows)
        samples.extend(detail)
    summary_frame = pd.DataFrame(summaries)
    threshold_frame = pd.DataFrame(thresholds)
    sample_frame = pd.DataFrame(samples)
    summary_frame.to_csv(args.output_dir / "scaling_summary.csv", index=False)
    threshold_frame.to_csv(
        args.output_dir / "scaling_thresholds.csv", index=False
    )
    sample_frame.to_csv(args.output_dir / "scaling_samples.csv", index=False)
    print(summary_frame.to_string(index=False))
    print(threshold_frame.to_string(index=False))


if __name__ == "__main__":
    main()
