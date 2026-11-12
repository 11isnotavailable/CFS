"""Aggregate pretrained BERT-family MLM CFS results with paired bootstrap."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def expected_best_k(values: np.ndarray, requested: int) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    count = len(values)
    k = min(requested, count)
    if k == count:
        return float(values[-1])
    denominator = math.comb(count, k)
    return float(
        sum(
            math.comb(rank, k - 1) * values[rank]
            for rank in range(k - 1, count)
        )
        / denominator
    )


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
        c=np.ones(heads),
        integrality=np.ones(heads),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(coverage, 1, np.inf),
        options={"time_limit": 30.0},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return int(round(result.fun))


def spectral_rank(scores: np.ndarray, valid: np.ndarray) -> float:
    matrix = np.where(np.isfinite(scores), np.maximum(scores, 0), 0)
    matrix[~valid] = 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    total = singular.sum()
    if total <= 0:
        return 0.0
    probability = singular[singular > 0] / total
    return float(np.exp(-(probability * np.log(probability)).sum()))


def model_metrics(directory: Path, tag: str, minimum_drop: float, threshold: float):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "shard_00000.npz") as archive:
        ids = archive["sample_id"].astype(int)
        drops = archive["single_drop"][:, 0].astype(np.float64)
        substitution = archive["substitution"][:, 0].astype(np.float64)
        loss = float(archive["loss_sum"].sum() / archive["token_count"].sum())
    scores = 1 - substitution / np.maximum(drops[:, :, None], 1e-12)
    diagonal = np.arange(drops.shape[1])
    scores[:, diagonal, diagonal] = np.nan
    valid = drops > minimum_drop
    rows = []
    for offset, sample_id in enumerate(ids):
        oracle = []
        matched_1 = []
        matched_2 = []
        matched_4 = []
        representatives = []
        best_errors = []
        for source in np.flatnonzero(valid[offset]):
            candidates = scores[offset, source]
            values = candidates[np.isfinite(candidates)]
            oracle.append(float(values.max()))
            matched_1.append(expected_best_k(values, 1))
            matched_2.append(expected_best_k(values, 2))
            matched_4.append(expected_best_k(values, 4))
            representative = int(np.nanargmax(candidates))
            representatives.append(representative)
            best_errors.append(float(substitution[offset, source, representative]))
        active_drops = drops[offset, valid[offset]]
        recovered = active_drops - np.asarray(best_errors)
        counts = np.bincount(representatives, minlength=drops.shape[1]).astype(float)
        probability = counts[counts > 0] / max(counts.sum(), 1)
        entropy = (
            float(-(probability * np.log(probability)).sum() / np.log(drops.shape[1]))
            if drops.shape[1] > 1 and len(probability)
            else 0.0
        )
        rows.append(
            {
                "tag": tag,
                "model": metadata["model"],
                "parameters": metadata["parameters"],
                "depth": metadata["depth"],
                "heads": metadata["heads"],
                "head_dim": metadata["head_dim"],
                "hidden_size": metadata["hidden_size"],
                "layer": metadata["layers"][0],
                "sample_id": sample_id,
                "mlm_loss": loss,
                "pseudo_perplexity": math.exp(min(loss, 30)),
                "valid_source_rate": float(valid[offset].mean()),
                "raw_oracle": float(np.mean(oracle)) if oracle else np.nan,
                "matched_1": float(np.mean(matched_1)) if matched_1 else np.nan,
                "matched_2": float(np.mean(matched_2)) if matched_2 else np.nan,
                "matched_4": float(np.mean(matched_4)) if matched_4 else np.nan,
                "weighted_recovery": float(recovered.sum() / max(active_drops.sum(), 1e-30)),
                "mean_drop_kl": float(active_drops.mean()) if len(active_drops) else np.nan,
                "cover": minimum_cover(scores[offset], valid[offset], threshold),
                "cover_per_head": minimum_cover(scores[offset], valid[offset], threshold)
                / metadata["heads"],
                "representative_entropy": entropy,
                "spectral_rank": spectral_rank(scores[offset], valid[offset]),
                "rank_per_head": spectral_rank(scores[offset], valid[offset])
                / metadata["heads"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(frame: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = [
        "valid_source_rate",
        "raw_oracle",
        "matched_1",
        "matched_2",
        "matched_4",
        "weighted_recovery",
        "mean_drop_kl",
        "cover",
        "cover_per_head",
        "representative_entropy",
        "spectral_rank",
        "rank_per_head",
    ]
    rows = []
    for tag, group in frame.groupby("tag", sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            simulations = np.asarray(
                [np.nanmean(rng.choice(values, len(values), replace=True)) for _ in range(repeats)]
            )
            rows.append(
                {
                    "tag": tag,
                    "metric": metric,
                    "mean": float(np.nanmean(values)),
                    "ci95_low": float(np.nanquantile(simulations, 0.025)),
                    "ci95_high": float(np.nanquantile(simulations, 0.975)),
                    "samples": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_contrasts(frame: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    order = [tag for tag in ["tiny", "small", "base", "large"] if tag in set(frame.tag)]
    metrics = ["raw_oracle", "matched_2", "weighted_recovery", "cover_per_head", "rank_per_head"]
    rows = []
    for left, right in zip(order[:-1], order[1:]):
        joined = frame[frame.tag == left].merge(
            frame[frame.tag == right], on="sample_id", suffixes=("_left", "_right")
        )
        for metric in metrics:
            difference = joined[f"{metric}_right"].to_numpy() - joined[f"{metric}_left"].to_numpy()
            simulations = np.asarray(
                [np.nanmean(rng.choice(difference, len(difference), replace=True)) for _ in range(repeats)]
            )
            rows.append(
                {
                    "contrast": f"{right}-{left}",
                    "metric": metric,
                    "mean_difference": float(np.nanmean(difference)),
                    "ci95_low": float(np.nanquantile(simulations, 0.025)),
                    "ci95_high": float(np.nanquantile(simulations, 0.975)),
                    "samples": len(difference),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for tag in ("tiny", "small", "base", "large"):
        directory = args.root / f"{tag}_final"
        if (directory / "shard_00000.npz").exists():
            frames.append(model_metrics(directory, tag, args.minimum_drop_kl, args.threshold))
    if not frames:
        raise FileNotFoundError(f"No completed BERT-family runs in {args.root}")
    samples = pd.concat(frames, ignore_index=True)
    samples.to_csv(args.output_dir / "sample_metrics.csv", index=False)
    summary = bootstrap_summary(samples, args.bootstrap, args.seed)
    summary.to_csv(args.output_dir / "bootstrap_summary.csv", index=False)
    contrasts = paired_contrasts(samples, args.bootstrap, args.seed)
    contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
    architecture = (
        samples[
            [
                "tag", "model", "parameters", "depth", "heads", "head_dim",
                "hidden_size", "layer", "mlm_loss", "pseudo_perplexity",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    architecture.to_csv(args.output_dir / "architecture_and_loss.csv", index=False)
    print(architecture.to_string(index=False))
    print(summary.pivot(index="tag", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    main()
