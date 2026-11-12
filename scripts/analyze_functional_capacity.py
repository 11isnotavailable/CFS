"""Graph-theoretic functional-capacity proxies from frozen CFS oracles.

The minimum representative cover is exact for the measured directed graph, but
it is not an executable joint-compression guarantee: CFS edges are measured by
one-source interventions and may interact when several sources are replaced at
once. Spectral effective rank is a descriptive topology statistic, not a
literal count of semantic functions.
"""

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
import torch

from scripts.train_representative_predictor import load_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        action="append",
        nargs=3,
        metavar=("NAME", "ORACLE_DIR", "CHECKPOINT"),
        required=True,
        help="Repeat once per model. CHECKPOINT may be '-' to omit params.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument(
        "--thresholds", default="0.25,0.5,0.75",
        help="Comma-separated CFS thresholds for exact graph cover.",
    )
    return parser.parse_args()


def parameter_count(path: str) -> int | None:
    if path == "-":
        return None
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    return int(sum(value.numel() for value in state.values()))


def subset_table(heads: int) -> tuple[np.ndarray, np.ndarray]:
    masks = np.arange(1 << heads, dtype=np.uint32)
    selections = (
        (masks[:, None] >> np.arange(heads, dtype=np.uint32)[None, :]) & 1
    ).astype(bool)
    cardinality = selections.sum(axis=1).astype(np.int16)
    return selections, cardinality


def exact_minimum_cover(
    scores: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    selections: np.ndarray,
    cardinality: np.ndarray,
) -> np.ndarray:
    """Minimum retained representatives covering every valid source.

    A retained source covers itself. A retained representative j covers source
    i when S[i,j] exceeds the threshold. Sources below the deletion-KL filter
    impose no coverage requirement because they are already removable under the
    study's validity convention.
    """
    samples, heads, _ = scores.shape
    output = np.zeros(samples, dtype=np.int16)
    representative_bits = np.arange(heads, dtype=np.uint32)
    for sample in range(samples):
        target = np.uint32(0)
        cover = np.zeros(heads, dtype=np.uint32)
        for source in range(heads):
            if not valid[sample, source]:
                continue
            source_bit = np.uint32(1 << source)
            target |= source_bit
            cover[source] |= source_bit
            candidates = np.flatnonzero(scores[sample, source] > threshold)
            for representative in candidates:
                if representative != source:
                    cover[representative] |= source_bit
        if target == 0:
            output[sample] = 0
            continue
        selected_cover = np.where(
            selections,
            cover[representative_bits][None, :],
            np.uint32(0),
        )
        unions = np.bitwise_or.reduce(selected_cover, axis=1)
        feasible = (unions & target) == target
        output[sample] = int(cardinality[feasible].min())
    return output


def spectral_effective_rank(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Entropy effective rank of a directed CFS affinity matrix.

    Valid heads receive unit self-affinity; positive directed recovery is clipped
    to [0,1]. Rows for deletion-filtered sources are zero. Singular values make
    the statistic applicable without pretending that CFS is symmetric.
    """
    samples, heads, _ = scores.shape
    values = np.zeros(samples, dtype=np.float64)
    for sample in range(samples):
        affinity = np.clip(np.nan_to_num(scores[sample], nan=0.0), 0.0, 1.0)
        affinity[~valid[sample]] = 0.0
        diagonal = np.arange(heads)
        affinity[diagonal, diagonal] = valid[sample].astype(np.float64)
        singular = np.linalg.svd(affinity, compute_uv=False)
        total = singular.sum()
        if total <= 0:
            values[sample] = 0.0
            continue
        probability = singular[singular > 0] / total
        values[sample] = math.exp(float(-(probability * np.log(probability)).sum()))
    return values


def analyze_entry(
    name: str,
    directory: Path,
    checkpoint: str,
    thresholds: list[float],
    minimum_drop: float,
) -> tuple[list[dict], list[dict]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    archive = load_shards(directory)
    substitutions = archive["substitution"].astype(np.float64)
    drops = archive["single_drop"].astype(np.float64)
    selected_layers = metadata["selected_layers"]
    total_layers = int(metadata["layers"])
    heads = int(metadata["heads"])
    params = parameter_count(checkpoint)
    selections, cardinality = subset_table(heads)
    summary_rows: list[dict] = []
    sample_rows: list[dict] = []

    for layer_index, layer in enumerate(selected_layers):
        layer_drop = drops[:, layer_index]
        scores = 1.0 - substitutions[:, layer_index] / np.maximum(
            layer_drop[:, :, None], 1e-12
        )
        diagonal = np.arange(heads)
        scores[:, diagonal, diagonal] = np.nan
        valid = layer_drop > minimum_drop
        oracle = np.nanmax(scores, axis=-1)
        effective_rank = spectral_effective_rank(scores, valid)
        for threshold in thresholds:
            cover = exact_minimum_cover(
                scores, valid, threshold, selections, cardinality
            )
            valid_counts = valid.sum(axis=1)
            cover_to_valid = np.divide(
                cover,
                valid_counts,
                out=np.zeros_like(cover, dtype=np.float64),
                where=valid_counts > 0,
            )
            spectral_to_valid = np.divide(
                effective_rank,
                valid_counts,
                out=np.zeros_like(effective_rank, dtype=np.float64),
                where=valid_counts > 0,
            )
            per_sample_oracle = np.divide(
                np.where(valid, oracle, 0.0).sum(axis=1),
                valid_counts,
                out=np.full(len(scores), np.nan, dtype=np.float64),
                where=valid_counts > 0,
            )
            for sample in range(len(scores)):
                sample_rows.append(
                    {
                        "model_size": name,
                        "layer": int(layer),
                        "depth_fraction": (
                            float(layer / (total_layers - 1))
                            if total_layers > 1
                            else 0.0
                        ),
                        "sample": sample,
                        "threshold": threshold,
                        "valid_heads": int(valid[sample].sum()),
                        "valid_head_ratio": float(valid[sample].mean()),
                        "oracle_recovery": float(per_sample_oracle[sample]),
                        "minimum_cover": int(cover[sample]),
                        "cover_ratio": float(cover[sample] / heads),
                        "cover_to_valid_ratio": float(cover_to_valid[sample]),
                        "spectral_effective_rank": float(effective_rank[sample]),
                        "spectral_rank_ratio": float(effective_rank[sample] / heads),
                        "spectral_to_valid_ratio": float(spectral_to_valid[sample]),
                    }
                )
            summary_rows.append(
                {
                    "model_size": name,
                    "model": metadata["model"],
                    "parameters": params,
                    "heads": heads,
                    "layer": int(layer),
                    "depth_fraction": (
                        float(layer / (total_layers - 1))
                        if total_layers > 1
                        else 0.0
                    ),
                    "samples": len(scores),
                    "threshold": threshold,
                    "valid_head_ratio": float(valid.mean()),
                    "oracle_recovery": float(oracle[valid].mean()),
                    "minimum_cover_mean": float(cover.mean()),
                    "minimum_cover_std": float(cover.std(ddof=1)),
                    "cover_ratio_mean": float((cover / heads).mean()),
                    "cover_to_valid_ratio_mean": float(cover_to_valid.mean()),
                    "spectral_effective_rank_mean": float(effective_rank.mean()),
                    "spectral_rank_ratio_mean": float((effective_rank / heads).mean()),
                    "spectral_to_valid_ratio_mean": float(spectral_to_valid.mean()),
                    "capacity_per_million_parameters": (
                        float(cover.mean() / (params / 1e6))
                        if params is not None
                        else np.nan
                    ),
                }
            )
    return summary_rows, sample_rows


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    samples: list[dict] = []
    for name, directory, checkpoint in args.entry:
        rows, detail = analyze_entry(
            name,
            Path(directory),
            checkpoint,
            thresholds,
            args.minimum_drop_kl,
        )
        summaries.extend(rows)
        samples.extend(detail)
    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(samples)
    summary.to_csv(args.output_dir / "functional_capacity_summary.csv", index=False)
    detail.to_csv(args.output_dir / "functional_capacity_samples.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
