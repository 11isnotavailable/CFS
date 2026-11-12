"""Complete the cheap Section 4 CFS validation summaries.

This script uses the frozen Experiment 1.5 oracle tensors.  It adds
parameter-space cosine, exposes the already-collected contribution cosine,
and summarizes how often a fixed source keeps the same best substitute across
inputs.  No model forward pass is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.model_loading import create_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    return parser.parse_args()


def parameter_cosine_matrices(model: torch.nn.Module) -> np.ndarray:
    matrices = []
    for block in model.blocks:
        attention = block.attn
        heads = int(attention.num_heads)
        dim = int(attention.qkv.in_features)
        head_dim = int(attention.head_dim)
        qkv = attention.qkv.weight.detach().float().reshape(3, heads, head_dim, dim)
        # torch Linear applies x @ W.T, so the columns of proj.weight are the
        # head-specific input slices contributing through W_O.
        wo = attention.proj.weight.detach().float().reshape(dim, heads, head_dim)
        vectors = []
        for head in range(heads):
            vectors.append(
                torch.cat(
                    [qkv[0, head].flatten(), qkv[1, head].flatten(),
                     qkv[2, head].flatten(), wo[:, head].flatten()]
                )
            )
        vectors = torch.nn.functional.normalize(torch.stack(vectors), dim=1)
        matrices.append((vectors @ vectors.T).cpu().numpy())
    return np.stack(matrices)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    # Preserve undefined correlations as NaN so pandas follows the original
    # Experiment 1.5 aggregation protocol (undefined constant rows are omitted).
    return float(value) if np.isfinite(value) else np.nan


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle = np.load(args.experiment_dir / "matrix_records.npz")
    features = np.load(args.analysis_dir / "baseline_features.npz")
    metadata = json.loads(
        (args.analysis_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    test_samples = set(int(value) for value in metadata["test_samples"])

    substitution = oracle["substitution"].astype(np.float64)
    drops = oracle["single_drop"].astype(np.float64)
    current_l2 = oracle["current_l2"].astype(np.float64)
    current_cosine_distance = oracle["current_cosine"].astype(np.float64)
    samples = oracle["sample"].astype(int)
    layers = oracle["layer"].astype(int)
    heads = substitution.shape[-1]
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    scores[drops <= args.minimum_drop_kl] = np.nan

    model, _ = create_timm_model(args.model, args.checkpoint)
    parameter_cosine = parameter_cosine_matrices(model)
    del model

    attention_similarity = 1.0 - features["attention_map_distance"].reshape(
        -1, heads, heads
    ).astype(np.float64)
    taylor = features["taylor"].reshape(-1, heads).astype(np.float64)
    contribution_norm = features["contribution_norm"].reshape(-1, heads).astype(
        np.float64
    )

    rank_rows = []
    for unit, (sample, layer) in enumerate(zip(samples, layers)):
        if int(sample) not in test_samples:
            continue
        for source in range(heads):
            if drops[unit, source] <= args.minimum_drop_kl:
                continue
            reps = np.asarray([head for head in range(heads) if head != source])
            target = scores[unit, source, reps]
            feature_values = {
                "parameter_cosine": parameter_cosine[layer, source, reps],
                "attention_map_similarity": attention_similarity[unit, source, reps],
                "contribution_cosine": 1.0
                - current_cosine_distance[unit, source, reps],
                "negative_contribution_l2": -current_l2[unit, source, reps],
                "contribution_norm": contribution_norm[unit, reps],
                "taylor_importance": taylor[unit, reps],
            }
            for feature, values in feature_values.items():
                rank_rows.append(
                    {
                        "sample": int(sample),
                        "layer": int(layer),
                        "source": int(source),
                        "feature": feature,
                        "spearman_with_cfs": safe_spearman(values, target),
                    }
                )
    rank_frame = pd.DataFrame(rank_rows)
    rank_summary = (
        rank_frame.groupby("feature")["spearman_with_cfs"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    rank_frame.to_csv(args.output_dir / "similarity_vs_cfs_records.csv", index=False)
    rank_summary.to_csv(args.output_dir / "similarity_vs_cfs_summary.csv", index=False)

    # Direct input-conditioning statistic over all 32 inputs.  For each fixed
    # (layer, source), measure how often its modal best substitute is selected.
    modal_rows = []
    for layer in sorted(set(layers.tolist())):
        layer_units = np.where(layers == layer)[0]
        for source in range(heads):
            best = []
            for unit in layer_units:
                row = scores[unit, source]
                if np.isfinite(row).any():
                    best.append(int(np.nanargmax(row)))
            counts = np.bincount(best, minlength=heads)
            modal = int(np.argmax(counts))
            coverage = float(counts[modal] / len(best))
            modal_rows.append(
                {
                    "layer": int(layer),
                    "source": int(source),
                    "inputs": int(len(best)),
                    "modal_substitute": modal,
                    "modal_count": int(counts[modal]),
                    "modal_coverage": coverage,
                    "switch_rate": 1.0 - coverage,
                    "distinct_best_substitutes": int(np.count_nonzero(counts)),
                }
            )
    modal_frame = pd.DataFrame(modal_rows)
    modal_frame.to_csv(args.output_dir / "input_conditioning_by_source.csv", index=False)
    layer_summary = (
        modal_frame.groupby("layer")
        .agg(
            sources=("source", "size"),
            mean_modal_coverage=("modal_coverage", "mean"),
            median_modal_coverage=("modal_coverage", "median"),
            min_modal_coverage=("modal_coverage", "min"),
            max_modal_coverage=("modal_coverage", "max"),
            mean_switch_rate=("switch_rate", "mean"),
            median_distinct_substitutes=("distinct_best_substitutes", "median"),
        )
        .reset_index()
    )
    layer_summary.to_csv(
        args.output_dir / "input_conditioning_by_layer.csv", index=False
    )
    overall = pd.DataFrame(
        [
            {
                "source_units": len(modal_frame),
                "mean_modal_coverage": modal_frame.modal_coverage.mean(),
                "median_modal_coverage": modal_frame.modal_coverage.median(),
                "q25_modal_coverage": modal_frame.modal_coverage.quantile(0.25),
                "q75_modal_coverage": modal_frame.modal_coverage.quantile(0.75),
                "mean_switch_rate": modal_frame.switch_rate.mean(),
                "median_switch_rate": modal_frame.switch_rate.median(),
                "mean_distinct_substitutes": modal_frame.distinct_best_substitutes.mean(),
                "median_distinct_substitutes": modal_frame.distinct_best_substitutes.median(),
            }
        ]
    )
    overall.to_csv(args.output_dir / "input_conditioning_summary.csv", index=False)

    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_dir": str(args.experiment_dir.resolve()),
                "analysis_dir": str(args.analysis_dir.resolve()),
                "model": args.model,
                "minimum_drop_kl": args.minimum_drop_kl,
                "similarity_protocol": "12 held-out images x 12 layers x 6 sources",
                "input_conditioning_protocol": "all 32 images x 12 layers x 6 sources",
                "forward_passes": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(rank_summary.to_string(index=False))
    print("\nINPUT CONDITIONING")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
