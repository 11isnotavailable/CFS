"""Audit whether Qwen CFS has comparable semantics across model sizes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_qwen_cfs import load_shards, minimum_cover


def expected_best_k(values: np.ndarray, k: int) -> float:
    values = np.sort(values[np.isfinite(values)])
    n = len(values)
    if not 1 <= k <= n:
        return float("nan")
    denominator = math.comb(n, k)
    return float(sum(
        math.comb(rank, k - 1) * values[rank]
        for rank in range(k - 1, n)
    ) / denominator)


def load_run(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return metadata, load_shards(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture = {
        "Qwen/Qwen2.5-0.5B": (896, 24, 14, 2, 64, 4864),
        "Qwen/Qwen2.5-1.5B": (1536, 28, 12, 2, 128, 8960),
        "Qwen/Qwen2.5-3B": (2048, 36, 16, 2, 128, 11008),
        "Qwen/Qwen2.5-7B": (3584, 28, 28, 4, 128, 18944),
    }
    architecture_rows, audit_rows = [], []
    schedule_hash = None
    for model_order, directory in enumerate(args.input_dirs):
        metadata, data = load_run(directory)
        model = metadata["model"]
        hidden, depth, heads, kv_heads, head_dim, intermediate = architecture[model]
        architecture_rows.append({
            "model_order": model_order, "model": model, "hidden_size": hidden,
            "layers": depth, "query_heads": heads, "kv_heads": kv_heads,
            "queries_per_kv": heads // kv_heads, "head_dim": head_dim,
            "intermediate_size": intermediate,
            "sampled_layers": ",".join(map(str, metadata["layers"])),
            "depth_fractions": ",".join(
                f"{(layer + 1) / depth:.2f}" for layer in metadata["layers"]
            ),
        })
        current_hash = metadata["prediction_position_schedule_sha256"]
        if schedule_hash is None:
            schedule_hash = current_hash
        elif current_hash != schedule_hash:
            raise ValueError(f"Position schedule mismatch: {directory}")
        drops = data["single_drop"].astype(np.float64)
        substitution = data["substitution"].astype(np.float64)
        best_alpha = data["best_alpha"].astype(np.float64)
        scores = 1 - substitution / np.maximum(drops[..., None], 1e-12)
        diagonal = np.arange(heads)
        scores[..., diagonal, diagonal] = np.nan
        group_size = heads // kv_heads
        for sample in range(len(drops)):
            for layer_offset, layer in enumerate(metadata["layers"]):
                local = scores[sample, layer_offset]
                choices = np.nanargmax(local, axis=-1)
                raw = np.nanmax(local, axis=-1)
                matched = {
                    k: np.mean([expected_best_k(row, k) for row in local])
                    for k in (1, 2, 4)
                }
                within_raw, cross_raw, within_m2, cross_m2 = [], [], [], []
                for source, row in enumerate(local):
                    source_group = source // group_size
                    same = np.asarray([
                        index != source and index // group_size == source_group
                        for index in range(heads)
                    ])
                    cross = np.asarray([
                        index // group_size != source_group for index in range(heads)
                    ])
                    within_raw.append(np.nanmax(row[same]))
                    cross_raw.append(np.nanmax(row[cross]))
                    within_m2.append(expected_best_k(row[same], 2))
                    cross_m2.append(expected_best_k(row[cross], 2))
                chosen_alpha = best_alpha[sample, layer_offset, diagonal, choices]
                cover = minimum_cover(local, np.ones(heads, dtype=bool), 0.5)
                audit_rows.append({
                    "model_order": model_order, "model": model, "sample": sample,
                    "layer": layer, "depth_fraction": (layer + 1) / depth,
                    "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
                    "raw_oracle": float(raw.mean()),
                    "matched_1": float(matched[1]), "matched_2": float(matched[2]),
                    "matched_4": float(matched[4]),
                    "within_kv_raw": float(np.mean(within_raw)),
                    "cross_kv_raw": float(np.mean(cross_raw)),
                    "within_kv_matched_2": float(np.mean(within_m2)),
                    "cross_kv_matched_2": float(np.mean(cross_m2)),
                    "best_rep_within_kv_rate": float(np.mean(
                        choices // group_size == diagonal // group_size
                    )),
                    "oracle_alpha_mean": float(np.mean(chosen_alpha)),
                    "oracle_alpha_at_min_rate": float(np.mean(chosen_alpha == 0)),
                    "oracle_alpha_at_max_rate": float(np.mean(chosen_alpha == 3)),
                    "cover": cover, "cover_per_head": cover / heads,
                    "covered_width": cover * head_dim,
                    "covered_width_fraction": cover * head_dim / hidden,
                    "redundancy_ratio": 1 - cover / heads,
                    "mean_drop_kl": float(drops[sample, layer_offset].mean()),
                })
    architecture_frame = pd.DataFrame(architecture_rows)
    audit = pd.DataFrame(audit_rows)
    summary = audit.groupby(
        ["model_order", "model", "heads", "kv_heads", "head_dim"], as_index=False
    ).mean(numeric_only=True)
    architecture_frame.to_csv(args.output_dir / "architecture.csv", index=False)
    audit.to_csv(args.output_dir / "sample_layer_audit.csv", index=False)
    summary.to_csv(args.output_dir / "audit_summary.csv", index=False)
    print("ARCHITECTURE\n", architecture_frame.to_string(index=False))
    columns = [
        "model", "head_dim", "raw_oracle", "matched_1", "matched_2", "matched_4",
        "within_kv_matched_2", "cross_kv_matched_2", "best_rep_within_kv_rate",
        "oracle_alpha_mean", "oracle_alpha_at_min_rate", "oracle_alpha_at_max_rate",
        "cover", "cover_per_head", "covered_width", "redundancy_ratio", "mean_drop_kl",
    ]
    print("\nAUDIT SUMMARY\n", summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
