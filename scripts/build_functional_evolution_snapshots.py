"""Materialize per-epoch functional snapshots from full CFS raw shards."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_cfs_developmental_oracles import expected_best_of_k
from analyze_functional_capacity import (
    exact_minimum_cover,
    spectral_effective_rank,
    subset_table,
)


def load_complete_shards(directory: Path) -> dict[str, np.ndarray]:
    files = sorted(directory.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"No shards found in {directory}")
    pieces: dict[str, list[np.ndarray]] = {}
    for file in files:
        with np.load(file) as archive:
            for key in archive.files:
                if key in {"alpha_grid"}:
                    continue
                pieces.setdefault(key, []).append(archive[key])
    return {
        key: np.concatenate(values, axis=0) for key, values in pieces.items()
    }


def safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-json", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--thresholds", default="0.25,0.5,0.75")
    args = parser.parse_args()

    thresholds = [float(value) for value in args.thresholds.split(",")]
    training = json.loads(args.training_json.read_text(encoding="utf-8"))
    history = {int(row["epoch"]): row for row in training["history"]}
    trajectory_rows: list[dict] = []

    for epoch_dir in sorted(args.root.glob("epoch_*")):
        epoch = int(epoch_dir.name.split("_")[-1])
        raw = epoch_dir / "raw"
        metadata = json.loads((raw / "metadata.json").read_text(encoding="utf-8"))
        archive = load_complete_shards(raw)
        substitutions = archive["substitution"].astype(np.float64)
        drops = archive["single_drop"].astype(np.float64)
        best_alpha = archive["best_alpha"].astype(np.float64)
        energies = archive["contribution_norm"].astype(np.float64)
        alpha_cube = archive["alpha_error_cube"].astype(np.float64)
        layers = metadata["selected_layers"]
        heads = int(metadata["heads"])
        selections, cardinality = subset_table(heads)
        layer_metrics = []

        for position, layer in enumerate(layers):
            layer_dir = epoch_dir / f"layer{int(layer):02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            replacement = substitutions[:, position]
            drop = drops[:, position]
            alpha = best_alpha[:, position]
            energy = energies[:, position]
            scores = 1.0 - replacement / np.maximum(drop[:, :, None], 1e-12)
            diagonal = np.arange(heads)
            scores[:, diagonal, diagonal] = np.nan
            valid = drop > args.minimum_drop_kl
            masked_scores = np.where(valid[:, :, None], scores, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_scores = np.nanmean(masked_scores, axis=0)
                mean_alpha_cube = np.nanmean(alpha_cube[:, position], axis=0)
            mean_replacement = np.nanmean(replacement, axis=0)
            mean_drop = np.nanmean(drop, axis=0)
            mean_alpha = np.nanmean(alpha, axis=0)
            mean_energy = np.nanmean(energy, axis=0)
            best_rep = np.nanargmax(mean_scores, axis=-1).astype(np.int16)

            np.save(layer_dir / "S.npy", mean_scores.astype(np.float32))
            np.save(layer_dir / "replace_kl.npy", mean_replacement.astype(np.float32))
            np.save(layer_dir / "drop_kl.npy", mean_drop.astype(np.float32))
            np.save(layer_dir / "alpha.npy", mean_alpha.astype(np.float32))
            np.save(layer_dir / "best_rep.npy", best_rep)
            np.save(layer_dir / "head_energy.npy", mean_energy.astype(np.float32))
            np.save(
                layer_dir / "alpha_error_cube_mean.npy",
                mean_alpha_cube.astype(np.float32),
            )
            np.save(layer_dir / "sample_S.npy", scores.astype(np.float32))
            np.save(layer_dir / "sample_replace_kl.npy", replacement.astype(np.float32))
            np.save(layer_dir / "sample_drop.npy", drop.astype(np.float32))
            np.save(layer_dir / "sample_alpha.npy", alpha.astype(np.float32))
            np.save(layer_dir / "sample_head_energy.npy", energy.astype(np.float32))
            for threshold in thresholds:
                suffix = f"{int(round(threshold * 100)):03d}"
                np.save(
                    layer_dir / f"adj_tau{suffix}.npy",
                    (mean_scores > threshold).astype(np.uint8),
                )

            oracle = np.nanmax(scores, axis=-1)
            best_substitution = np.nanmin(
                np.where(np.eye(heads, dtype=bool)[None], np.nan, replacement),
                axis=-1,
            )
            candidate_rows = scores[valid]
            candidates = candidate_rows[np.isfinite(candidate_rows)].reshape(
                int(valid.sum()), heads - 1
            )
            rank = spectral_effective_rank(scores, valid)
            recovered = drop - best_substitution
            weighted = np.divide(
                recovered.sum(axis=1), drop.sum(axis=1),
                out=np.full(len(drop), np.nan), where=drop.sum(axis=1) > 0,
            )
            metrics = {
                "layer": int(layer),
                "valid_source_ratio": float(valid.mean()),
                "raw_oracle": float(oracle[valid].mean()),
                "matched_1": float(expected_best_of_k(candidates, 1).mean()),
                "matched_2": float(expected_best_of_k(candidates, 2).mean()),
                "matched_4": float(expected_best_of_k(candidates, 4).mean()),
                "weighted_recovery": safe_mean(weighted),
                "mean_deletion_kl": float(drop[valid].mean()),
                "mean_best_substitution_kl": float(best_substitution[valid].mean()),
                "spectral_rank": float(rank.mean()),
                "rank_per_head": float((rank / heads).mean()),
            }
            for threshold in thresholds:
                cover = exact_minimum_cover(
                    scores, valid, threshold, selections, cardinality
                )
                key = f"tau_{threshold:.2f}"
                metrics[f"cover_{key}"] = float(cover.mean())
                metrics[f"cover_per_head_{key}"] = float((cover / heads).mean())
            layer_metrics.append(metrics)
            trajectory_rows.append({"epoch": epoch, **metrics})

        train_row = history[epoch]
        snapshot = {
            "epoch": epoch,
            "accuracy": float(train_row["tune_accuracy"]),
            "loss": float(train_row["tune_loss"]),
            "minimum_drop_kl": args.minimum_drop_kl,
            "thresholds": thresholds,
            "probe_samples": int(len(substitutions)),
            "layers": layer_metrics,
        }
        (epoch_dir / "metrics.json").write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8"
        )

    frame = pd.DataFrame(trajectory_rows)
    frame.to_csv(args.root / "functional_trajectory_by_layer.csv", index=False)
    macro = frame.groupby("epoch", as_index=False).mean(numeric_only=True)
    macro.to_csv(args.root / "functional_trajectory_macro.csv", index=False)


if __name__ == "__main__":
    main()
