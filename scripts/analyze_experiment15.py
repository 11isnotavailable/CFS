"""Analyze matched-policy controls from saved Experiment 1.5 oracle curves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.375)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--graph-threshold", type=float, default=0.2)
    return parser.parse_args()


def valid_representatives(heads: int, source: int) -> np.ndarray:
    return np.asarray([head for head in range(heads) if head != source])


def choose_train_policies(
    curves: np.ndarray,
    drops: np.ndarray,
    samples: np.ndarray,
    layers: np.ndarray,
    train_samples: set[int],
    valid: np.ndarray,
) -> tuple[dict, dict, dict]:
    """Fit fixed representatives and/or alphas using training images only."""
    _, heads, _, alpha_count = curves.shape
    fixed_joint = {}
    fixed_rep = {}
    fixed_alpha_per_pair = {}

    for layer in np.unique(layers):
        unit_selector = np.asarray(
            [
                int(sample) in train_samples and int(unit_layer) == int(layer)
                for sample, unit_layer in zip(samples, layers)
            ]
        )
        for source in range(heads):
            usable = unit_selector & valid[:, source]
            if not usable.any():
                continue
            representatives = valid_representatives(heads, source)
            ratio_cube = (
                curves[usable, source, :, :]
                / drops[usable, source, None, None]
            )
            restricted_cube = ratio_cube[:, representatives, :]
            restricted = np.mean(restricted_cube, axis=0)
            flat = int(np.nanargmin(restricted))
            rep_offset, alpha_index = np.unravel_index(flat, restricted.shape)
            fixed_joint[(int(layer), source)] = (
                int(representatives[rep_offset]),
                int(alpha_index),
            )

            per_unit_best_alpha = np.min(restricted_cube, axis=-1)
            mean_conditional_alpha = np.mean(per_unit_best_alpha, axis=0)
            fixed_rep[(int(layer), source)] = int(
                representatives[np.argmin(mean_conditional_alpha)]
            )

            for representative_offset, representative in enumerate(representatives):
                fixed_alpha_per_pair[(int(layer), source, int(representative))] = (
                    int(np.argmin(restricted[representative_offset]))
                )
    return fixed_joint, fixed_rep, fixed_alpha_per_pair


def evaluate_policies(
    curves: np.ndarray,
    drops: np.ndarray,
    current_l2: np.ndarray,
    samples: np.ndarray,
    layers: np.ndarray,
    test_samples: set[int],
    valid: np.ndarray,
    fixed_joint: dict,
    fixed_rep: dict,
    fixed_alpha_per_pair: dict,
) -> pd.DataFrame:
    _, heads, _, _ = curves.shape
    rows = []
    for unit, (sample, layer) in enumerate(zip(samples, layers)):
        if int(sample) not in test_samples:
            continue
        for source in range(heads):
            if not valid[unit, source]:
                continue
            representatives = valid_representatives(heads, source)
            drop = float(drops[unit, source])
            pair_curves = curves[unit, source, representatives, :]

            conditional_alpha_errors = np.nanmin(pair_curves, axis=-1)
            full_oracle = float(np.nanmin(conditional_alpha_errors))
            random_expected = float(np.nanmean(conditional_alpha_errors))

            similarity_rep = int(
                representatives[
                    np.argmin(current_l2[unit, source, representatives])
                ]
            )
            similarity_error = float(
                np.nanmin(curves[unit, source, similarity_rep, :])
            )

            joint_rep, joint_alpha = fixed_joint[(int(layer), source)]
            fixed_joint_error = float(
                curves[unit, source, joint_rep, joint_alpha]
            )

            trained_rep = fixed_rep[(int(layer), source)]
            fixed_rep_cond_alpha_error = float(
                np.nanmin(curves[unit, source, trained_rep, :])
            )

            conditional_rep_fixed_alpha_errors = []
            for representative in representatives:
                alpha_index = fixed_alpha_per_pair[
                    (int(layer), source, int(representative))
                ]
                conditional_rep_fixed_alpha_errors.append(
                    curves[unit, source, representative, alpha_index]
                )
            conditional_rep_fixed_alpha_error = float(
                np.nanmin(conditional_rep_fixed_alpha_errors)
            )

            similarity_fixed_alpha = fixed_alpha_per_pair[
                (int(layer), source, similarity_rep)
            ]
            similarity_fixed_alpha_error = float(
                curves[
                    unit,
                    source,
                    similarity_rep,
                    similarity_fixed_alpha,
                ]
            )

            policies = {
                "direct_drop": drop,
                "random_rep_conditional_alpha": random_expected,
                "similarity_rep_fixed_alpha": similarity_fixed_alpha_error,
                "similarity_rep_conditional_alpha": similarity_error,
                "fixed_rep_fixed_alpha": fixed_joint_error,
                "fixed_rep_conditional_alpha": fixed_rep_cond_alpha_error,
                "conditional_rep_fixed_alpha": conditional_rep_fixed_alpha_error,
                "conditional_rep_conditional_alpha": full_oracle,
            }
            for policy, error in policies.items():
                rows.append(
                    {
                        "sample": int(sample),
                        "layer": int(layer),
                        "source": source,
                        "policy": policy,
                        "drop_kl": drop,
                        "error_kl": error,
                        "recovery_score": 1.0 - error / drop,
                    }
                )
    return pd.DataFrame(rows)


def compute_score_matrices(
    substitution: np.ndarray,
    drops: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    scores = 1.0 - substitution / np.maximum(drops[:, :, None], 1e-12)
    heads = scores.shape[1]
    scores[:, np.arange(heads), np.arange(heads)] = np.nan
    scores[~valid] = np.nan
    return scores


def asymmetry_statistics(scores: np.ndarray) -> dict[str, float]:
    absolute = []
    one_way_10 = []
    one_way_25 = []
    for matrix in scores:
        heads = matrix.shape[0]
        for left in range(heads):
            for right in range(left + 1, heads):
                forward, backward = matrix[left, right], matrix[right, left]
                if not np.isfinite(forward) or not np.isfinite(backward):
                    continue
                absolute.append(abs(forward - backward))
                one_way_10.append(
                    (forward > 0.1 and backward <= 0.0)
                    or (backward > 0.1 and forward <= 0.0)
                )
                one_way_25.append(
                    (forward > 0.25 and backward <= 0.0)
                    or (backward > 0.25 and forward <= 0.0)
                )
    values = np.asarray(absolute)
    return {
        "pairs": int(len(values)),
        "median_absolute_asymmetry": float(np.median(values)),
        "q75_absolute_asymmetry": float(np.quantile(values, 0.75)),
        "one_way_fraction_threshold_0.10": float(np.mean(one_way_10)),
        "one_way_fraction_threshold_0.25": float(np.mean(one_way_25)),
    }


def draw_graph(
    scores: np.ndarray,
    samples: np.ndarray,
    layers: np.ndarray,
    output: Path,
    threshold: float,
) -> dict[str, int]:
    positive = np.where(np.isfinite(scores), np.maximum(scores, 0.0), 0.0)
    unit_strength = positive.sum(axis=(1, 2))
    edge_counts = ((scores >= threshold) & np.isfinite(scores)).sum(axis=(1, 2))
    # Prefer an illustrative graph with several strong edges but avoid a
    # nearly complete graph whose topology is unreadable.
    eligible = np.where((edge_counts >= 4) & (edge_counts <= 10))[0]
    unit = (
        int(eligible[np.argmax(unit_strength[eligible])])
        if len(eligible)
        else int(np.argmax(unit_strength))
    )
    matrix = scores[unit]
    heads = matrix.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(heads))
    for source in range(heads):
        for representative in range(heads):
            score = matrix[source, representative]
            if source != representative and np.isfinite(score) and score >= threshold:
                graph.add_edge(source, representative, weight=float(score))

    positions = nx.circular_layout(graph)
    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="#d9ecff",
        edgecolors="#28536b",
        node_size=1200,
    )
    nx.draw_networkx_labels(graph, positions, font_size=12)
    edges = list(graph.edges(data=True))
    if edges:
        widths = [1.0 + 4.0 * edge[2]["weight"] for edge in edges]
        colors = [edge[2]["weight"] for edge in edges]
        nx.draw_networkx_edges(
            graph,
            positions,
            width=widths,
            edge_color=colors,
            edge_cmap=plt.cm.viridis,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.08",
        )
    plt.title(
        f"Conditional substitution graph · sample {samples[unit]} "
        f"· layer {layers[unit]} · S≥{threshold}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return {
        "sample": int(samples[unit]),
        "layer": int(layers[unit]),
        "edges": int(graph.number_of_edges()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.experiment_dir / "matrix_records.npz")
    curves = archive["substitution_curves"].astype(np.float64)
    substitution = archive["substitution"].astype(np.float64)
    drops = archive["single_drop"].astype(np.float64)
    current_l2 = archive["current_l2"].astype(np.float64)
    samples = archive["sample"].astype(int)
    layers = archive["layer"].astype(int)

    sample_frame = pd.read_csv(args.experiment_dir / "sample_records.csv")
    unique_samples = sample_frame["sample"].to_numpy()
    train, test = train_test_split(
        unique_samples,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=sample_frame["dataset_label"],
    )
    train_samples, test_samples = set(train.tolist()), set(test.tolist())

    # Mask source heads whose direct-drop effect is too close to numerical noise.
    valid = drops > args.minimum_drop_kl
    fixed_joint, fixed_rep, fixed_alpha_per_pair = choose_train_policies(
        curves,
        drops,
        samples,
        layers,
        train_samples,
        valid,
    )
    policy_frame = evaluate_policies(
        curves,
        drops,
        current_l2,
        samples,
        layers,
        test_samples,
        valid,
        fixed_joint,
        fixed_rep,
        fixed_alpha_per_pair,
    )
    policy_frame.to_csv(args.output_dir / "policy_decomposition.csv", index=False)

    summary = (
        policy_frame.groupby("policy")
        .agg(
            units=("recovery_score", "size"),
            mean_recovery=("recovery_score", "mean"),
            median_recovery=("recovery_score", "median"),
            mean_kl=("error_kl", "mean"),
            median_kl=("error_kl", "median"),
        )
        .sort_values("mean_recovery")
    )
    summary.to_csv(args.output_dir / "policy_summary.csv")

    plt.figure(figsize=(10, 5.8))
    order = summary.index.tolist()
    sns.barplot(
        data=policy_frame,
        y="policy",
        x="recovery_score",
        order=order,
        errorbar=("ci", 95),
    )
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel(r"Recovery score $S=1-D_{sub}/D_{drop}$")
    plt.ylabel("")
    plt.title("Matched-policy decomposition on held-out images")
    plt.tight_layout()
    plt.savefig(args.output_dir / "policy_decomposition.png", dpi=180)
    plt.close()

    scores = compute_score_matrices(substitution, drops, valid)
    np.savez_compressed(
        args.output_dir / "substitution_scores.npz",
        scores=scores,
        valid=valid,
        sample=samples,
        layer=layers,
    )
    asymmetry = asymmetry_statistics(scores)
    graph = draw_graph(
        scores,
        samples,
        layers,
        args.output_dir / "directed_substitution_graph.png",
        args.graph_threshold,
    )

    alpha = pd.read_csv(args.experiment_dir / "pair_records.csv")
    plt.figure(figsize=(9, 5.2))
    sns.boxplot(data=alpha, x="layer", y="best_alpha")
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1)
    plt.title("Per-input oracle alpha by layer")
    plt.tight_layout()
    plt.savefig(args.output_dir / "alpha_by_layer.png", dpi=180)
    plt.close()

    metadata = {
        "experiment_dir": str(args.experiment_dir.resolve()),
        "train_samples": sorted(train_samples),
        "test_samples": sorted(test_samples),
        "minimum_drop_kl": args.minimum_drop_kl,
        "valid_source_fraction": float(valid.mean()),
        "asymmetry": asymmetry,
        "graph": graph,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(json.dumps(metadata, indent=2))
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
