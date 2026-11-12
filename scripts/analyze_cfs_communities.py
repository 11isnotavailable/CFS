"""Analyze input-conditional community structure in frozen CFS graphs.

This script deliberately avoids calling connected components equivalence
classes.  Directed CFS is not necessarily symmetric or transitive.  We report
both weak components of the directed threshold graph and components of a more
conservative mutual-substitution graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.train_representative_predictor import load_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry", action="append", nargs=2, metavar=("NAME", "ORACLE_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-drop-kl", type=float, default=1e-5)
    parser.add_argument("--thresholds", default="0.25,0.5,0.75")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--matched-heads", type=int, default=3,
        help="Size of induced valid-head subgraphs used as a node-count control.",
    )
    parser.add_argument("--matched-repeats", type=int, default=32)
    return parser.parse_args()


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    """Connected components of a symmetric boolean adjacency matrix."""
    nodes = len(adjacency)
    unseen = set(range(nodes))
    components: list[list[int]] = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = [root]
        while stack:
            node = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[node])) & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
            component.extend(neighbors)
        components.append(component)
    return components


def normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probability = counts[counts > 0] / total
    return float(-(probability * np.log(probability)).sum() / math.log(len(counts)))


def graph_metrics(scores: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    indices = np.flatnonzero(valid)
    count = len(indices)
    if count == 0:
        return {key: np.nan for key in METRIC_NAMES}
    local = scores[np.ix_(indices, indices)]
    directed = np.nan_to_num(local, nan=-np.inf) > threshold
    np.fill_diagonal(directed, False)
    mutual = directed & directed.T
    weak = directed | directed.T
    possible = count * (count - 1)
    directed_density = float(directed.sum() / possible) if possible else 0.0
    mutual_density = float(mutual.sum() / possible) if possible else 0.0
    weak_components = connected_components(weak)
    mutual_components = connected_components(mutual)
    weak_sizes = np.asarray([len(value) for value in weak_components])
    mutual_sizes = np.asarray([len(value) for value in mutual_components])
    in_degree = directed.sum(axis=0).astype(np.float64)
    return {
        "directed_edge_density": directed_density,
        "mutual_edge_density": mutual_density,
        "weak_component_count": float(len(weak_components)),
        "weak_component_ratio": float(len(weak_components) / count),
        "weak_largest_ratio": float(weak_sizes.max() / count),
        "mutual_component_count": float(len(mutual_components)),
        "mutual_component_ratio": float(len(mutual_components) / count),
        "mutual_largest_ratio": float(mutual_sizes.max() / count),
        "mutual_isolated_ratio": float((mutual.sum(axis=1) == 0).mean()),
        "in_degree_hub_share": (
            float(in_degree.max() / in_degree.sum()) if in_degree.sum() else 0.0
        ),
        "in_degree_entropy": normalized_entropy(in_degree),
    }


METRIC_NAMES = [
    "directed_edge_density", "mutual_edge_density",
    "weak_component_count", "weak_component_ratio", "weak_largest_ratio",
    "mutual_component_count", "mutual_component_ratio", "mutual_largest_ratio",
    "mutual_isolated_ratio", "in_degree_hub_share", "in_degree_entropy",
]


def load_scores(directory: Path, minimum_drop: float) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    archive = load_shards(directory)
    substitutions = archive["substitution"][:, 0].astype(np.float64)
    drops = archive["single_drop"][:, 0].astype(np.float64)
    scores = 1.0 - substitutions / np.maximum(drops[:, :, None], 1e-12)
    diagonal = np.arange(scores.shape[-1])
    scores[:, diagonal, diagonal] = np.nan
    return scores, drops > minimum_drop, metadata


def bootstrap_interval(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize_samples(frame: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (model, heads, threshold, control), group in frame.groupby(
        ["model_size", "heads", "threshold", "control"], sort=False
    ):
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_interval(values, repetitions, rng)
            rows.append({
                "model_size": model, "heads": heads, "threshold": threshold,
                "control": control, "metric": metric,
                "mean": float(np.nanmean(values)), "ci_low": low, "ci_high": high,
                "samples": int(np.isfinite(values).sum()),
            })
    return pd.DataFrame(rows)


def paired_contrasts(
    frame: pd.DataFrame, repetitions: int, seed: int,
) -> pd.DataFrame:
    """Paired next-minus-previous contrasts using aligned image indices."""
    rng = np.random.default_rng(seed)
    models = list(dict.fromkeys(frame.model_size))
    rows = []
    for previous, current in zip(models[:-1], models[1:]):
        for threshold in sorted(frame.threshold.unique()):
            for control in frame.control.unique():
                left = frame[
                    (frame.model_size == previous) & (frame.threshold == threshold)
                    & (frame.control == control)
                ].set_index("sample")
                right = frame[
                    (frame.model_size == current) & (frame.threshold == threshold)
                    & (frame.control == control)
                ].set_index("sample")
                common = left.index.intersection(right.index)
                if len(common) == 0:
                    continue
                for metric in METRIC_NAMES:
                    difference = (
                        right.loc[common, metric].to_numpy(dtype=np.float64)
                        - left.loc[common, metric].to_numpy(dtype=np.float64)
                    )
                    difference = difference[np.isfinite(difference)]
                    if len(difference) == 0:
                        continue
                    draws = rng.integers(0, len(difference), size=(repetitions, len(difference)))
                    means = difference[draws].mean(axis=1)
                    low, high = np.quantile(means, [0.025, 0.975])
                    rows.append({
                        "contrast": f"{current}-{previous}", "previous": previous,
                        "current": current, "threshold": threshold, "control": control,
                        "metric": metric, "mean_difference": float(difference.mean()),
                        "ci_low": float(low), "ci_high": float(high),
                        "paired_samples": len(difference),
                    })
    return pd.DataFrame(rows)


def choose_representative_sample(frame: pd.DataFrame, model: str, threshold: float) -> int:
    group = frame[(frame.model_size == model) & (frame.threshold == threshold) & (frame.control == "full")]
    targets = group[["directed_edge_density", "mutual_edge_density", "mutual_largest_ratio"]].median()
    scale = group[["directed_edge_density", "mutual_edge_density", "mutual_largest_ratio"]].std().replace(0, 1)
    distance = (((group[targets.index] - targets) / scale) ** 2).sum(axis=1)
    return int(group.loc[distance.idxmin(), "sample"])


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    models = list(dict.fromkeys(summary.model_size))
    colors = {0.25: "#4c78a8", 0.5: "#f58518", 0.75: "#54a24b"}
    metrics = [
        ("directed_edge_density", "Directed edge density"),
        ("mutual_edge_density", "Mutual edge density"),
        ("mutual_component_ratio", "Mutual components / valid heads"),
        ("mutual_largest_ratio", "Largest mutual component / valid heads"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for threshold in sorted(summary.threshold.unique()):
            part = summary[(summary.metric == metric) & (summary.threshold == threshold) & (summary.control == "full")]
            part = part.set_index("model_size").loc[models].reset_index()
            x = np.arange(len(models))
            axis.errorbar(
                x, part["mean"],
                yerr=np.vstack([part["mean"] - part["ci_low"], part["ci_high"] - part["mean"]]),
                marker="o", capsize=2.5, color=colors.get(threshold), label=fr"$\tau={threshold:g}$",
            )
        axis.set_xticks(np.arange(len(models)), models)
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, ncols=3, fontsize=8)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_matched_control(summary: pd.DataFrame, threshold: float, output: Path) -> None:
    models = list(dict.fromkeys(summary.model_size))
    metrics = [
        ("directed_edge_density", "Directed edge density"),
        ("mutual_edge_density", "Mutual edge density"),
        ("mutual_component_ratio", "Mutual components / nodes"),
        ("mutual_largest_ratio", "Largest mutual component / nodes"),
    ]
    controls = [("full", "Full graph", "#4c78a8"), ("matched_h3", "Matched 3-head induced graph", "#e45756")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for control, control_label, color in controls:
            part = summary[(summary.metric == metric) & (summary.threshold == threshold) & (summary.control == control)]
            part = part.set_index("model_size").reindex(models).reset_index()
            axis.errorbar(
                np.arange(len(models)), part["mean"],
                yerr=np.vstack([part["mean"] - part["ci_low"], part["ci_high"] - part["mean"]]),
                marker="o", capsize=2.5, color=color, label=control_label,
            )
        axis.set_xticks(np.arange(len(models)), models)
        axis.set_ylabel(label); axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(fr"Node-count control at $\tau={threshold:g}$", fontsize=11)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_representative_graphs(
    entries: list[tuple[str, np.ndarray, np.ndarray]], samples: dict[str, int],
    threshold: float, output: Path,
) -> None:
    fig, axes = plt.subplots(1, len(entries), figsize=(3.2 * len(entries), 3.25), constrained_layout=True)
    if len(entries) == 1:
        axes = [axes]
    for axis, (model, scores, valid) in zip(axes, entries):
        sample = samples[model]
        matrix = scores[sample]
        heads = len(matrix)
        angles = np.linspace(0, 2 * np.pi, heads, endpoint=False) + np.pi / 2
        xy = np.column_stack([np.cos(angles), np.sin(angles)])
        for source in range(heads):
            if not valid[sample, source]:
                continue
            for target in range(heads):
                if source == target or not valid[sample, target] or not np.isfinite(matrix[source, target]) or matrix[source, target] <= threshold:
                    continue
                start, end = xy[source], xy[target]
                axis.annotate(
                    "", xy=end * 0.88, xytext=start * 0.88,
                    arrowprops=dict(arrowstyle="->", lw=0.5 + 1.5 * min(matrix[source, target], 1), alpha=0.42, color="#4c78a8"),
                )
        axis.scatter(xy[:, 0], xy[:, 1], s=105, c=np.where(valid[sample], "#f58518", "#bdbdbd"), zorder=3, edgecolors="white")
        for head, (x, y) in enumerate(xy):
            axis.text(x, y, str(head), ha="center", va="center", fontsize=7, zorder=4)
        axis.set_title(f"{model} (H={heads})\nrepresentative input {sample}", fontsize=9)
        axis.set_aspect("equal")
        axis.set_xlim(-1.25, 1.25); axis.set_ylim(-1.25, 1.25); axis.axis("off")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(item) for item in args.thresholds.split(",")]
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    graph_data = []
    for model, raw_directory in args.entry:
        scores, valid, metadata = load_scores(Path(raw_directory), args.minimum_drop_kl)
        heads = scores.shape[-1]
        graph_data.append((model, scores, valid))
        for sample in range(len(scores)):
            for threshold in thresholds:
                rows.append({
                    "model_size": model, "heads": heads, "sample": sample,
                    "threshold": threshold, "control": "full",
                    "valid_heads": int(valid[sample].sum()),
                    **graph_metrics(scores[sample], valid[sample], threshold),
                })
                valid_indices = np.flatnonzero(valid[sample])
                if len(valid_indices) >= args.matched_heads:
                    matched_values = []
                    for _ in range(args.matched_repeats):
                        subset = rng.choice(valid_indices, size=args.matched_heads, replace=False)
                        matched_values.append(graph_metrics(
                            scores[sample][np.ix_(subset, subset)],
                            np.ones(args.matched_heads, dtype=bool), threshold,
                        ))
                    averaged = {key: float(np.nanmean([value[key] for value in matched_values])) for key in METRIC_NAMES}
                    rows.append({
                        "model_size": model, "heads": heads, "sample": sample,
                        "threshold": threshold, "control": f"matched_h{args.matched_heads}",
                        "valid_heads": args.matched_heads, **averaged,
                    })
    samples = pd.DataFrame(rows)
    summary = summarize_samples(samples, args.bootstrap_repetitions, args.seed + 1)
    contrasts = paired_contrasts(samples, args.bootstrap_repetitions, args.seed + 2)
    samples.to_csv(args.output_dir / "community_samples.csv", index=False)
    summary.to_csv(args.output_dir / "community_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "community_paired_contrasts.csv", index=False)
    plot_summary(summary, args.output_dir / "community_evolution")
    graph_threshold = min(thresholds, key=lambda value: abs(value - 0.5))
    plot_matched_control(
        summary, graph_threshold, args.output_dir / "community_matched_head_control"
    )
    representative = {
        model: choose_representative_sample(samples, model, graph_threshold)
        for model, _, _ in graph_data
    }
    plot_representative_graphs(
        graph_data, representative, graph_threshold,
        args.output_dir / "community_representative_graphs",
    )
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps({
        "minimum_drop_kl": args.minimum_drop_kl,
        "thresholds": thresholds,
        "matched_heads": args.matched_heads,
        "matched_repeats": args.matched_repeats,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "representative_samples_at_tau": {str(graph_threshold): representative},
        "interpretation_warning": (
            "Components are descriptive structures of thresholded single-source CFS graphs, "
            "not equivalence classes or executable joint-compression groups."
        ),
    }, indent=2), encoding="utf-8")
    display = summary[(summary.threshold == graph_threshold) & (summary.control == "full")]
    display = display[display.metric.isin([
        "directed_edge_density", "mutual_edge_density", "mutual_component_ratio",
        "mutual_largest_ratio", "mutual_isolated_ratio", "in_degree_hub_share",
    ])]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
