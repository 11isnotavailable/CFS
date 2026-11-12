"""Create exploratory visualizations for the ViT-S CFS training trajectory.

The script consumes the compact snapshots produced by
``build_functional_evolution_snapshots.py``.  It intentionally keeps the
plots separate from the paper figures: these are diagnostic views used to
decide which training-dynamics story is supported by the data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


LAYERS = (3, 7, 11)
COLORS = {3: "#2878B5", 7: "#F28E2B", 11: "#59A14F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gif-fps", type=int, default=8)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_training_history(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["history"])[
        ["epoch", "train_loss", "tune_loss", "tune_accuracy", "learning_rate"]
    ]


def load_matrix(data_root: Path, epoch: int, layer: int, name: str) -> np.ndarray:
    return np.load(
        data_root / f"epoch_{epoch:03d}" / f"layer{layer:02d}" / f"{name}.npy"
    )


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.nanstd(left[valid]) == 0 or np.nanstd(right[valid]) == 0:
        return np.nan
    return float(spearmanr(left[valid], right[valid]).statistic)


def row_rank_signature(matrix: np.ndarray, source: int) -> np.ndarray:
    candidates = np.asarray([j for j in range(matrix.shape[0]) if j != source])
    values = matrix[source, candidates]
    return rankdata(values, method="average")


def make_trajectory_plot(
    macro: pd.DataFrame, history: pd.DataFrame, output_dir: Path
) -> None:
    frame = macro.merge(history, on="epoch", validate="one_to_one")
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11), sharex=True)

    ax = axes[0]
    ax.plot(frame.epoch, 100 * frame.tune_accuracy, color="#1f4e79", lw=2.2)
    ax.set_ylabel("Validation accuracy (%)", color="#1f4e79")
    ax.tick_params(axis="y", labelcolor="#1f4e79")
    other = ax.twinx()
    other.plot(frame.epoch, frame.tune_loss, color="#b23a48", lw=1.9)
    other.set_ylabel("Validation loss", color="#b23a48")
    other.tick_params(axis="y", labelcolor="#b23a48")
    ax.set_title("Performance")

    ax = axes[1]
    for column, label, color in [
        ("raw_oracle", "Raw oracle", "#222222"),
        ("matched_2", "Matched-2", "#7A5195"),
        ("weighted_recovery", "Weighted recovery", "#EF5675"),
    ]:
        ax.plot(frame.epoch, frame[column], label=label, lw=2, color=color)
    ax.set_ylabel("Recovery")
    ax.set_title("Functional substitutability")
    ax.legend(frameon=False, ncol=3)

    ax = axes[2]
    ax.plot(
        frame.epoch,
        frame["cover_per_head_tau_0.25"],
        label="Cover/head, $\\tau=.25$",
        color="#4E79A7",
        lw=2,
    )
    ax.plot(
        frame.epoch,
        frame["cover_per_head_tau_0.50"],
        label="Cover/head, $\\tau=.50$",
        color="#76B7B2",
        lw=2,
    )
    ax.plot(
        frame.epoch,
        frame.rank_per_head,
        label="Spectral rank/head",
        color="#E15759",
        lw=2,
    )
    ax.set_ylabel("Normalized structure metric")
    ax.set_xlabel("Epoch")
    ax.set_title("Functional coverage and rank")
    ax.legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.axvspan(1, 5, color="0.7", alpha=0.10)
    fig.suptitle("Performance and Functional Organization During Training", fontsize=15)
    fig.tight_layout()
    save_figure(fig, output_dir, "trajectory_metrics")


def draw_top1_graph(
    ax: plt.Axes,
    score: np.ndarray,
    drop: np.ndarray,
    title: str,
    global_log_drop_range: tuple[float, float],
) -> None:
    heads = score.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(heads))
    for source in range(heads):
        row = score[source].copy()
        row[source] = np.nan
        representative = int(np.nanargmax(row))
        value = float(row[representative])
        if np.isfinite(value) and value > 0:
            graph.add_edge(source, representative, weight=value)

    pos = nx.circular_layout(graph, scale=1.0)
    low, high = global_log_drop_range
    log_drop = np.log10(np.maximum(drop, 1e-12))
    normalized = np.clip((log_drop - low) / max(high - low, 1e-12), 0, 1)
    sizes = 420 + 900 * normalized
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=sizes,
        node_color="#DCEAF7",
        edgecolors="#315B7D",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)
    edges = list(graph.edges(data=True))
    if edges:
        weights = np.asarray([edge[2]["weight"] for edge in edges])
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=edges,
            width=1.0 + 6.0 * np.clip(weights, 0, 1),
            edge_color=weights,
            edge_cmap=plt.cm.viridis,
            edge_vmin=0,
            edge_vmax=max(0.5, float(np.nanmax(weights))),
            arrows=True,
            arrowsize=14,
            connectionstyle="arc3,rad=0.08",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_axis_off()


def graph_drop_range(data_root: Path) -> tuple[float, float]:
    values = []
    for epoch in range(1, 101):
        for layer in LAYERS:
            values.extend(load_matrix(data_root, epoch, layer, "drop_kl").tolist())
    log_values = np.log10(np.maximum(np.asarray(values), 1e-12))
    return float(np.quantile(log_values, 0.02)), float(np.quantile(log_values, 0.98))


def make_graph_snapshots(data_root: Path, output_dir: Path) -> None:
    epochs = [1, 25, 50, 75, 100]
    drop_range = graph_drop_range(data_root)
    fig, axes = plt.subplots(len(LAYERS), len(epochs), figsize=(15, 8.5))
    for row, layer in enumerate(LAYERS):
        for col, epoch in enumerate(epochs):
            draw_top1_graph(
                axes[row, col],
                load_matrix(data_root, epoch, layer, "S"),
                load_matrix(data_root, epoch, layer, "drop_kl"),
                f"Layer {layer} | epoch {epoch}",
                drop_range,
            )
    fig.suptitle(
        "Evolution of the Mean Conditional Substitution Graph\n"
        "Each source retains its strongest positive outgoing edge; node size encodes deletion KL",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "functional_graph_snapshots")


def make_graph_animation(data_root: Path, output_dir: Path, fps: int) -> None:
    drop_range = graph_drop_range(data_root)
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(12, 4.2))

    def update(epoch_index: int):
        epoch = epoch_index + 1
        for ax, layer in zip(axes, LAYERS):
            ax.clear()
            draw_top1_graph(
                ax,
                load_matrix(data_root, epoch, layer, "S"),
                load_matrix(data_root, epoch, layer, "drop_kl"),
                f"Layer {layer}",
                drop_range,
            )
        fig.suptitle(f"CFS graph evolution | epoch {epoch:03d}", fontsize=14)
        return []

    movie = animation.FuncAnimation(fig, update, frames=100, interval=1000 / fps)
    movie.save(output_dir / "cfs_graph_evolution.gif", writer="pillow", fps=fps, dpi=110)
    plt.close(fig)


def make_role_stability(data_root: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(15, 5.2), sharey=True)
    for ax, layer in zip(axes, LAYERS):
        final = load_matrix(data_root, 100, layer, "S")
        heatmap = np.full((100, final.shape[0]), np.nan, dtype=float)
        for epoch in range(1, 101):
            current = load_matrix(data_root, epoch, layer, "S")
            for source in range(final.shape[0]):
                similarity = safe_spearman(
                    row_rank_signature(current, source),
                    row_rank_signature(final, source),
                )
                heatmap[epoch - 1, source] = similarity
                rows.append(
                    {
                        "epoch": epoch,
                        "layer": layer,
                        "source_head": source,
                        "rank_similarity_to_final": similarity,
                    }
                )
        image = ax.imshow(
            heatmap.T,
            aspect="auto",
            origin="lower",
            extent=(1, 100, -0.5, final.shape[0] - 0.5),
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Epoch")
        ax.set_yticks(range(final.shape[0]))
        ax.set_yticklabels([f"Head {head}" for head in range(final.shape[0])])
    axes[0].set_ylabel("Source head")
    color_axis = fig.add_axes([0.915, 0.18, 0.018, 0.64])
    fig.colorbar(
        image,
        cax=color_axis,
        label="Rank similarity to epoch 100",
    )
    fig.suptitle("Stabilization of Head Functional Identity", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.12, top=0.86, wspace=0.12)
    save_figure(fig, output_dir, "functional_role_stability")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "functional_role_stability.csv", index=False)
    return frame


def make_graph_stability(data_root: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for layer in LAYERS:
        final = load_matrix(data_root, 100, layer, "S")
        final_values = off_diagonal(final)
        final_best = np.nanargmax(final, axis=1)
        for epoch in range(1, 101):
            current = load_matrix(data_root, epoch, layer, "S")
            current_best = np.nanargmax(current, axis=1)
            rows.append(
                {
                    "epoch": epoch,
                    "layer": layer,
                    "matrix_spearman_to_final": safe_spearman(
                        off_diagonal(current), final_values
                    ),
                    "top1_representative_agreement": float(
                        np.mean(current_best == final_best)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "graph_similarity_to_final.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    for layer in LAYERS:
        subset = frame[frame.layer == layer]
        axes[0].plot(
            subset.epoch,
            subset.matrix_spearman_to_final,
            label=f"Layer {layer}",
            color=COLORS[layer],
            lw=2,
        )
        axes[1].plot(
            subset.epoch,
            subset.top1_representative_agreement,
            label=f"Layer {layer}",
            color=COLORS[layer],
            lw=2,
        )
    axes[0].set_ylabel("Matrix Spearman to epoch 100")
    axes[1].set_ylabel("Top-1 edge agreement")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylim(-1.0, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3)
    fig.suptitle("Convergence Toward the Final Functional Graph", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir, "graph_similarity")
    return frame


def make_rank_plot(by_layer: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for layer in LAYERS:
        subset = by_layer[by_layer.layer == layer]
        axes[0].plot(
            subset.epoch,
            subset.rank_per_head,
            label=f"Layer {layer}",
            color=COLORS[layer],
            lw=2,
        )
        axes[1].plot(
            subset.epoch,
            subset["cover_per_head_tau_0.50"],
            label=f"Layer {layer}",
            color=COLORS[layer],
            lw=2,
        )
    axes[0].set_title("Spectral rank/head")
    axes[1].set_title("Cover/head at $\\tau=.50$")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Depth-Resolved Functional Consolidation", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir, "rank_and_cover_by_layer")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    macro = pd.read_csv(args.data_root / "functional_trajectory_macro.csv")
    by_layer = pd.read_csv(args.data_root / "functional_trajectory_by_layer.csv")
    history = load_training_history(args.training_json)

    make_trajectory_plot(macro, history, args.output_dir)
    make_graph_snapshots(args.data_root, args.output_dir)
    make_role_stability(args.data_root, args.output_dir)
    make_graph_stability(args.data_root, args.output_dir)
    make_rank_plot(by_layer, args.output_dir)
    make_graph_animation(args.data_root, args.output_dir, args.gif_fps)


if __name__ == "__main__":
    main()
