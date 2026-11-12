"""Build compact Section 5 tables from existing CFS artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_functional_capacity import spectral_effective_rank


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "section5_delivery"


def vit_table(dataset: str) -> pd.DataFrame:
    if dataset == "imagenet1k":
        base = ROOT / "outputs" / "imagenet1k_final" / "outputs"
        scaling = pd.read_csv(
            base / "imagenet1k_scaling_analysis_test512_alpha31" / "scaling_summary.csv"
        )
        capacity = pd.read_csv(
            base / "imagenet1k_functional_capacity_test512_alpha31" / "functional_capacity_summary.csv"
        )
        task = "ImageNet-1K classification"
    else:
        scaling = pd.read_csv(ROOT / "outputs" / "vit_scaling_analysis_tsbl" / "scaling_summary.csv")
        # The compact capacity table is reconstructed from the frozen primary
        # result reported in FUNCTIONAL_CAPACITY_RESULTS.md.
        capacity = pd.DataFrame(
            {
                "model_size": ["tiny", "small", "base", "large"],
                "cover_ratio_mean": [0.800, 0.733, 0.636, 0.379],
                "spectral_rank_ratio_mean": [0.890, 0.868, 0.756, 0.475],
                "minimum_cover_mean": [2.400, 4.396, 7.637, 6.062],
                "spectral_effective_rank_mean": [2.670, 5.209, 9.075, 7.594],
            }
        )
        task = "ImageNet-1K classifier on Imagenette-160 inputs"
    capacity = capacity[capacity.get("threshold", pd.Series([0.5] * len(capacity))).eq(0.5)]
    merged = scaling.merge(capacity, on="model_size", suffixes=("", "_capacity"))
    params = {"tiny": 5_717_416, "small": 22_050_664, "base": 86_567_656, "large": 304_326_632}
    layers = {"tiny": 11, "small": 11, "base": 11, "large": 23}
    # The ImageNet-1K source table already carries parameter metadata, whereas
    # the compact Imagenette reconstruction does not.  Overwrite uniformly so
    # both exported tables use the same canonical architecture values.
    merged["parameters"] = merged.model_size.map(params)
    merged["head_dim"] = 64
    merged["evaluated_layers"] = merged.model_size.map(layers).astype(str)
    merged["task"] = task
    merged["task_performance"] = merged.classification_accuracy
    merged["raw_oracle"] = merged.oracle_recovery
    merged["matched_2"] = merged.matched_best_of_2_recovery
    merged["cover_per_head_tau_0.5"] = merged.cover_ratio_mean
    merged["rank_per_head"] = merged.spectral_rank_ratio_mean
    merged["mean_deletion_kl"] = np.nan
    merged["weighted_recovery"] = np.nan
    merged["samples"] = 512
    columns = [
        "model_size", "model", "parameters", "heads", "head_dim", "task",
        "task_performance", "evaluated_layers", "samples", "valid_source_coverage",
        "raw_oracle", "matched_2", "cover_per_head_tau_0.5", "rank_per_head",
        "mean_deletion_kl", "weighted_recovery",
    ]
    return merged[columns]


def bert_table() -> pd.DataFrame:
    base = ROOT / "outputs" / "bert_mlm_natural_scaling" / "analysis"
    architecture = pd.read_csv(base / "architecture_and_loss.csv")
    metrics = pd.read_csv(base / "bootstrap_summary.csv").pivot(index="tag", columns="metric", values="mean")
    table = architecture.merge(metrics, left_on="tag", right_index=True)
    table["task"] = "pretrained MLM on frozen MNLI text"
    table["task_performance"] = table.mlm_loss
    table["task_performance_name"] = "MLM loss"
    table["evaluated_layers"] = table.layer.astype(str)
    table["samples"] = 32
    table["raw_oracle"] = table.raw_oracle
    table["matched_2"] = table.matched_2
    table["cover_per_head_tau_0.5"] = table.cover_per_head
    # Recompute with the same self-affinity convention used by the ViT and
    # controlled-scaling capacity analyses.  The original BERT aggregation
    # omitted the unit diagonal, so retain it as an explicitly legacy column.
    table["rank_per_head_legacy_no_self"] = table.rank_per_head
    aligned_rank = {}
    for tag in table.tag:
        with np.load(base.parent / f"{tag}_final" / "shard_00000.npz") as archive:
            drops = archive["single_drop"][:, 0].astype(np.float64)
            substitution = archive["substitution"][:, 0].astype(np.float64)
        scores = 1 - substitution / np.maximum(drops[:, :, None], 1e-12)
        diagonal = np.arange(drops.shape[1])
        scores[:, diagonal, diagonal] = np.nan
        valid = drops > 1e-5
        aligned_rank[tag] = float((spectral_effective_rank(scores, valid) / drops.shape[1]).mean())
    table["rank_per_head"] = table.tag.map(aligned_rank)
    return table[
        ["tag", "model", "parameters", "depth", "heads", "head_dim", "hidden_size",
         "task", "task_performance_name", "task_performance", "pseudo_perplexity",
         "evaluated_layers", "samples", "valid_source_rate", "raw_oracle", "matched_2",
         "cover_per_head_tau_0.5", "rank_per_head", "rank_per_head_legacy_no_self",
         "mean_drop_kl", "weighted_recovery"]
    ]


def qwen_table() -> pd.DataFrame:
    architecture = pd.read_csv(ROOT / "outputs" / "qwen_measurement_audit" / "architecture.csv")
    rows = []
    model_specs = [
        ("Qwen/Qwen2.5-0.5B", 494_032_768, 14, 2, 24, 64, "5,11,17", 2.9938624002970755, 19.96263747733893, 0.3607911190277241, 0.21385315620893197, 0.8288690476190477, 0.005381772669330704, 0.40415616333271953),
        ("Qwen/Qwen2.5-1.5B", 1_543_714_304, 12, 2, 28, 128, "6,13,20", 2.7024201876483858, 14.915787080568979, 0.27251864183106517, 0.16200514026237084, 0.954861111111111, 0.004994484602144035, 0.3020734910933044),
        ("Qwen/Qwen2.5-3B", 3_085_938_688, 16, 2, 36, 128, "8,17,26", 2.58187250688934, 13.221873042674027, 0.3198443829488708, 0.19526065630013323, 0.8984375, 0.0025676158175353216, 0.35219375347439624),
        ("Qwen/Qwen2.5-7B", 7_615_616_512, 28, 4, 28, 128, "6,13,20", 2.456998761419417, 11.669735270464203, 0.37123391366365754, 0.21804189353392983, 0.8240327380952381, 0.0016927586210288479, 0.39297009616477147),
        ("Qwen/Qwen2.5-14B", 14_770_033_664, 40, 8, 48, 128, "11,23,35", 2.3133842521347106, 10.108576805306225, 0.4430, 0.2745, 0.6992, 0.00174, 0.4697),
    ]
    recovered = pd.read_csv(OUT / "qwen_random8_unified_rank.csv")
    if "Qwen/Qwen2.5-14B" not in set(recovered.model):
        recovered = pd.concat([recovered, qwen14_rank_rows()], ignore_index=True)
        recovered.to_csv(OUT / "qwen_random8_unified_rank.csv", index=False)
    recovered = recovered[recovered.aggregation.eq("three_layer_mean")]
    rank_by_model = dict(zip(recovered.model, recovered.rank_per_head))
    for spec in model_specs:
        model, params, heads, kv, depth, head_dim, layers, loss, ppl, raw, matched, cover, drop, weighted = spec
        rows.append(
            {
                "model": model, "parameters": params, "depth": depth,
                "query_heads": heads, "kv_heads": kv, "head_dim": head_dim,
                "task": "causal LM on frozen FineWeb-Edu slice",
                "task_performance_name": "all-token validation loss",
                "task_performance": loss, "perplexity": ppl,
                "evaluated_layers": layers, "depth_fractions": "0.25,0.50,0.75",
                "samples": 32, "valid_source_rate": 1.0,
                "raw_oracle": raw, "matched_2": matched,
                "cover_per_head_tau_0.5": cover,
                "rank_per_head": rank_by_model[model],
                "mean_deletion_kl": drop, "weighted_recovery": weighted,
                "rank_status": (
                    "computed from local raw matrices with unit self-affinity"
                    if "14B" in model else
                    "computed from recovered 3090 raw matrices with unit self-affinity"
                ),
            }
        )
    return pd.DataFrame(rows)


def qwen14_rank_rows() -> pd.DataFrame:
    directory = ROOT / "outputs" / "qwen14b_final" / "qwen14b_random8_32_alpha16"
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    drops, substitutions = [], []
    for shard in sorted(directory.glob("shard_*.npz")):
        with np.load(shard) as archive:
            drops.append(archive["single_drop"].astype(np.float64))
            substitutions.append(archive["substitution"].astype(np.float64))
    drop = np.concatenate(drops)
    substitution = np.concatenate(substitutions)
    heads = drop.shape[-1]
    selected_layers = metadata.get("selected_layers", [11, 23, 35])
    rows = []
    for layer in range(drop.shape[1]):
        scores = 1 - substitution[:, layer] / np.maximum(drop[:, layer, :, None], 1e-12)
        diagonal = np.arange(heads)
        scores[:, diagonal, diagonal] = np.nan
        valid = drop[:, layer] > 1e-5
        rank = spectral_effective_rank(scores, valid)
        rows.append({
            "model": "Qwen/Qwen2.5-14B",
            "run": "qwen14b_random8_32_alpha16",
            "aggregation": "per_layer",
            "samples": len(drop),
            "layer": int(selected_layers[layer]),
            "layer_offset": layer,
            "heads": heads,
            "minimum_drop_kl": 1e-5,
            "spectral_effective_rank": float(rank.mean()),
            "rank_per_head": float((rank / heads).mean()),
            "valid_source_rate": float(valid.mean()),
        })
    frame = pd.DataFrame(rows)
    overall = {
        "model": "Qwen/Qwen2.5-14B",
        "run": "qwen14b_random8_32_alpha16",
        "aggregation": "three_layer_mean",
        "samples": len(drop),
        "layer": -1,
        "layer_offset": -1,
        "heads": heads,
        "minimum_drop_kl": 1e-5,
        "spectral_effective_rank": float(frame.spectral_effective_rank.mean()),
        "rank_per_head": float(frame.rank_per_head.mean()),
        "valid_source_rate": float(frame.valid_source_rate.mean()),
    }
    return pd.concat([frame, pd.DataFrame([overall])], ignore_index=True)


def gmacs(depth: int, width: int, classes: int = 100) -> float:
    tokens = 197
    patch = 196 * width * 3 * 16 * 16
    blocks = depth * (12 * tokens * width * width + 2 * tokens * tokens * width)
    classifier = width * classes
    return (patch + blocks + classifier) / 1e9


def controlled_axis(axis: str, configs: list[dict]) -> pd.DataFrame:
    base = ROOT / "outputs" / "controlled_scaling_smallbase_v1"
    head = pd.read_csv(base / f"analysis_{axis}_all" / "head_repartition_summary.csv")
    capacity = pd.read_csv(base / f"analysis_{axis}_capacity_all" / "functional_capacity_summary.csv")
    capacity = capacity[capacity.threshold.eq(0.5)].reset_index(drop=True)
    head = head.reset_index(drop=True)
    values = []
    for index, config in enumerate(configs):
        h, c = head.iloc[index], capacity.iloc[index]
        values.append(
            {
                **config,
                "parameters": int(c.parameters),
                "estimated_gmacs_224": gmacs(config["depth"], config["width"]),
                "accuracy": float(h.accuracy),
                "mean_deletion_kl": float(h.mean_drop_kl),
                "raw_oracle": float(h.oracle_recovery),
                "matched_2": float(h.matched_best_of_2),
                "cover_per_head_tau_0.5": float(c.cover_ratio_mean),
                "rank_per_head": float(c.spectral_rank_ratio_mean),
                "weighted_recovery": float(h.kl_weighted_recovery),
                "valid_source_rate": float(h.valid_source_ratio),
                "samples": int(h.samples),
            }
        )
    return pd.DataFrame(values)


def training_dynamics() -> None:
    source = ROOT / "outputs" / "functional_evolution_visualizations"
    layer = pd.read_csv(source / "data" / "functional_trajectory_by_layer.csv")
    performance = []
    for epoch in range(1, 101):
        payload = json.loads(
            (source / "data" / f"epoch_{epoch:03d}" / "metrics.json").read_text(encoding="utf-8")
        )
        performance.append({"epoch": epoch, "accuracy": payload["accuracy"], "loss": payload["loss"]})
    performance = pd.DataFrame(performance)
    joined = layer.merge(performance, on="epoch")
    ordered = ["epoch", "layer", "accuracy", "loss"] + [
        value for value in joined.columns if value not in {"epoch", "layer", "accuracy", "loss"}
    ]
    joined[ordered].to_csv(OUT / "training_dynamics_epoch_layer.csv", index=False)

    macro = pd.read_csv(source / "data" / "functional_trajectory_macro.csv").drop(columns=["layer"])
    macro.merge(performance, on="epoch").to_csv(OUT / "training_dynamics_epoch_macro.csv", index=False)
    graph = pd.read_csv(source / "graph_similarity_to_final.csv")
    graph.to_csv(OUT / "training_dynamics_graph_similarity.csv", index=False)
    role = pd.read_csv(source / "functional_role_stability.csv")
    role.to_csv(OUT / "training_dynamics_head_role_stability.csv", index=False)

    # A compact checkpoint-by-layer view for prose/table drafting.  The full
    # 100-epoch files above remain authoritative.
    role_mean = role.groupby(["epoch", "layer"], as_index=False).rank_similarity_to_final.mean()
    role_mean = role_mean.rename(columns={"rank_similarity_to_final": "mean_head_role_similarity_to_final"})
    checkpoints = joined[joined.epoch.isin([1, 5, 10, 25, 50, 75, 100])]
    checkpoints = checkpoints.merge(graph, on=["epoch", "layer"]).merge(
        role_mean, on=["epoch", "layer"]
    )
    checkpoint_columns = [
        "epoch", "layer", "accuracy", "loss", "raw_oracle", "matched_2",
        "cover_per_head_tau_0.50", "rank_per_head", "mean_deletion_kl",
        "matrix_spearman_to_final", "top1_representative_agreement",
        "mean_head_role_similarity_to_final",
    ]
    checkpoints[checkpoint_columns].to_csv(
        OUT / "training_dynamics_layer_summary.csv", index=False
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vit_table("imagenet1k").to_csv(OUT / "natural_scaling_vit.csv", index=False)
    vit_table("imagenette").to_csv(OUT / "natural_scaling_vit_imagenette_replication.csv", index=False)
    bert_table().to_csv(OUT / "natural_scaling_bert.csv", index=False)
    qwen_table().to_csv(OUT / "natural_scaling_qwen.csv", index=False)
    pd.DataFrame(
        [{
            "family": "DeiT", "available_models": "DeiT-S only",
            "complete_natural_scaling_sweep": False,
            "recommended_section5_use": "omit as a natural-scaling family; retain discovery/validation experiments",
        }]
    ).to_csv(OUT / "natural_scaling_deit_availability.csv", index=False)

    controlled_axis("depth", [
        {"model_id": "D6", "depth": 6, "width": 384, "heads": 6, "head_dim": 64},
        {"model_id": "D12", "depth": 12, "width": 384, "heads": 6, "head_dim": 64},
        {"model_id": "D18", "depth": 18, "width": 384, "heads": 6, "head_dim": 64},
    ]).to_csv(OUT / "controlled_scaling_depth.csv", index=False)
    controlled_axis("width", [
        {"model_id": "W192", "depth": 12, "width": 192, "heads": 3, "head_dim": 64},
        {"model_id": "W384", "depth": 12, "width": 384, "heads": 6, "head_dim": 64},
        {"model_id": "W768", "depth": 12, "width": 768, "heads": 12, "head_dim": 64},
    ]).to_csv(OUT / "controlled_scaling_width.csv", index=False)
    controlled_axis("granularity", [
        {"model_id": "H3", "depth": 12, "width": 384, "heads": 3, "head_dim": 128},
        {"model_id": "H6", "depth": 12, "width": 384, "heads": 6, "head_dim": 64},
        {"model_id": "H12", "depth": 12, "width": 384, "heads": 12, "head_dim": 32},
    ]).to_csv(OUT / "controlled_scaling_head_granularity.csv", index=False)
    training_dynamics()
    print(f"Wrote Section 5 delivery to {OUT}")


if __name__ == "__main__":
    main()
