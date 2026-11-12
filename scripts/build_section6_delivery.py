"""Build compact Section 6 tables from frozen observability/router artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT = OUTPUTS / "section6_delivery"


def observability_ladder() -> None:
    ladder = pd.read_csv(OUTPUTS / "full_ladder_analysis" / "ladder_summary.csv")
    baselines = pd.read_csv(OUTPUTS / "depth_replication_analysis" / "depth_baselines.csv")
    late = baselines[baselines.layer.eq(10)].iloc[0]
    ladder["model"] = "deit_small_patch16_224"
    ladder["task"] = "Imagenette-160 classification"
    ladder["layer"] = 10
    ladder["taylor_recovery"] = late.taylor_recovery
    ladder["oracle_recovery"] = late.oracle_recovery
    columns = [
        "model", "task", "layer", "level", "slug", "mean_recovery",
        "recovery_seed_std", "mean_accuracy", "accuracy_seed_std",
        "mean_pair_spearman", "spearman_seed_std", "taylor_recovery",
        "oracle_recovery",
    ]
    ladder[columns].to_csv(OUT / "observability_ladder.csv", index=False)


def endpoint(frame: pd.DataFrame, label: str) -> pd.Series:
    return frame[frame.level.eq(label)].iloc[0]


def cross_model_observability() -> None:
    rows = []

    deit = pd.read_csv(OUTPUTS / "full_ladder_analysis" / "ladder_summary.csv")
    deit_base = pd.read_csv(OUTPUTS / "depth_replication_analysis" / "depth_baselines.csv")
    f, g = endpoint(deit, "L2 Q/K moments"), endpoint(deit, "L7 gradient")
    b = deit_base[deit_base.layer.eq(10)].iloc[0]
    rows.append(_cross_row("DeiT-S", "Imagenette", 10, "L2 Q/K moments", f, g, b))

    vit = pd.read_csv(OUTPUTS / "vitaug_replication_analysis" / "summary.csv")
    f, g = endpoint(vit, "L2 Q/K moments"), endpoint(vit, "L7 gradient")
    rows.append(_cross_row("ViT-S AugReg", "Imagenette", 10, "L2 Q/K moments", f, g, f))

    dino = pd.read_csv(OUTPUTS / "dino_clean_replication_analysis" / "summary.csv")
    f, g = endpoint(dino, "L2 Q/K moments"), endpoint(dino, "L7 gradient")
    rows.append(_cross_row("DINO ViT-S", "Imagenette linear probe", 10, "L2 Q/K moments", f, g, f))

    sst = pd.read_csv(OUTPUTS / "bert_cfs_analysis" / "endpoint_summary.csv")
    sst = sst[sst.layer.eq(10)]
    f, g = sst[sst.endpoint.eq("Post-WO")].iloc[0], sst[sst.endpoint.eq("Gradient")].iloc[0]
    sst_base = pd.read_csv(OUTPUTS / "bert_cfs_analysis" / "baseline_summary.csv")
    sst_base = sst_base[sst_base.layer.eq(10)]
    rows.append(_bert_row("BERT-base", "SST-2", 10, "Post-WO", f, g, sst_base))

    mnli = pd.read_csv(OUTPUTS / "bert_mnli_cfs_analysis" / "endpoint_summary.csv")
    f, g = mnli[mnli.endpoint.eq("Post-WO")].iloc[0], mnli[mnli.endpoint.eq("Gradient")].iloc[0]
    mnli_base = pd.read_csv(OUTPUTS / "bert_mnli_cfs_analysis" / "baseline_summary.csv")
    rows.append(_bert_row("BERT-base", "MNLI", 10, "Post-WO", f, g, mnli_base))
    pd.DataFrame(rows).to_csv(OUT / "observability_cross_model.csv", index=False)


def _cross_row(model, task, layer, source, forward, gradient, baseline):
    return {
        "model": model,
        "task": task,
        "layer": layer,
        "forward_source": source,
        "forward_recovery": forward.mean_recovery,
        "forward_seed_std": forward.recovery_seed_std,
        "functional_source": "gradient sensitivity",
        "functional_recovery": gradient.mean_recovery,
        "functional_seed_std": gradient.recovery_seed_std,
        "functional_minus_forward": gradient.mean_recovery - forward.mean_recovery,
        "forward_representative_accuracy": forward.mean_accuracy,
        "functional_representative_accuracy": gradient.mean_accuracy,
        "forward_pair_spearman": forward.mean_pair_spearman,
        "functional_pair_spearman": gradient.mean_pair_spearman,
        "taylor_recovery": baseline.taylor_recovery,
        "oracle_recovery": baseline.oracle_recovery,
        "seeds": "17,23,41",
    }


def _bert_row(model, task, layer, source, forward, gradient, baseline):
    taylor = baseline[baseline.policy.eq("highest_taylor_importance")].mean_recovery.iloc[0]
    oracle = baseline[baseline.policy.eq("functional_oracle")].mean_recovery.iloc[0]
    return {
        "model": model,
        "task": task,
        "layer": layer,
        "forward_source": source,
        "forward_recovery": forward.mean_recovery,
        "forward_seed_std": forward.seed_std,
        "functional_source": "gradient sensitivity",
        "functional_recovery": gradient.mean_recovery,
        "functional_seed_std": gradient.seed_std,
        "functional_minus_forward": gradient.mean_recovery - forward.mean_recovery,
        "forward_representative_accuracy": forward.representative_accuracy,
        "functional_representative_accuracy": gradient.representative_accuracy,
        "forward_pair_spearman": forward.pair_spearman,
        "functional_pair_spearman": gradient.pair_spearman,
        "taylor_recovery": taylor,
        "oracle_recovery": oracle,
        "seeds": "17,23,41",
    }


def router_tables() -> None:
    r1_dir = OUTPUTS / "vit_base_router_final_execution_multiseed"
    r2_dir = OUTPUTS / "vit_base_joint_router_calibrated_multiseed_w05"
    r1 = pd.read_csv(r1_dir / "aggregate_multiseed.csv")
    r2 = pd.read_csv(r2_dir / "aggregate_multiseed.csv")
    boot = pd.read_csv(r2_dir / "paired_bootstrap.csv")
    boot.to_csv(OUT / "router_bootstrap.csv", index=False)
    r2.to_csv(OUT / "router_multiseed_summary.csv", index=False)

    dense = r2[r2.policy.eq("dense")].iloc[0]
    rows = []
    for keep in [3, 6, 9]:
        router = r2[(r2.keep.eq(keep)) & r2.policy.eq("dynamic_cfs_router_prune")].iloc[0]
        taylor = r2[(r2.keep.eq(keep)) & r2.policy.eq("taylor_prune")].iloc[0]
        accuracy_ci = boot[(boot.keep.eq(keep)) & boot.router.eq("dynamic_cfs_router_prune") &
                           boot.baseline.eq("taylor_prune") & boot.metric.eq("correct")].iloc[0]
        kl_ci = boot[(boot.keep.eq(keep)) & boot.router.eq("dynamic_cfs_router_prune") &
                     boot.baseline.eq("taylor_prune") & boot.metric.eq("kl_to_dense")].iloc[0]
        rows.append({
            "keep": keep,
            "keep_fraction": keep / 12,
            "test_images": 448,
            "router_seeds": 3,
            "dense_accuracy": dense.accuracy_mean,
            "cfs_r2_accuracy": router.accuracy_mean,
            "cfs_r2_accuracy_seed_std": router.accuracy_seed_std,
            "taylor_accuracy": taylor.accuracy_mean,
            "accuracy_delta_cfs_minus_taylor": router.accuracy_mean - taylor.accuracy_mean,
            "accuracy_delta_ci95_low": accuracy_ci.ci95_low,
            "accuracy_delta_ci95_high": accuracy_ci.ci95_high,
            "cfs_r2_dense_agreement": router.agreement_mean,
            "taylor_dense_agreement": taylor.agreement_mean,
            "cfs_r2_kl": router.kl_mean,
            "cfs_r2_kl_seed_std": router.kl_seed_std,
            "taylor_kl": taylor.kl_mean,
            "kl_delta_cfs_minus_taylor": router.kl_mean - taylor.kl_mean,
            "kl_delta_ci95_low": kl_ci.ci95_low,
            "kl_delta_ci95_high": kl_ci.ci95_high,
            "relative_kl_reduction": (taylor.kl_mean - router.kl_mean) / taylor.kl_mean,
        })
    pd.DataFrame(rows).to_csv(OUT / "router_main_results.csv", index=False)

    ablation_ci = pd.read_csv(
        OUTPUTS / "vit_base_joint_router_execution_multiseed_w05" /
        "joint_vs_pairwise_bootstrap.csv"
    )
    ablation_rows = []
    for keep in [3, 6, 9]:
        for stage, frame in [("R1 structured pairwise supervision", r1),
                             ("R2 + joint-subset supervision", r2)]:
            row = frame[(frame.keep.eq(keep)) & frame.policy.eq("dynamic_cfs_router_prune")].iloc[0]
            ablation_rows.append({
                "keep": keep, "stage": stage, "accuracy": row.accuracy_mean,
                "accuracy_seed_std": row.accuracy_seed_std, "kl_to_dense": row.kl_mean,
                "kl_seed_std": row.kl_seed_std,
            })
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(OUT / "router_ablation.csv", index=False)
    ablation_ci.to_csv(OUT / "router_ablation_bootstrap.csv", index=False)

    calibration_controls(r2_dir, r2)
    online_consistency()


def calibration_controls(directory: Path, summary: pd.DataFrame) -> None:
    bootstrap = pd.read_csv(directory / "calibrated_pairing_bootstrap.csv")
    rows = []
    comparisons = [
        ("same learned pairwise calibration", "dynamic_cfs_router_merge", "taylor_router_merge"),
        ("oracle pairwise alpha", "dynamic_cfs_router_oracle_merge", "taylor_oracle_merge"),
    ]
    for keep in [3, 6, 9]:
        for control, cfs_policy, taylor_policy in comparisons:
            cfs = summary[(summary.keep.eq(keep)) & summary.policy.eq(cfs_policy)].iloc[0]
            taylor = summary[(summary.keep.eq(keep)) & summary.policy.eq(taylor_policy)].iloc[0]
            for metric, cfs_value, taylor_value in [
                ("accuracy", cfs.accuracy_mean, taylor.accuracy_mean),
                ("dense_agreement", cfs.agreement_mean, taylor.agreement_mean),
                ("kl_to_dense", cfs.kl_mean, taylor.kl_mean),
            ]:
                source_metric = "correct" if metric == "accuracy" else metric
                ci = bootstrap[(bootstrap.keep.eq(keep)) & bootstrap.router.eq(cfs_policy) &
                               bootstrap.baseline.eq(taylor_policy) &
                               bootstrap.metric.eq(source_metric)].iloc[0]
                rows.append({
                    "keep": keep, "control": control, "metric": metric,
                    "cfs_policy": cfs_policy, "taylor_policy": taylor_policy,
                    "cfs_value": cfs_value, "taylor_value": taylor_value,
                    "delta_cfs_minus_taylor": cfs_value - taylor_value,
                    "ci95_low": ci.lo, "ci95_high": ci.hi,
                })
    pd.DataFrame(rows).to_csv(OUT / "router_selection_calibration_controls.csv", index=False)


def online_consistency() -> None:
    directory = OUTPUTS / "vit_base_router_final_online_seed17"
    summary = pd.read_csv(directory / "online_execution_summary.csv")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    summary = summary[summary.keep.lt(12)].copy()
    summary["images"] = metadata["images"]
    summary["mask_match"] = summary.images - summary.mask_mismatch
    summary["mask_agreement"] = summary.mask_match / summary.images
    summary["timing_claim"] = "correctness only; dense fused QKV remains active"
    summary.to_csv(OUT / "router_online_consistency.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observability_ladder()
    cross_model_observability()
    router_tables()
    print(f"Wrote Section 6 delivery tables to {OUT}")


if __name__ == "__main__":
    main()
