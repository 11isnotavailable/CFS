# Source Mapping

This file maps the curated protocols to the released implementation and compact
result artifacts.

| Curated document | Primary internal sources |
|---|---|
| Core algorithm | `CFS_MATHEMATICAL_DEFINITIONS.md`, `src/head_intervention.py`, `src/text_head_intervention.py`, `src/qwen_head_intervention.py` |
| Discovery and validation | `EXPERIMENT0.md`, `EXPERIMENT15.md`, `scripts/run_experiment0.py`, `scripts/analyze_experiment15.py`, `scripts/analyze_section4_validation_patches.py` |
| Observability | `FULL_OBSERVABILITY_LADDER_RESULTS.md`, `scripts/collect_observability_features.py`, `scripts/analyze_full_ladder.py`, `scripts/analyze_gradient_ablation.py` |
| Scaling/generalization | `scripts/analyze_functional_capacity.py`, `scripts/collect_bert_mlm_scaling.py`, `scripts/collect_qwen_cfs.py`, `scripts/controlled_gpt2/`, `results/section5/` |
| Dynamic router | `src/dynamic_cfs_router.py`, `src/runtime_cfs_router.py`, `scripts/train_dynamic_cfs_router.py`, `scripts/evaluate_dynamic_cfs_execution.py`, `results/section6/` |
| Training dynamics | `scripts/build_functional_evolution_snapshots.py`, `scripts/plot_functional_evolution.py`, `results/section5/training_dynamics_*.csv` |

## Public-release cleanup checklist

- Remove machine-specific absolute paths and credentials.
- Replace private storage locations with documented download/preparation steps.
- Freeze protocol-version identifiers and configuration hashes.
- Add unit tests for all-one gating, contribution decomposition, GQA query-head
  indexing, per-input aggregation, exact Matched-\(k\), and cover solving.
- Publish compact sample indices and checksums.
- Re-run every public command in a clean environment before tagging release.

