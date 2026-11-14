# Artifact index

`REPORT.md` contains the Chinese experimental report, English reviewer response,
complete result tables, qualifications and reproduction commands. The report
does not claim a uniform original-R2 selection advantage over learned gating.

* `summary.csv/json`: seed-averaged input statistics and seed-mean SD.
* `per_seed_summary.csv`: each training seed separately.
* `paired_bootstrap.csv/json`, `per_seed_paired_bootstrap.csv`: paired differences.
* `shared_hard_table.*`, `shared_learned_compensation_table.*`: selector controls.
* `full_pipeline_table.*`, `accuracy_table.csv`: system and accuracy comparisons.
* `all_results_with_ci.tex`: full numeric results and confidence intervals.
* `figure1_dynamic_routing.pdf/png`, `figure2_paired_difference.pdf/png`: main plots.
* `historical_comparison.csv`, `checks.json`: original-results reproduction audit.
* `live_feature_sensitivity/`: fixed-weight current-forward-feature sensitivity.
* `complexity.csv`, `router_costs.csv/json`: parameter/validation metadata and
  isolated router cost (`router_costs_protocol.json` documents the timing scope).
* `verification.json`: completed artifact-integrity and deterministic-smoke checks.
* `environment.txt`: exact remote Python environment.

The complete local archive is
`D:/tries2/outputs/learned_gating_final_20261115.tar.gz` (also stored on the
authorized experiment server at `/data/learned_gating_final_20261115.tar.gz`).
SHA256: `d9d9758e2caa528d61ddcafc6f025a80a23a584cbd5e4804c27f9b8f8a9bbff7`.
The extracted local directory has the same basename without `.tar.gz`.

The archive preserves six newly trained router checkpoints, three portable
original R2 checkpoints, all training logs/configs, original exported teacher
labels and frozen test images, primary and live-feature per-input CSV/JSON,
mask/scale arrays, dense logits, exact bootstrap draws/replicates and scripts.
It excludes the large unchanged ViT backbone, available locally at
`D:/tries2/assets/vit_scaling/base.pth`; its SHA256 is in `checks.json`.
Bulk arrays/checkpoints are kept outside Git. Original experimental files were
not overwritten. Optional P1 direct-subset prediction was not run.

Validation: all 48 project tests passed; all six full runs contain 60 epochs;
train/tune/test paths and image basenames are disjoint; training exports contain
no test teacher; repeated same-seed smoke weights match bitwise; every evaluated
method uses exactly the same dense prediction for each input. Main and live
feature sensitivity evaluations also have bitwise-identical dense logits.
