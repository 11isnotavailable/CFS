# Artifact index and reproduction dependencies

Start with `REPORT.md`, including the full ablation, limitations and separate
English replies for WCQn and UWwP. The validation-preselected headline is A3
(Joint+CFS+Alpha, lambda0.01); it was not changed to the best test result.

* `pilot_validation.csv`, `selection_lock.json`: all15 pilots and locked choices.
* `routing_analysis/`: all budgets, seed means/SD, paired input-bootstrap
  intervals, accuracy, compensation decomposition and fixed-threshold groups.
* `interaction_analysis/`: actual interaction statistics, within-input
  correlations, disjoint-input ridge comparisons and CFS incremental value.
* `additive_final/`: held-out-mask cross-validation statistics, including
  training-vs-held-out prediction error and constant-predictor controls.
* `figures/`: four PDF/PNG figures, all selected training histories, fixed-probe
  losses and gradient diagnostics. Gradient bands are seed min/max, not CIs.
* `descriptive_checks/`: every head pair, direct auxiliary ablations, high-minus-
  low advantage, numerical-noise diagnostics and sign-prediction controls.
* `verification.json`: all21 runs completed; exact measured quartet identities;
  baseline reconstruction; 56,448 uniform routing rows.

Complete archive:
`D:/tries2/outputs/cfs_interaction_final_20261115.tar.gz` (95,928,844 bytes).
SHA256: `e2ef22c661ec10d52a8320cd2d8bcfdba121fbae37c60e90d130d757602ea666`.
Extracted local directory: `D:/tries2/outputs/cfs_interaction_final_20261115/`.
Remote archive: `/data/cfs_interaction_final_20261115.tar.gz`.
Remote experiment directory: `/data/cfs_interaction_20261115/`.

The archive includes all21 new checkpoints/configs/logs (not just winners),
exported auxiliary targets, fixed test indices, tune images, every measured
mask and raw KL, all38,016 quartet identities, full56,448 routing rows,
per-pair/per-input prediction files and full numerical results. Bulk weights
and arrays are not duplicated in Git. Intermediate `eval_A0/A1/A2/A3` reuse
the previous evaluator's internal `R2_joint_only` label for the substituted
checkpoint; `routing_per_input.csv` maps these unambiguously to A0/A1/A2/A3.

The archive intentionally does not copy unchanged baseline data/weights again.
Reproduction needs the previous archive
`outputs/learned_gating_final_20261115.tar.gz`, SHA256
`d9d9758e2caa528d61ddcafc6f025a80a23a584cbd5e4804c27f9b8f8a9bbff7`,
plus `assets/vit_scaling/base.pth`, SHA256
`683c6d82c1e832981a5fdce9ae66be4f02542b4bd7b7fc822b08f53c005de8df`.
Place the backbone at the baseline directory's `base.pth` and point `baseline/`
to that extracted baseline directory. Run reproduction commands in a fresh
experiment directory; outputs deliberately refuse to overwrite existing runs.

Validation: 53 project tests passed. Original results/checkpoints/backbone and
paper were unchanged. No new pairwise teacher or scaling experiment was run.
All2000-replicate bootstrap results use input units; seeds are averaged within
input and separately reported. Exact replication is determined by seed42 and
saved input ordering. Conditional subgroup thresholds use tune inputs only.

Evidence boundary: A3 has a small K9 improvement over Joint-only; A1's pure-CFS
increment remains unresolved. CFS adds K9 held-out interaction-prediction value,
but the complete causal chain to A3 routing gains is not established.
Training-budget matching refers to epochs/updates, not equal forward/backward
FLOPs or wall-clock cost; objectives and subset-search work differ.
