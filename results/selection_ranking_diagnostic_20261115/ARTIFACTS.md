# Artifact index

Start with `REPORT.md`. This task is a frozen-model diagnostic only: no
training, backbone forward, new teacher collection, test or confirmation use.

- `analysis/compact_summary.csv`: readable overview for all four models,
  three budgets, and train/tune separately.
- `analysis/diagnostic_summary.*`, `paired_methods.*`: full input-bootstrap
  means, intervals, SD and model contrasts.
- `analysis/shortlist_bounds.*`, `union_bounds.*`: label-using oracle repair
  upper bounds, not deployable selectors.
- `analysis/coverage_summary.*`, `disagreement_summary.*`: full-search choice
  coverage and same-input model disagreements.
- `analysis/regret_tail_descriptive.*`, `A3_R2_loss_decomposition.*`: why
  best-candidate hit rate and mean raw KL need not order methods identically.
- `analysis/unmeasured_training_candidate_queue.csv`: 406 unique candidates
  from prediction-defined high-disagreement training inputs. Not measured,
  not representative of the entire training population, not training data yet.
- `selection_teacher_provenance.json`: exact exported masks/KL/index matches
  and original shard hashes; the old pool already used anchors/local mutations.
- `verification.json`: immutable source hashes and reconstruction checks.

Full archive: `D:/tries2/outputs/cfs_selection_diagnostic_final_20261115.tar.gz`.
Size: 143,836,862 bytes.
SHA256: `ba032848d413ac58e276f5677b85bcab398694c254128be4b743ab703599bbe7`.
Extracted local directory: `D:/tries2/outputs/cfs_selection_diagnostic_final_20261115/`.
Remote archive: `/data/cfs_selection_diagnostic_final_20261115.tar.gz`.
Remote directory: `/data/cfs_selection_diagnostic_20261115/`.

Archive contains source/tests, protocols, all 149,760 per-input/seed/budget
metric rows, 221,184 training candidate inventory rows, true labelled pools,
all pool predictions and full-search top20 candidate identities/scores,
bootstrap draws, reports, tables and figures. Unmeasured candidate KL remains
NaN. There are no newly fitted model weights. All 63 local tests passed.

Unchanged reproduction dependencies:

- `outputs/cfs_interaction_final_20261115.tar.gz`, SHA256
  `e2ef22c661ec10d52a8320cd2d8bcfdba121fbae37c60e90d130d757602ea666`.
- `outputs/learned_gating_final_20261115.tar.gz`, SHA256
  `d9d9758e2caa528d61ddcafc6f025a80a23a584cbd5e4804c27f9b8f8a9bbff7`.

Set `source/` to the extracted interaction archive and `baseline/` to the
learned-gating archive, then run the report's commands in a fresh directory.
No backbone checkpoint or images are required for this diagnostic.
The separate provenance audit can be rerun with the original local directories
`outputs/vit_base_joint_teacher_train4096_c48` and
`outputs/vit_base_joint_teacher_tune64_c48` via
`audit_selection_teacher_provenance.py`; all original shard hashes are recorded.
