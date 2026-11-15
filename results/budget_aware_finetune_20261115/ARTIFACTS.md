# Artifact index

Start with `REPORT.md`. The prespecified validation rule admitted no update;
the primary remains the original A3. W1.5 is a diagnostic result, not a new
selected model. No tested update attained all three requested test targets.

- `selection_lock.json`: validation-only selection and four diagnostic representatives.
- `validation_trajectories.csv`: all 132 seed/weight/epoch records, including initial states.
- `validation_candidates.csv`: all 40 weight/epoch candidates averaged over seeds.
- `analysis/summary.*`, `paired.*`, `targets.*`: CSV, JSON and LaTeX results.
- `analysis/per_seed_summary.csv`: individual training-seed results.
- `analysis/gradient_diagnostics.csv`, `probe_losses.csv`: fixed-probe diagnostics.
- `analysis/figure*.pdf/png`: test comparisons and validation trajectories.
- `verification.json`: 12 runs, 7,680 updates, unchanged source weights and 72,576 routing rows.

Complete archive: `D:/tries2/outputs/cfs_budget_final_20261115.tar.gz`.
Size: 48,835,524 bytes.
SHA256: `645ccc3f45496c0a3b5521c8e34738ad494b39762ffb67c99275bc34f7e40750`.
Extracted directory: `D:/tries2/outputs/cfs_budget_final_20261115/`.
Remote archive: `/data/cfs_budget_final_20261115.tar.gz`.
Remote working directory: `/data/cfs_budget_20261115/`.

The archive contains all run histories/configurations/logs, the 12 validation-
locked representative checkpoints, full per-input evaluations, bootstrap draws,
code and reports. All 120 epoch checkpoints remain in the remote `runs/`
directory; only the 12 representatives are duplicated in this compact archive.
Intermediate `eval_w*/per_input.csv` uses the existing evaluator's internal
`R2_joint_only` name for the substituted checkpoint. The combined
`routing_per_input.csv` maps these to W1/W1.5/W2/W3 explicitly.

Unchanged dependencies are kept in the previous archives:

- `outputs/cfs_interaction_final_20261115.tar.gz`, SHA256
  `e2ef22c661ec10d52a8320cd2d8bcfdba121fbae37c60e90d130d757602ea666`.
- `outputs/learned_gating_final_20261115.tar.gz`, SHA256
  `d9d9758e2caa528d61ddcafc6f025a80a23a584cbd5e4804c27f9b8f8a9bbff7`.
- `assets/vit_scaling/base.pth`, SHA256
  `683c6d82c1e832981a5fdce9ae66be4f02542b4bd7b7fc822b08f53c005de8df`.

For reproduction, set `source/` to the extracted interaction archive and
`baseline/` to the extracted learned-gating archive; the source archive's
own `baseline/` must also resolve. Put the backbone at baseline `base.pth`.
Use a fresh output directory and the commands in `REPORT.md`; existing runs
are not overwritten. The complete local test suite passed: 55 tests.
