# Artifact index and reproduction dependencies

Start with `REPORT.md`. This is a bounded negative/mixed feasibility result,
not a newly selected higher-order routing method. Every residual variant
selected zero strength on validation; the primary remains original A3.
Unit-strength variants are uniformly reported diagnostics, declared in the
pre-test amendment before either evaluation cohort was processed.

- `selection_lock.json`, `calibration_choices.csv`, `training_history.csv`:
  all nine fits, 180 epoch records, and every calibration candidate.
- `confirmation_manifest.json`: 256 locked images and content hashes,
  excluded from the 4,608 original routing inputs by both path and hash.
- `analysis/summary.*`, `paired.*`, `per_seed.*`: complete hard-gating KL,
  accuracy, paired input confidence intervals and seed-level results.
- `analysis/prediction_summary.*`, `prediction_paired.*`: residual-held-out
  mask prediction and regret results, including all controls.
- `analysis/prediction_range_audit.csv`: negative predicted KL is preserved
  and disclosed, rather than hidden by clipping.
- `analysis/timings.csv`, `figure*.pdf/png`: measured scoring costs and figures.
- `verification.json`: frozen-source hashes, mask budgets, selected fallback,
  confirmation provenance and exact dense/baseline reconstruction checks.

Full archive: `D:/tries2/outputs/cfs_residual_final_20261115.tar.gz`.
Size: 183,895,099 bytes.
SHA256: `8efe83241915ac12d0177728770773b3fbd8e6f37aec09ac5bb2ec668d922b2b`.
Extracted directory: `D:/tries2/outputs/cfs_residual_final_20261115/`.
Remote archive: `/data/cfs_residual_final_20261115.tar.gz`.
Remote working directory: `/data/cfs_residual_20261115/`.

The archive includes all 180 epoch checkpoints plus nine locked representatives,
all per-mask predictions, 57,024 real hard-gating records, frozen confirmation
images and features, cached frozen-router outputs, bootstrap draws, source,
tests, protocol/amendment and logs. Bulk arrays/weights/images are kept out of
Git. Initial float32 mask-error diagnostics remain on the server under
`prediction_diagnostics_initial` and `analysis_initial`; the final archive uses
consistent float64 error arithmetic, so zero correction has exactly zero
prediction-error difference. Fitting, selection and real gating are unchanged.

Reproduction dependencies are the unchanged previous archives:

- `outputs/cfs_interaction_final_20261115.tar.gz`, SHA256
  `e2ef22c661ec10d52a8320cd2d8bcfdba121fbae37c60e90d130d757602ea666`.
- `outputs/learned_gating_final_20261115.tar.gz`, SHA256
  `d9d9758e2caa528d61ddcafc6f025a80a23a584cbd5e4804c27f9b8f8a9bbff7`.
- `assets/vit_scaling/base.pth`, SHA256
  `683c6d82c1e832981a5fdce9ae66be4f02542b4bd7b7fc822b08f53c005de8df`.

Set `source/` to the interaction archive, `baseline/` to the learned-gating
archive, resolve the source archive's own baseline link, and put the backbone
at baseline `base.pth`. For a fresh run copy only confirmation images,
`confirmation.npz` and `manifest.json`, not the already computed features.
See `REPORT.md` for commands. The full local suite passed 59 tests.

The additional 256 inputs have now been evaluated and must not be described
as an untouched final test set in subsequent algorithm development.
