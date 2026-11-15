# Reproducibility and artifact inventory

Run root: `/data/cfs_coalitional_20261116` on the authorized experiment server.
Full local archive: `D:/tries2/outputs/cfs_coalitional_artifacts_20261116.tar.gz`.
Remote archive: `/data/cfs_coalitional_artifacts_20261116.tar.gz`.
Size: **1,155,532,704 bytes**. SHA256:
`5ab9b6388da03bfd7fce877d2e449d789e87d1691f0465d400b51f1200903678`.

The archive includes every teacher record and its raw deletion/substitution
KL, grid coefficients, identities, pre-attention feature and cached suffix
tensors; all frozen images/splits; all predictions; all selected models and
saved training checkpoints; logs; benchmark raw measurements; source and tests.
There are 1,856 teacher input/pool records, 51 formal fitted models, and 2 pilot
models. No raw teacher tensor was omitted. The continuous-sensitivity records
store optimized coefficients with their corresponding optimized losses;
`joint_kl` also includes retained grid/refined-pair feasible solutions. Main
teacher `joint_alpha` always corresponds to its selected feasible solution.

The unchanged backbone is excluded from the archive because it already exists
at `D:/tries2/assets/vit_scaling/base.pth` and remotely at
`/data/cfs_learned_gating_20261115/base.pth`. SHA256:
`683c6d82c1e832981a5fdce9ae66be4f02542b4bd7b7fc822b08f53c005de8df`.
The loader and verification use that remote path; map it to the same-hash
checkpoint when reproducing elsewhere. No backbone training took place.

## Start here

- [REPORT.md](REPORT.md): Q1–Q5, methods, all budget comparisons, limitations,
  execution commands, and the English rebuttal.
- [REBUTTAL_nLkz.md](REBUTTAL_nLkz.md): standalone English response draft.
- [main_table.csv](main_table.csv), [main_table.tex](main_table.tex): main curves.
- [summary.csv](summary.csv), [per_seed.csv](per_seed.csv): means, input-level
  95% CIs, input SD and training-seed SD.
- [paired.csv](paired.csv), [pair_information_paired.csv](pair_information_paired.csv):
  paired task differences and direct-Joint versus predicted-Pairwise aggregation.
- [per_input.csv](per_input.csv): all 52,992 input-level evaluation records.
  Its JSON duplicate is in the full archive, avoiding another large Git file.
- [oracle.csv](oracle.csv), [oracle_per_input.csv](oracle_per_input.csv): oracle
  gains, input/source improvement fractions and deletion-threshold sensitivity.
- [teacher_events.csv](teacher_events.csv), [teacher_per_input_cost.csv](teacher_per_input_cost.csv),
  [teacher_stage_wall.csv](teacher_stage_wall.csv), [full_relation_space.csv](full_relation_space.csv):
  actual CUDA-event, synchronized wall, complete-stage and H=12 full-space costs.
- [predictor_cost.csv](predictor_cost.csv): single-relation/whole-pool latency,
  parameter counts, weight bytes and explicitly scoped linear FLOPs.
- [continuous_sensitivity.csv](continuous_sensitivity.csv): both tasks refined
  continuously on the same predetermined 32 test inputs.
- Figure PDFs: [oracle/cost](figure1_oracle_cost.pdf),
  [predictability](figure2_predictability.pdf), [oracle gap](figure3_oracle_predictor_gap.pdf).
- [verification.json](verification.json), [input_exclusion_audit.json](input_exclusion_audit.json),
  [environment.json](environment.json), [source_manifest.json](source_manifest.json):
  identities, historical exclusion, immutable backbone, selected models and code.

## Protocol qualifications

The main 39 models were locked before generating test teachers. Twelve
additional models provide an exact grid-intervention-budget control; they were
declared before manually inspecting test metrics, after the first test program
had started, and have their own lock. Both original lock files are preserved.
See [original protocol](../../docs/COALITIONAL_CFS_PROTOCOL.md) and
[budget amendment](../../docs/COALITIONAL_COST_CONTROL_AMENDMENT.md).

Equal estimated GPU-path cost is not literal identical wall time for each
sparse training-label subset. The separate intervention-count control matches
31 versus 961 candidate evaluations per label, with less than 1% integer
rounding. Shared dense/source-drop costs are reported separately. First-call
setup costs are retained in raw benchmarks and excluded from warm estimates.

Selectors use oracle alpha at evaluation; they do not implement a complete
forward-only compensation system. Candidate pools123/2027 evaluate the same
fixed models on alternative relationships, rather than independently retrained
models. Oracle-Pairwise-information controls are privileged diagnostics, not
free inference features. All unfavorable results and unbounded gain ratios
remain in the data.

Validation: 72 local tests and 4 new tests in the production environment passed.
Input path/content exclusion was independently rechecked against all4,864
historical routing/confirmation images. Original CFS/routing/scaling files
were left unchanged. Large weights and raw images are backed up outside Git.
