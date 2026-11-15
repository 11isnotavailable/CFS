# Selection regret / candidate ranking diagnostic

Frozen scope, 2026-11-15, parent 3b56ba1. No model training, new teacher,
backbone forward, hyperparameter selection, or new confirmation/test evaluation.
Read existing train4096/tune64 features and their 48 measured masks per K.
Use frozen A3, original R2, Gating and the existing joint-only R2 (A0) control,
seeds17/23/41. Reuse exact original utility implementations and checkpoints.

Important correction: all four original routers already optimize the original
candidate-standardized listwise joint ranking loss (with differing auxiliary
objectives). The previous residual extension used MSE. A future ranking study
would test candidate construction/local decision emphasis, not introduce
ranking supervision for the first time.

## Measured candidate-pool diagnostics

For every input/seed/budget/method, save chosen KL, 48-mask oracle KL, regret,
pool range, chosen true rank, global Spearman and pair ordering accuracy,
ordering among the five true best candidates, best-vs-other comparisons,
and exact prediction ties. Ignore true KL ties within 1e-8 in pair accuracy;
score ties earn half credit. A computational tie tolerance is not a claim of
functional equivalence. Report exact decision regret without clipping it away.

For predicted shortlists of 1/3/5/10/20 candidates, save minimum true KL,
remaining regret and oracle retrieval. This is a label-using upper bound on
shortlist repair, not an implemented router. Compute removed regret as a ratio
of aggregate regrets, not the mean of unstable per-input ratios.
Distinguish near-top errors (chosen true rank2..5) from farther errors (>5),
including their shares of total regret. Good global ordering does not imply
good local decisions; poor global ordering does not isolate a local bottleneck.

Measure A3/R2/Gating choice disagreements and actual paired KL differences in
the SAME labelled pool. Report oracle selection among their three choices and
the union of their predicted top3/top5 shortlists. These are upper bounds, not
trained ensembles. A0 remains a separate structure/objective diagnostic.

## Candidate coverage audit

Score the full 220/924/220 candidate sets without backbone inference. Record
whether each deployed full-search choice is in the labelled pool; only then
attach its known KL. Do not impute KL for unmeasured choices or infer global
regret from a 48-mask oracle. Record full A3/R2 choice disagreements and export
all training candidates with known/unknown-label status for possible future
hard-negative collection. No such collection is launched in this task.

## Statistics and interpretation

Train is in-sample for all four routers; tune was used for checkpoint selection.
Neither is a fresh generalization test. Use paired input bootstrap, 2,000 draws,
seed42, average training seeds within input. Save per-seed means and input SD.
Keep train/tune and budgets separate, no independent mask/pair resampling.
Report both favorable and unfavorable findings and all controls.

Recommend whether a small hard-candidate measurement diagnostic is warranted
using observed regret, shortlist coverage and actual candidate disagreement.
Do not treat a positive oracle upper bound as proof a learned selector will
achieve it, or missing candidate labels as proof of distribution-shift harm.
Do not start decision-focused training, cross-model experiments or rebuttal
rewriting without the subsequent task that chooses among those directions.
