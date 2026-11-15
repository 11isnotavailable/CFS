# Pairwise vs Joint-2, frozen before formal collection (2026-11-16)

Original ViT-B AugReg checkpoint, last attention block11, 12 heads; no backbone
training or changes to existing CFS/routing/scaling. Original temperature1
KL(dense || intervened), denominator max(Ddrop,1e-12), valid Ddrop>1e-5.
Main oracle uses the original alpha grid: 31 equally spaced points on [0,3].
Joint-2 uses the exact Cartesian grid, with retained previously measured
pairwise feasible solutions to guard only against floating-point batch noise.
Alpha1 means unchanged representative; both alphas1 mean source deletion.

Eight TRAIN-input benchmarks found batched exact 31x31 grid (~0.036s/input)
faster than 60/120-step gradient optimization (~0.074/0.148s/input). Therefore
the grid is feasible and avoids local-search failures; this is not a slow
serial 961-forward implementation. Continuous optimization improves recovery
slightly beyond the discrete grid, so a separate BOTH-task continuous-refinement
sensitivity is required on 32 predetermined test inputs. Do not call the
31-grid result an exact continuous optimum. No grid resolution is reduced.

FP32 backbone/suffix and KL, TF32 disabled, same for both methods. This precision
change from historical FP16 is explicit. Historical training labels used16
alphas and old joint-subset caches concern hard deletion, not this substitution
question; they cannot supply the required matched labels. Reuse weights,
preprocessing, data split and cached final-block computation instead.

Frozen train1024 subset of original4096, tune64 original routing validation,
fresh test256, seed20261116. Test excludes all4608 original routing images plus
256 later confirmation images by path AND content hash. No assertion of public
images never appearing in unrelated experiments. Data manifests precede labels.

Per-source candidate pool: uniformly chosen four non-source heads, shared across
inputs/tasks, seed42, independent of any labels. Pairwise4 and Joint2 six
relations/source. Additional test pools123/2027 are a sensitivity analysis of
the same fixed predictors, not extra independent inputs or retraining seeds.

Predictors adapt original PairScorePredictor level0: same2304 pre-attention
[CLS,patch mean,patch variance], global256 encoder, source/candidate embeddings16,
256 scorer and dropout. Symmetric candidate mean/absolute difference represents
sets; singleton duplicates the candidate. Exactly equal trainable parameters.
Original predictor was ranking-only, not calibrated CFS regression. This study
uses standardized-target MSE for BOTH tasks, with raw decoded scores for NMSE,
and reports ranking/recovery separately. No posthoc clipping of predictions.
Training-set feature normalization and budget-specific target mean/std only.

Label budgets2500/5000/10000/20000: nested, uniformly permuted valid relation
indices, separate tasks, fixed sampling seed. Same supervised-label number is
not the same number of images; save and report unique training-image counts.
Three optimizer seeds17/23/41; AdamW3e-4,decay1e-4,batch128,2000 steps per fit.
Validation every250steps; choose minimum validation MSE, earlier ties. Fixed
training steps for all tasks and budgets. No test-directed configurations.

Cost-matched curve: keep Pairwise budgets above, set Joint2 count=floor(pair
count * warm batched per-label pair time / warm batched per-label joint time).
Use medians of benchmark inputs1..7, excluding first-call setup. This is a
standardized AMORTIZED suffix-GPU-cost budget estimate, not a claim each sparse
label subset was recollected in exactly equal wall time. Report actual collection
times, serial/batched costs, shared dense capture cost, and exact forward counts
alongside this estimate. All labels are generated once and subsampled offline.

Evaluation: raw CFS labels, valid sources, uniform source mean within independent
input; no head/coalition pseudo-replication. NMSE is aggregate model MSE divided
by test MSE of that budget's TRAIN constant. Regret normalized by corresponding
random-choice regret, as ratio of aggregate means. Source Spearman(undefined
flat rows=0, rate reported), best-hit, top2 recall, random baselines, oracle,
selected recovery, regret and joint-minus-pair gain. Oracle-alpha condition is
explicit: selectors do NOT predict coefficients and are not complete deployed
forward-only substitution systems. 2000 document bootstraps, seed42, seeds
averaged within input, per-seed SD separately. Comparisons are exploratory,
without multiplicity correction.

D2: true-Pairwise max/mean and a small symmetric Pairwise-only regressor trained
on the same Joint2 labels; clearly privileged information-content diagnostics
with additional pair-oracle measurements. Also evaluate max/mean of predicted
Pairwise scores as forward-only selection diagnostics (still oracle-alpha).
No claim these different-information comparisons isolate causal scientific value.

All outputs in independent run root, resumable per-input teachers, immutable
original data. Final deliverables include raw data/weights/logs, audit, three
figures, tables, paper-ready text, and unfavorable findings without selection.
