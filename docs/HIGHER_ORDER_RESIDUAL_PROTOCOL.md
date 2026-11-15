# Frozen-router residual feasibility study

Frozen before new residual training/evaluation, 2026-11-15; parent a7c87f5.
Only direction A is implemented. No adapters, capacity penalties, backbone
training, changes to CFS, or additional pairwise intervention teacher.

## Data and estimand

The existing 38,016 measured quartets belong to tune/test inputs, not training
inputs. They must not become training supervision. Existing training masks
have only K=3/6/9 and do not identify background-conditioned pair interactions.
We therefore fit a **latent pair-decomposed joint-KL residual**, supervised
only by training joint KL. Its pair terms are not identified causal second
differences. This is a feasibility test of structured residual prediction,
not a claim to have learned the measured physical interaction of each pair.

Reuse 4,096 training inputs, 64 tune inputs and the previously inspected 448
test inputs. Split each input/budget's 48 unique training masks into 36 fit and
12 diagnostic masks using seed 42. Duplicates must be rejected. The frozen
routers previously saw these labels: this is residual-held-out-mask evaluation,
not wholly unseen data for the complete pipeline. Fit calibration, residual
normalization and residual weights only on the 36 masks. Tune selects epochs
and residual strength; diagnostic masks and old test never select anything.

Before training, lock 256 additional Imagenette validation images, seed
20261115, excluding canonical paths from all three old routing splits. These
are new confirmation inputs for this routing series, not a new dataset/domain.
Image bytes and frozen paths are archived. No confirmation model outputs are
computed until every candidate and selection rule is locked. Do not expand
the search if either old test or confirmation is unfavorable.

## Predictors and controls

All A3 and original R2 weights remain frozen. Primary base is A3. Original R2
and Learned Gating are unchanged evaluation references. Frozen A3 produces
its 256-dimensional encoder output, normalized predicted importance, and
directed CFS graph. Retain the original clipped structured utility U exactly.
Fit per-budget a,b to training KL, with b>=1e-8, for D0=a-b*U. This monotone
calibration cannot change the base ranking. Residual targets are (D-D0)/sd_K.

Three residual variants have exactly identical trainable parameter counts:

1. Plain: a set MLP on encoded input, retained mask, importance and budget;
   matching feature slots contain mask/importance summaries, not explicit CFS.
2. CFS: shared pair MLP for each unordered dropped pair, conditioned on input,
   full retained mask, importance, pair identities, budget, and eight directed
   CFS features (both directions, each source's max retained replacement,
   product/common coverage and cross-representative scores). Average the pair
   outputs. The explicit retained context prevents summing unrelated backgrounds.
3. Shuffled: the identical pair architecture and CFS features, with independently
   permuted graph head identities per input, fixed by split/seed. Preserve the
   graph's values and input-level distribution but destroy head alignment.

All variants share a 256->32 context projection and a 103->64->1 MLP.
Predict D_new=D0+gamma_K*sd_K*residual. Averaging P dropped pairs is equivalent
to a sum with coefficient gamma_K/P; it does not presume physical additivity.
Plain and shuffled controls receive the same frozen features/base utility.
Plain still has implicit CFS information through the frozen A3 encoder; the
ablation tests explicit graph structure, not absence of all CFS information.

## Fixed finite fit and selection

Seeds 17/23/41, each using its matching frozen A3 checkpoint. Three variants,
nine runs, 20 epochs each, AdamW lr=3e-4, decay=1e-4, batch 64 inputs. Each
step uses all 36 fit masks per budget; same order/initialization across variants
within seed. Equal-budget standardized KL MSE; no weight or architecture search.
Each variant chooses one common epoch across seeds by mean tune normalized MSE
(gamma=1), tie to earlier epoch. Store every epoch checkpoint until selection.
Then choose gamma_K from {0,.25,.5,1} by mean tune chosen-mask KL across seeds,
tie to smaller gamma, identically for every variant. Gamma=0 is exact A3 fallback.
Report all variants regardless of success. CFS is the prespecified proposed
method; promote only if tune K9 improves and K3/K6 do not worsen. Never choose
the method after test/confirmation. Validation has 48 masks/K; deployed routing
enumerates all 220/924/220 masks. Report this proxy mismatch explicitly.

## Evaluation

After locking, report mask-prediction MSE/MAE and chosen-mask regret, including
residual-held-out training masks. Exhaustively rank all candidate masks and
evaluate real hard gating on old test and new confirmation with the unchanged
ViT-B last-block intervention and KL(p_dense||p_routed). No compensation or
wall-clock acceleration claims. Save individual inputs/seeds/budgets/masks,
accuracy, parameter counts, training/inference times and selection details.
Use 2,000 paired input bootstraps (seed42), averaging the three seed results
inside each input, separate cohorts; include per-seed results. CIs conditional
on these seeds, unadjusted for multiple contrasts. Key contrast: CFS vs plain
and shuffled, not merely CFS vs A3. Old test is exploratory; confirmation is
not used for tuning. Negative and mixed findings remain in the report.
