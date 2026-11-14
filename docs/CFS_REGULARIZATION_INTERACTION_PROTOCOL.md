# Frozen plan: CFS regularization and non-additive interactions

Date: 2026-11-15; baseline commit b494b13. This protocol is fixed before new
validation pilots or interaction measurements. All existing test results are
already known; no new test result will select a loss, lambda, seed or threshold.

## Part A

Reuse original architecture, 4096/64/448 split, feature normalization, 48-mask
joint teachers, AdamW (3e-4,1e-4), batch64, 60 epochs, seeds17/23/41. A0 and A4
reuse checkpoints. L_joint is the original listwise joint-set loss. L_CFS is
original representative listwise CE plus 0.5 original score SmoothL1. L_del is
original normalized-deletion CE; L_alpha is original representative-probability
weighted alpha SmoothL1. Preserve original target clipping and diagonal rules.
Do not include the original exhaustive structured-subset auxiliary in A1-A3:
that target also depends on deletion weights and would confound the ablation.

New objective: 0.5 * [L_joint + lambda*(L_CFS + d*0.5*L_del + a*0.1*L_alpha)].
Here (d,a)=(0,0),(1,0),(0,1) for A1,A2,A3. Overall0.5 preserves A0 optimizer
scale. Report raw loss/gradient norms; do not retrospectively rescale losses.
Alpha output is optimized only in A3; otherwise frozen, as in A0.

Seed17 pilots: all A1/A2/A3 at lambda=.01,.03,.1,.3,1 (15 runs). Lambda0
is the existing A0. Choose one lambda per variant by minimum mean tune regret
over all three K, first checkpoint on ties, smaller lambda on equal regret.
Freeze those choices; train seeds23/41 for each (six more runs). Report all
pilot validation results and all three full variants, regardless of sign.
Choose a single headline configuration by seed17 validation before any new
held-out evaluation. No optional gradient surgery or additional search is
preauthorized by this plan; diagnose conflicts without assuming causation.

On a fixed first64 training-input probe, eval mode at epochs0,1,10,30,60,
measure losses, gradient norms and pairwise gradient cosines on common global
encoder + head embeddings. Include every K-specific joint gradient. This is a
diagnostic probe, not an optimization update. Save original R2 and A0 endpoint
diagnostics for context. Validation records include each budget separately.

Evaluate selected variants alongside every prior baseline with identical hard
gates, shared original-R2 compensator, and full-pipeline attribution. Extra
pair labels are not free: report existing collection grid16, H(H-1)=132 pairs
and12 deletions/input, plus possible implementation overhead; no new pair
teacher computation. Bootstrap2000 independent inputs, seed42, average router
seeds per input; also show seed-specific statistics.

Artifact audit clarification: the actual train cache uses16 alpha grid points;
tune/test shards use31, all on[0,3]. New optimization uses train pair labels
only; tune checkpoint selection uses joint KL. This preserves the actual
cached protocol, including its train/evaluation oracle-grid difference.

## Part B

B1 offline: all4096 training and64 tune inputs with existing48 masks/K. Dedup
identical masks before 4-fold mask cross-validation. Fit per-input/per-budget
centered least-squares using pseudoinverse (minimum-norm, zero-sum gauge);
intercept reconstructed from training means. Record rank, held-out RMSE/MAE,
training RMSE and constant-mean held-out RMSE. Do not interpret prediction
error alone as a mathematical proof of non-additivity. No labels borrowed
between folds. Existing exhaustive-oracle files saved minima, not full KL.

Limited new measurements: all64 tune inputs and128 test inputs drawn uniformly
without replacement with RNG42 before inspecting any new interaction result.
At each K=3,6,9 and each of66 unordered pairs, choose exactly one background T
of size K-2 from the other10 heads, with deterministic seed42. Evaluate the
four actual masks T,T+i,T+j,T+i+j. Never infer an unobserved mask KL.
Also evaluate48 uniform unique K-masks/input for held-out-mask B1/B4 on this
subset. Cache last-block input and replay the unchanged attention/MLP suffix;
smoke-check against full-forward masks on first4 tune inputs. Freeze backbone.
Keep dense logits, unique masks, raw KL, all quartet indices and input identity.

I=D(T+i+j)-D(T+i)-D(T+j)+D(T). Positive I means adding j increases i's
marginal KL change (diminishing KL reduction when both individually help);
negative I means decreasing marginal KL change. These names are conditional:
KL is a nonmonotone discrepancy, not task reward, and signs alone do not prove
semantic redundancy/synergy. Report both signed I and |I|, within each K.

B3: use both-source-valid original CFS pairs (drop>1e-5), report excluded rate.
Show directional scores, bidirectional mean/max/asymmetry, contribution cosine
and Taylor controls. Primary regressors are fixed ridge(alpha=1) standardized
on64 tune inputs, tested on128 disjoint inputs, separately per K. Compare
Taylor+cosine+deletion controls, CFS-only, and controls+CFS (symmetric mean,
max,absolute asymmetry; signed log1p transform for unbounded CFS tails).
Predict signed I and |I| separately. Fit target scaling on tune only. Report
held-out error/sign accuracy, and paired input-bootstrap incremental MSE.
All regression arms additionally receive pair identity (66 indicators) and
the actual background mask (12 indicators), controlling static pair/context
identity. These feature details were fixed before any B3 regression results.
No test hyperparameter selection; pairs from one image are never independent
bootstrap units. Within-input Spearman correlations averaged/bootstrapped over
inputs are primary correlation summaries; no pair-pooled pseudo-replication.

B4: input complexity = mean |I| across66 pairs, separately per K (all actual
interventions, not validity-filtered). Fix high/low thresholds at tune medians.
Report continuous Spearman complexity vs gating error and gating-minus-router
KL; report low/high on128 measured test inputs and ungrouped full448 routing
results. Also show held-out-mask additive residual vs conditional advantage.
These retrospective associations with already-frozen routing are analytical
oracle diagnostics, not runtime-accessible gating or causal mediation.

All intervals unadjusted95%, no universal significance or wall-clock speedup
claim. Archive all negative findings and all budgets. No backbone training,
no scaling experiment, no overwritten historical result.
