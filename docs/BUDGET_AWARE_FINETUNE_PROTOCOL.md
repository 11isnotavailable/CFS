# Frozen plan: budget-aware A3 fine-tuning

2026-11-15, baseline commit60bf8a7. Only proposal1 is run. No ensemble,
architecture change, CFS redefinition, backbone update or teacher recollection.
Known test targets are reporting criteria, never checkpoint selection criteria.

Four relative weight sets: (1,1,1), (1,1,1.5), (1,1,2), (1,1,3). Three seeds
17/23/41, each initialized from its validation-selected A3(lambda=.01)
checkpoint. Ten epochs, batch64, AdamW lr3e-5 (one tenth original), decay1e-4,
constant learning rate. The old checkpoints lack optimizer states, so AdamW
moments are reset uniformly. No search over learning rate, length or schedule.
Weight1 is a continuation control, not a new untreated baseline.

Objective:
0.5 * [(L3+L6+w9*L9)/(2+w9) + .01*(L_CFS+.1*L_alpha)].
Normalize the joint weights so their sum is1; the overall objective/auxiliary
scale stays comparable when w9 increases. All constituent losses, train inputs,
candidate masks, targets and original feature normalization are reused.
Budget weighting changes relative optimization, not the CFS definition.

First measure the actual starting checkpoints on the fixed first64 training
inputs: raw budget losses, common-trunk gradient norms/cosines and weighted
contribution norms. Repeat at epochs1/5/10 without optimization on the probe.
Earlier trajectory diagnostics suggest budget conflict, not an inherently
smaller K9 gradient. Do not claim starvation from loss magnitudes alone.

## Validation lock before new test evaluation

Record each budget's candidate-set KL regret on the existing64 tune inputs at
epochs0–10. Average the three seed regrets for each common epoch and weight.
An eligible candidate must have mean tune K3 and K6 regret no greater than
the corresponding starting A3 means (numerical tolerance1e-8), and K9 strictly
lower. Select minimum K9 among eligible candidates, tie-break mean regret,
smaller weight, earlier epoch. This produces ONE weight and ONE epoch shared
by all seeds. If none qualifies, retain original A3 as primary; explicitly
report failure to find a validated non-degrading update.

For transparency also lock a representative for each of the four weights:
use the same constrained rule if feasible; otherwise the minimum-K9 epoch,
clearly marked infeasible. Such diagnostics do not replace the primary fallback.
Publish every validation trajectory and feasibility flag. No test-driven
configuration changes, additional search or individual-seed picking.

## Evaluation and uncertainty

Only after the selection JSON exists, evaluate all four locked representatives
on the original448 inputs with shared hard masks. Reuse the existing evaluator,
same dense reference, and report the unchanged original R2, Gating and A3.
Retain shared-original-R2-compensation results as secondary, not the selection
criterion. Each input has three seed results; average within input then run
2000 paired input bootstrap replicates, seed42. Report per-seed means and SD.
Intervals are unadjusted and conditional on three seeds; do not call a point
estimate win statistical superiority or unresolved differences equivalence.

Final acceptance is descriptive: KL3<=.10517, KL6<=.02480, KL9<.00700, along
with exact differences to original unrounded A3 and original R2. Test failure
does not authorize more tuning. Proposal2 fusion remains unrun this round.
