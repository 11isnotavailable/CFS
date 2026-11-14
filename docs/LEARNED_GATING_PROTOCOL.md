# Frozen plan: learned head gating (2026-11-15)

This plan is fixed before any new held-out evaluation. No loss/width/temperature
search, no test-based tuning, no backbone training, no teacher recomputation.

## Audited artifacts

Original routing uses Imagenette2-160, ViT-B AugReg checkpoint
`assets/vit_scaling/base.pth`, last attention block 11, 12 heads. It does **not**
use the ImageNet scaling/MLP experiment's inputs. Existing train/tune/test counts
are 4096/64/448. Export exact cached FP16-quantized prehead summaries as FP32;
freeze original train mean/std. Summary is CLS, patch mean and population
variance at the attention wrapper input (after block norm1 in this ViT).
Use original checkpoint, preprocessing, frozen paths and ImageNet class targets.

Original R2 has 825091 parameters, R1's pairwise/deletion/alpha/structured losses
plus 0.5 joint-set ranking. Existing 48-mask pools per input and K include
CFS/R1 anchors, Taylor and other baselines, hard negatives and random masks.
Those candidate pools may favor their anchor generators; all new learners use
the identical cached pools. Gating does not receive anchor identity, pairwise
labels or oracle ordering. Raw KL is retained without modifying the archive.

## Predeclared models and training

* Original R2: reuse seeds 17/23/41, unchanged.
* Gating: 2304 -> 304 -> 384 -> 12, GELU, first-layer dropout 0.1;
  822460 parameters. Sum head scores to score a subset; stable Top-K deployment.
* Joint-only R2 structure: same original architecture, freshly initialized,
  trained on exactly the same joint KL as gating. Alpha output is unused/frozen
  (129 parameters); 824962 optimized parameters. This is an explicitly labelled
  supervision control, **not** a redefinition/replacement of original R2.

One model supports K=3,6,9. Seeds 17/23/41; AdamW lr=3e-4, decay=1e-4,
batch=64, 60 epochs, no schedule, no early termination. Both new models use
0.5 times listwise cross entropy; targets softmax(-standardized KL/0.5),
population SD floored at 1e-6, predicted utilities divided by 0.05. All
normalization and comparisons stay within one input and budget. Choose first
checkpoint achieving minimum mean tune candidate-set KL regret, as original R2.
No held-out teacher is exported to or opened by training. Smoke test uses 64
training examples and two epochs, followed by fresh full training.

## Compensation and attribution

Primary unbiased shared protocol: same hard zero/one masks for all methods.
This is also the paper's conservative primary deployment. It has no learned
compensator and isolates selection at the stated logical budget.

Secondary original shared learned calibration: apply the **same** frozen R2
scores/alpha and `compose_scales` rule to Taylor, gating, original R2 and
joint-only R2 masks for each seed. This compensator was jointly trained with
original R2; shared application does not establish selection-neutral training.
Do not describe gating plus this compensator as a standalone CFS-free system;
its deployed system includes the additional original R2. Report original R2
merge against standalone gating hard masks separately, with the arithmetic
selection/compensation decomposition, without a causal attribution claim.

## Evaluation and uncertainty

One dense forward reference per input/batch reused by every policy, KL(dense||
routed), T=1, original FP32 log-softmax and nonnegative numerical clamp. Save
predictions, correct flags, masks/scales, dense logits and per-input metrics.
Re-run all policies on identical images/checkpoint/environment; compare original
R2/Taylor to historical CSV without merging historical/current measurements.

2000 paired input bootstrap draws, seed 42. Average seeds per image for pooled
curves and paired CIs; retain each seed separately and report seed SD. Seeds
are sensitivity repetitions, not independent images. Accuracy is ImageNet
top-1. Report no physical backbone speedup. Optional P1 deferred in favor of
finishing the specified P0 and supervision/compensation controls.
