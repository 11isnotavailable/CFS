# Protocol 04: Forward-Only Dynamic CFS Router

> Router implementation and evaluation scripts are included in this repository;
> compact summaries are available in `results/section6/`.

## Goal and scope

Test whether a low-cost forward-only router can exploit CFS under a fixed
logical head budget. The final protocol uses frozen ViT-B AugReg on Imagenette,
layer 11, and budgets \(K\in\{3,6,9\}\) out of 12 heads.

The primary application is **hard logical routing**. It does not claim measured
hardware speedup because the current timm backend still executes dense fused
QKV and output-projection kernels.

## Data and leakage control

- Pairwise and joint teacher training: 4,096 images.
- Checkpoint selection: 64 disjoint tune images.
- Final reporting: 448 held-out images.
- Router seeds: 17, 23, and 41.
- Frozen backbone; only router parameters are optimized.
- Router training: 60 epochs, batch 64, AdamW, learning rate `3e-4`, weight
  decay `1e-4`.

## Runtime information

The router observes only the target layer's pre-attention residual:

\[
z=[X_{\mathrm{CLS}};\operatorname{Mean}(X_{\mathrm{patch}});
\operatorname{Var}(X_{\mathrm{patch}})].
\]

Train-set normalization statistics are frozen. The router receives no Q/K/V,
head output, downstream activation, gradient, label, or oracle score at
inference time.

## Model

- Global encoder: `2304 -> 256 -> 256`, GELU, dropout 0.1.
- Learned 32-dimensional identity embedding per head.
- Pair feature: `[E(z), e_i, e_j, abs(e_i-e_j), e_i*e_j]`.
- Pair MLP: `384 -> 256 -> 128`.
- Heads: `tanh` CFS score, `3*sigmoid` alpha, and source-importance logit.
- Total trainable parameters: approximately 0.825M.

## R1: structured pairwise supervision

Optimize the weighted sum of:

1. listwise representative cross-entropy;
2. pairwise CFS Smooth-L1, weight 0.5;
3. cross-entropy to normalized deletion KL, weight 0.5;
4. target-weighted oracle-alpha Smooth-L1, weight 0.1;
5. exact structured subset-ranking loss at all three budgets, weight 1.0.

Pair target temperature is 0.1 and subset temperature is 0.05.

## R2: direct joint-subset supervision

Retain every R1 loss and add a joint-set ranking loss with weight 0.5. For each
training image and budget, evaluate 48 simultaneous hard masks. Candidate pools
contain pairwise-CFS and router anchors, deletion oracle, Taylor, magnitude,
current-L2, static-CFS, one-swap hard negatives, and random subsets. Convert
standardized negative true KL into target preferences at temperature 0.5.

R2 is the final model because pairwise substitutions need not compose through
the nonlinear downstream network. Select its checkpoint by joint-set regret on
the tune split, never on the held-out test split.

## Exact subset selection

For each input enumerate all size-\(K\) subsets and maximize predicted
structured utility. With 12 heads, candidate counts are 220, 924, and 220 for
budgets 3, 6, and 9. Report that selection is exact rather than greedy.

## Evaluation

Compare on identical held-out examples and budgets:

- oracle joint subset;
- R2 learned router;
- scalar Taylor;
- magnitude;
- current-space similarity;
- static CFS;
- random subsets.

Report downstream accuracy, dense-reference KL, logical head budget, regret to
oracle, and paired uncertainty across examples/seeds. Verify online residual
summaries reproduce offline-selected masks. Predicted-alpha and oracle-alpha
merging are attribution ablations; hard gating remains primary.

