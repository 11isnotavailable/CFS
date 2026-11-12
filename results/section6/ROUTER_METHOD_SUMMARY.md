# Final CFS Router Method Summary

## Scope

The final application is a forward-only router for the final attention layer of
`vit_base_patch16_224.augreg_in21k_ft_in1k` on Imagenette. It uses a fixed
logical head budget and predicts a sample-conditioned representative subset.
The primary result is hard routing; calibrated merging is an attribution
ablation.

## Runtime input

For the pre-attention residual tensor (X_l\in\mathbb{R}^{N\times d}), the
router receives only

\[
z_l(x)=\left[X_{l,\mathrm{CLS}};
\operatorname{Mean}(X_{l,\mathrm{patch}});
\operatorname{Var}(X_{l,\mathrm{patch}})\right].
\]

For ViT-B, this is a 2,304-dimensional vector. Its mean and standard deviation
are estimated on the 4,096 training images and frozen. No Q/K/V tensor, head
output, downstream activation, gradient, label, or oracle score is available
at inference time.

## Network

- Global encoder (E): `2304 -> 256 -> 256`, with GELU after both linear
  layers and dropout 0.1 after the first GELU.
- Learned head identity embeddings: (e_i\in\mathbb{R}^{32}), one for each of
  12 heads.
- Pair feature:

\[
\phi_{ij}(x)=
[E(z_l(x));e_i;e_j;|e_i-e_j|;e_i\odot e_j].
\]

- Pair MLP: `384 -> 256 -> 128`, GELU activations and dropout 0.1 after the
  first GELU.
- Score head: scalar followed by `tanh`, producing
  (hat S_{i\rightarrow j}\in[-1,1]).
- Alpha head: scalar followed by (3\operatorname{sigmoid}(\cdot)), producing
  pairwise (hat\alpha_{i\rightarrow j}\in[0,3]).
- Importance head: `[E(z); e_i] -> 128 -> 1`, producing head logits
  (hat w_i).
- Total trainable router parameters: approximately 0.825M. The ViT backbone is
  frozen.

## Structured subset utility

For a retained set (R), predicted importance is normalized with a softmax and
the router scores

\[
\hat U(R\mid x)=
\sum_i \hat w_i(x)
\begin{cases}
1,&i\in R,\\
\max_{j\in R}\operatorname{clip}(\hat S_{i\rightarrow j},0,1),&i\notin R.
\end{cases}
\]

For each budget (K\in\{3,6,9\}), all subsets are enumerated exactly:

\[
R^*(x)=\arg\max_{|R|=K}\hat U(R\mid x).
\]

The candidate counts are 220, 924, and 220, respectively. Thus all three
reported budgets use exact subset selection, not greedy search.

## R1 supervision

R1 uses the 4,096-image pairwise CFS teacher and combines:

1. listwise cross-entropy over candidate representatives for every source;
2. Smooth-L1 regression on pairwise CFS scores, weight 0.5;
3. cross-entropy to normalized single-head deletion KL, supervising
   (hat w_i), weight 0.5;
4. target-distribution-weighted Smooth-L1 on pairwise oracle alpha, weight 0.1;
5. structured subset-ranking cross-entropy over every exact subset at all three
   budgets, weight 1.0.

The target pair temperature is 0.1 and subset temperature is 0.05. Therefore
the final R1 checkpoint is already a **structured pairwise-supervised router**;
it is not a pure pair-regression-only baseline.

## R2 joint-subset supervision

Pairwise substitutions need not compose through the nonlinear downstream
network. R2 retains every R1 term and adds direct joint-set supervision.

For each image and each budget, the teacher evaluates 48 candidate masks using
simultaneous hard gating and records their true downstream KL to the dense
model. Candidate pools contain pairwise-CFS and R1-router anchors, deletion
oracle, Taylor, magnitude, current-L2, static-CFS, one-swap hard negatives, and
random subsets.

Within every candidate pool, the true KL values are standardized and converted
to a target preference distribution using negative KL and temperature 0.5.
Predicted subset utilities use temperature 0.05. Their cross-entropy is added
with weight 0.5. R2 checkpoint selection minimizes joint-set regret on the
disjoint tune set.

The joint teacher contains 4,096 training images and 64 tune images, 48
candidates per budget, for all (K\in\{3,6,9\}).

## Data split and seeds

- Pairwise/joint teacher training: 4,096 Imagenette training images.
- Checkpoint selection: 64 disjoint tune images.
- Final reporting: 448 held-out images.
- Router seeds: 17, 23, and 41.
- Backbone: frozen ViT-B AugReg, layer 11.
- Router training: 60 epochs, batch size 64, AdamW, learning rate (3\times
  10^{-4}), weight decay (10^{-4}).

## Calibration and execution

The primary R2 result applies the predicted mask as a hard gate; predicted
alpha is not required. The calibration ablation assigns each dropped source to
its highest-scoring retained representative and composes pairwise scales as

\[
s_j=1+\sum_{i\mapsto j}(\hat\alpha_{ij}-1),
\]

clipped to `[0,6]`. A second control replaces predicted alpha with oracle
pairwise alpha for both CFS-selected and Taylor-selected sets.

Online execution recomputes the same residual summary inside the target layer
before attention and agrees with frozen offline masks on 447/448 images at each
budget.

## Compute claim boundary

The reported quantities are logical head budgets and theoretical structural
allocation. The current timm backend retains fused dense QKV and output
projection kernels. Hard masks therefore do not establish wall-clock sparse
speedup, and the measured online times are correctness diagnostics only.

