# Protocol 02: Observability Ladder

> Collection and analysis code is available in `scripts/`; compact summaries
> are available in `results/section6/`.

## Question

At what computational stage, and from what information, does conditional
functional substitutability become observable?

## Frozen development design

- Backbone: DeiT-S/16; primary layer 10 (zero-based).
- Teacher: directed CFS over a 31-point alpha grid.
- Probe train set: 4,096 images.
- Endpoint-selection tune set: 128 disjoint images.
- Frozen final ladder test: 512 further disjoint images.
- Probe seeds: 17, 23, and 41.
- Shared pair-score model capacity, optimizer, schedule, and checkpoint rule.
- Target: complete directed pair score \(\hat S(x,i,j)\), with representative
  selection derived by argmax.

No image identity or augmentation variant may cross the split boundary.

## Information levels

| Level | Added information | Interpretation |
|---|---|---|
| L0 | residual summaries | global state before head computation |
| L2 | token-wise Q/K moments | head-specific query/routing geometry |
| L3 | attention-logit statistics | routing-pattern information |
| L4 | V summaries | routed content statistics |
| L5 | \(A_iV_i\) | aggregated head content |
| L6 | post-\(W_O\) contribution | intervention-aligned forward contribution |
| L7 | signed and absolute gate gradients | downstream sensitivity ceiling |

High-dimensional head-stage features use a fixed, non-learned projection.
Each stage is evaluated under the same split and probe capacity. Gradient
features are an analysis ceiling, not a forward-only deployment signal.

## Baselines and outcomes

Report mean achieved CFS recovery, best-representative accuracy, pairwise
Spearman correlation, random representative, static representative, scalar
Taylor representative, and functional oracle. Bootstrap paired differences at
the image level.

## Frozen replications

1. Repeat selected endpoints L0, L2, L7, Taylor, and oracle on ViT-S AugReg
   without reselecting the best level.
2. Repeat early/middle/late layers with identical splits.
3. For text, retain residual, Q/K, post-output-projection contribution,
   gradient sensitivity, Taylor, and oracle endpoints under the same
   intervention semantics.

The protocol measures observability, not the causal time at which a function
“forms.” Compressed summaries also do not prove that full tensors are
insufficient.

