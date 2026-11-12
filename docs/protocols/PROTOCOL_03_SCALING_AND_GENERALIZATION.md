# Protocol 03: Scaling and Cross-Domain Generalization

> Evaluation scripts and processed summary tables are included in this
> repository. Third-party model weights and datasets are not redistributed.

## A. Natural model-family trajectories

Natural scaling is a cross-section of public model families, not a
single-variable causal intervention.

| Family | Task and frozen evaluation slice | CFS layers | Positions/samples | Alpha grid |
|---|---|---|---|---|
| ViT Tiny/Small/Base/Large | ImageNet-1K classification | final block | shared 512 images | 31 points on [0,3] |
| BERT Tiny/Small/Base/Large | pretrained MLM on frozen MNLI text | final layer | 32 CFS texts; 8 masks/text | 16 points on [0,3] |
| Qwen2.5 0.5B–14B base | causal LM on frozen FineWeb-Edu slice | approx. 25/50/75% depth | 32 contexts; 8 deterministic positions/context | 16 points on [0,3] |

For every model report task loss/accuracy, parameter count, layer/head
metadata, head dimension, Raw Oracle, Matched-2, Cover/Head at \(\tau=0.5\),
Rank/Head, deletion KL, valid-source rate, and weighted recovery.

Important comparability limits:

- Raw Oracle grows with candidate-set size; interpret it jointly with
  Matched-2.
- Absolute KL is task-output-space dependent and should not be compared as if
  it had a common scale across classification, MLM, and causal LM.
- Qwen2.5 simultaneously changes depth, width, GQA structure, and head count;
  0.5B also uses 64-dimensional query heads while 1.5B+ use 128-dimensional
  query heads.
- Query heads, not KV groups, are the CFS units for GQA models.

## B. Controlled ViT scaling

Train from random initialization on the same frozen ImageNet-100 split using
the same optimizer, augmentations, seed, batch protocol, and 75-epoch budget.
Evaluate the same final-layer 512-image CFS probe set.

### Depth axis

Hold width 384, six heads, and head dimension 64 fixed; vary depth over
6/12/18.

### Width axis

Hold depth 12 and head dimension 64 fixed; vary `(width, heads)` over
`(192,3)`, `(384,6)`, and `(768,12)`.

### Head-granularity axis

Hold depth 12 and width 384 fixed; vary head decomposition over
`3x128`, `6x64`, and `12x32`. Parameter count and dense analytic MACs are
effectively fixed, so this axis isolates decomposition granularity rather than
overall capacity.

## C. Shared analysis

- Compute CFS per input before aggregation.
- Use deletion KL > `1e-5` as the valid-source threshold.
- Report Raw, Matched-2, Cover/Head, Rank/Head, weighted recovery, and absolute
  deletion/replacement KL together.
- Use exact subset enumeration for small head counts and exact MILP/set-cover
  solving where enumeration is infeasible.
- Do not claim that CFS explains a universal scaling law. The supported claim
  is that functional organization is an additional measured axis along natural
  and controlled scaling trajectories.

