# Protocol 05: Functional Organization During Training

> Checkpoint-analysis and visualization scripts are included in this repository;
> compact trajectory tables are available in `results/section5/`.

## Question

How does conditional functional organization emerge as a Transformer is
optimized from random initialization?

## Fixed trajectory

- Model: canonical ViT-S/16, depth 12, width 384, six 64-dimensional heads.
- Task: ImageNet-100 classification.
- Training: one fixed seed and recipe, epochs 1 through 100.
- Layers: 3, 7, and 11 (zero-based).
- Probe set: the same frozen 512 images at every epoch.
- Alpha grid: 16 uniformly spaced values on `[0,3]`.
- Valid-source threshold: deletion KL > `1e-5`.
- Cover thresholds: 0.25, 0.50, and 0.75.

Epoch 0 is excluded from the canonical database; epochs 1–100 are included.

## Functional snapshot schema

For every epoch and layer save:

- raw per-input directed `S` matrices;
- raw replacement KL matrices;
- deletion KL vectors;
- optimal-alpha matrices;
- best representatives;
- validity masks;
- threshold adjacency matrices;
- head contribution energy.

For every epoch save recognition accuracy/loss and aggregate Raw Oracle,
Matched-1/2/4, weighted recovery, mean deletion KL, mean best-substitution KL,
Cover/Head at all thresholds, and Rank/Head.

Raw arrays must be retained so thresholds, normalization, and graph statistics
can be recomputed without rerunning model interventions.

## Evolution analyses

1. Plot task performance and functional metrics against epoch.
2. Compare each epoch's flattened valid CFS matrix with epoch 100 using
   Spearman correlation.
3. Measure top-1 best-representative agreement with epoch 100.
4. Track each source head's ranked functional signature over time.
5. Visualize top-1 or top-2 directed substitution edges at selected epochs;
   use deletion KL for node size and do not encode alpha as edge existence.
6. Report layer-wise trajectories rather than silently pooling all sources.

## Interpretation safeguards

- Stable functional graphs before accuracy convergence indicate temporal
  ordering, not proof of causal necessity.
- Falling Oracle can reflect increasing branch independence, but must be
  interpreted with deletion KL, replacement KL, coverage, rank, and accuracy.
- Changing CFS must not be attributed to reorganization if every head merely
  becomes weak; absolute KL and energy controls are mandatory.
- Head indices remain fixed across checkpoints, but index identity alone does
  not guarantee unchanged semantic role.

