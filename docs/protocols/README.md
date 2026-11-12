# Conditional Functional Substitutability: Protocols

This directory is the curated protocol package for the CFS project. It
contains the definitions, algorithmic pseudocode, and principal experimental
protocols needed to understand and reproduce the paper's measurements. It is
deliberately organized around scientific questions rather than the chronology
of exploratory experiments.

The accompanying repository contains the author-owned implementation, tests,
analysis scripts, and compact paper-result tables. Third-party datasets and
pretrained checkpoints are referenced through their original providers and
remain subject to their licenses.

## Contents

| File | Purpose |
|---|---|
| `CFS_CORE_ALGORITHM_PSEUDOCODE.md` | Canonical definition and pseudocode for CFS measurement, aggregation, graph metrics, and the learned router. |
| `PROTOCOL_01_DISCOVERY_AND_VALIDATION.md` | Falsification-first discovery, controls, calibration decomposition, conditionality, and directionality. |
| `PROTOCOL_02_OBSERVABILITY.md` | Frozen information ladder from forward representations to downstream sensitivity. |
| `PROTOCOL_03_SCALING_AND_GENERALIZATION.md` | Cross-family natural scaling and controlled depth/width/head-granularity experiments. |
| `PROTOCOL_04_DYNAMIC_ROUTER.md` | Final forward-only structured CFS router, teacher construction, training, and evaluation. |
| `PROTOCOL_05_TRAINING_DYNAMICS.md` | Epoch-by-epoch functional-organization evolution protocol. |
| `REPRODUCIBILITY_AND_RELEASE.md` | Shared invariants, leakage controls, reporting rules, and planned release contents. |

## Canonical interpretation

CFS asks whether the downstream effect of removing branch `i` can be recovered
by recalibrating another branch `j`, conditional on the current input. It is
not a representation-similarity score and it is not identical to single-head
importance. The directed score is

\[
S_{i\rightarrow j}(x)
=1-\frac{D^*_{i\rightarrow j}(x)}
{\max(D_i^{\mathrm{drop}}(x),\varepsilon)}.
\]

The score must be computed per input before aggregation. Sources with deletion
KL at or below `1e-5` are excluded from normalized scientific summaries.

## Claim boundary

The protocols support claims about conditional functional organization,
observability, scaling trajectories, and logical head allocation. Unless a
sparse execution backend is separately demonstrated, logical head budgets and
masked dense execution must not be described as measured wall-clock speedups.

