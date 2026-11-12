# Protocol 01: Discovery and Validation

> The corresponding implementation is available in `src/` and `scripts/`.

## Scientific question

Do attention heads exhibit input-conditioned downstream substitutability that
cannot be reduced to head importance, current-space similarity, or scalar
calibration alone?

## Stage A: falsification-first discovery

- Backbone: pretrained DeiT-S/16.
- Unit: post-output-projection attention-head contribution.
- Intervention: single deletion, pair deletion, and directed scalar
  substitution over a fixed alpha grid.
- Outcome: reference-to-intervention KL on final predictive distributions.
- First-pass scope: all layers on a small frozen image subset; no learned
  predictor is trained.
- Required implementation check: all-one gates reproduce dense logits within
  numerical tolerance.

Persist per-sample deletion KL, pairwise replacement KL, optimal alpha, CFS
matrix, and head contributions needed for current-space controls.

## Stage B: alternative-explanation controls

Evaluate the following against the same directed source-candidate pairs:

1. normalized current-contribution L2 distance;
2. contribution cosine distance;
3. attention-map similarity;
4. magnitude/energy;
5. scalar Taylor source importance;
6. random representative;
7. best fixed representative per source;
8. conditional representative with fixed alpha;
9. random representative with oracle alpha;
10. conditional representative with oracle alpha;
11. a zero-information/dummy control when applicable.

The central decomposition separates representative selection, contribution
calibration, and input conditioning. An oracle-alpha result alone is not
evidence of substitutability.

## Stage C: conditionality and directionality

- Compare best-representative overlap across different images.
- Compare CFS-matrix Spearman correlation for same-image augmentations versus
  unrelated images.
- Quantify \(|S_{i\rightarrow j}-S_{j\rightarrow i}|\), reciprocal edge rates,
  and strong one-way edge rates under a frozen threshold.
- Treat directionality as a measured tendency, not an assumed equivalence
  relation.

## Stage D: subset oracle

For small head counts, enumerate every retained subset at fixed budgets and
evaluate true downstream KL. Compare exact best-subset selection with random,
magnitude, Taylor, current similarity, static CFS, and learned routing.

## Reporting

Report both normalized recovery and absolute KL. Include uncertainty over
input examples, disclose all thresholds, and keep discovery/tuning examples
disjoint from final evaluation examples. Claims should read “current
representation similarity is insufficient to characterize CFS,” not that
similarity methods are universally invalid.

