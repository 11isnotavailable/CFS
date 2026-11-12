# Reproducibility Rules and Release Plan

## Release statement

This repository releases the author-owned CFS research implementation,
including:

- head-intervention implementations for vision, encoder language models, and
  decoder-only/GQA language models;
- exact CFS, Matched-\(k\), cover, rank, and graph-analysis code;
- frozen split/sample/position indices and protocol manifests;
- router teacher-generation, training, and evaluation code;
- controlled-scaling training configurations;
- training-dynamics snapshot and plotting scripts;
- compact summary tables where redistribution is permitted;
- environment specifications, tests, and commands used for reported tables.

Third-party model weights and datasets will not be relicensed. The release will
instead provide acquisition/preparation instructions and immutable identifiers
where their licenses require this.

## Shared experimental invariants

1. Verify all-one gates reproduce the dense network.
2. Use post-output-projection query-head contributions as branches.
3. Use reference-to-intervention KL consistently.
4. Compute CFS per input/source before dataset aggregation.
5. Preserve directionality and exclude self-candidates.
6. Mark deletion KL <= `1e-5` as invalid for normalized recovery.
7. Freeze alpha grid, layer indices, examples, token positions, and thresholds
   within each comparison.
8. Report Raw Oracle with Matched-\(k\) whenever candidate counts differ.
9. Report normalized recovery with absolute deletion/replacement errors.
10. Separate train, tune, and final-test identities; augmentation variants do
    not constitute independent identities.
11. Use at least three router/probe seeds for learned comparisons.
12. Record model revision, tokenizer/preprocessing revision, configuration,
    sample IDs, and software environment.

## Cross-family comparability

CFS has a common normalized meaning across model families, but the underlying
output space differs. Classification KL, masked-token KL, and next-token KL
must not be compared as identical absolute scales. Natural public model-family
trajectories also change several architectural variables simultaneously; only
explicit controlled experiments support single-axis causal interpretations.

## Minimum table fields

Every main CFS table should include model/task, parameter count, layer/head
metadata, sample count, alpha grid, valid-source rate, task performance, Raw
Oracle, Matched-2, Cover/Head, Rank/Head, and an absolute-effect statistic.

## Artifact integrity

Released processed artifacts will include checksums and a machine-readable
manifest. Large raw activations are not required for reproduction when compact
per-input CFS matrices, deletion errors, replacement errors, alpha values, and
frozen sample identifiers are provided.

## License boundary

No software license is declared in the initial release. Third-party assets are
not redistributed and remain governed by their original licenses.

