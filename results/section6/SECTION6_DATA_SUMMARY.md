# Section 6 Data Summary

## Status

All experiments required for the current Section 6 story are already present
locally. No remote GPU needs to be started. The delivery package covers the
observability ladder, cross-model replication, final three-seed R2 routing,
paired bootstrap, calibration attribution controls, R1/R2 ablation, and online
mask consistency.

## 6.1 Observability

### Frozen DeiT-S ladder

| Information | Recovery | Seed std | Best-rep. acc. | Pair Spearman |
|---|---:|---:|---:|---:|
| Residual | 0.2204 | 0.0008 | 36.85% | 0.2553 |
| Q/K moments | 0.2251 | 0.0015 | 38.82% | 0.2718 |
| Attention logits | 0.2232 | 0.0034 | 38.24% | 0.2678 |
| V content | 0.2245 | 0.0012 | 38.65% | 0.2790 |
| (A V) | 0.2236 | 0.0025 | 38.53% | 0.2732 |
| Post-(W_O) | 0.2247 | 0.0004 | 39.03% | 0.2793 |
| Gradient sensitivity | **0.2643** | 0.0012 | **50.68%** | **0.4701** |

The corresponding Taylor recovery is 0.2318 and functional oracle is 0.3231.
The complete machine-readable table is `observability_ladder.csv`.

### Cross-model/task replication

| Model/task | Forward endpoint | Forward | Gradient | Gap |
|---|---|---:|---:|---:|
| DeiT-S / Imagenette | Q/K | 0.2251 | 0.2643 | +0.0391 |
| ViT-S AugReg / Imagenette | Q/K | 0.3037 | 0.3674 | +0.0637 |
| DINO ViT-S / linear probe | Q/K | 0.4844 | 0.5954 | +0.1110 |
| BERT-base / SST-2 | Post-(W_O) | 0.8833 | 0.9503 | +0.0670 |
| BERT-base / MNLI | Post-(W_O) | 0.6352 | 0.7548 | +0.1196 |

The safe claim is that downstream sensitivity improves observability over
forward probes across these settings. It does **not** always beat Taylor: DINO
is a counterexample, and the DeiT advantage over Taylor is depth-dependent.

## 6.2 Final ViT-B routing

The final primary method is R2 hard routing with joint-subset supervision. It
uses 4,096 train, 64 tune, 448 held-out test images and router seeds 17/23/41.
Dense accuracy is 81.92%.

| Keep | R2 accuracy | Taylor accuracy | R2 KL | Taylor KL | Relative KL reduction |
|---:|---:|---:|---:|---:|---:|
| 3/12 | 81.03% | 80.58% | 0.1190 | 0.2105 | 43.46% |
| 6/12 | 82.59% | 81.47% | 0.0281 | 0.0513 | 45.21% |
| 9/12 | 82.07% | 81.03% | 0.0070 | 0.0138 | 49.23% |

Paired-image bootstrap intervals for R2-minus-Taylor KL are:

- 3/12: `[-0.1137, -0.0717]`;
- 6/12: `[-0.0325, -0.0157]`;
- 9/12: `[-0.0110, -0.0035]`.

Thus the fidelity gain is resolved at every budget. Accuracy differences are
positive but their 95% intervals cross zero and should be described as
directional.

## Attribution controls

With the same learned calibration, CFS sets reduce KL relative to Taylor sets
by 0.0696, 0.0178, and 0.0068 at 3/12, 6/12, and 9/12. Even with oracle
pairwise alpha for both sides, the reductions remain 0.0604, 0.0154, and
0.0049. All six paired KL intervals exclude zero. The gain therefore cannot be
attributed only to alpha calibration.

## R1 versus R2

R2-minus-R1 KL changes are:

- 3/12: -0.0027, CI `[-0.0054, +0.0002]`;
- 6/12: -0.0034, CI `[-0.0050, -0.0017]`;
- 9/12: -0.0018, CI `[-0.0024, -0.0012]`.

Joint supervision gives a resolved fidelity improvement at 6/12 and 9/12 and
a borderline improvement at 3/12. Accuracy is statistically neutral.

One requested three-step ablation is not available: there is no final ViT-B
run that removes the structured subset loss while retaining only pairwise
regression/listwise supervision. Existing R1 already includes exact structured
subset supervision. The defensible current ablation is therefore R1 versus R2,
not a fabricated three-row sequence. If an isolated no-structured baseline is
later required, it can be trained and evaluated locally from frozen teachers;
no remote server is necessary.

## Online consistency and practical execution

At each budget, online and frozen offline masks match for 447/448 images
(99.7768%). The single mismatch is a numerical boundary case. The timing
columns must not be presented as sparse speedup because fused dense QKV remains
active.

## File index

- `observability_ladder.csv`: full seven-level DeiT-S ladder.
- `observability_cross_model.csv`: compact five-setting replication table.
- `ROUTER_METHOD_SUMMARY.md`: exact final architecture, losses, supervision,
  splits, seeds, and execution rules.
- `router_main_results.csv`: final R2 versus Taylor table with paired CIs.
- `router_bootstrap.csv`: complete R2 policy contrasts.
- `router_multiseed_summary.csv`: all final R2 execution policies.
- `router_selection_calibration_controls.csv`: learned- and oracle-alpha
  attribution controls.
- `router_ablation.csv`: R1/R2 means.
- `router_ablation_bootstrap.csv`: R2-minus-R1 paired intervals.
- `router_online_consistency.csv`: online correctness and mask agreement.

The package is reproducible with `scripts/build_section6_delivery.py`.

