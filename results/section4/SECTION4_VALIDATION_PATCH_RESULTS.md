# Section 4 Validation Patch Results

## 1. Scope

This document collects the requested validation results for:

1. parameter/contribution similarity versus CFS;
2. direct input-conditioned best-substitute statistics;
3. directionality;
4. the importance/calibration/representative decomposition;
5. an exact joint-subset oracle for simultaneous computation removal.

## 2. Similarity versus CFS

### Protocol

- Model: pretrained `deit_small_patch16_224`.
- Oracle data: Experiment 1.5.
- Evaluation split: the same 12 held-out images used by the Taylor and policy-decomposition experiments.
- Units: `image x layer x source head` over 12 layers and 6 heads.
- Candidate ranking: five candidate representatives for each fixed source.
- Correlation: Spearman correlation between each proxy and candidate CFS score.
- Undefined constant rows are omitted, matching the original Experiment 1.5 aggregation. This leaves 861 defined rows from 864 valid source units.
- Parameter vector for head `i`:
  `vec(W_Q^i, W_K^i, W_V^i, W_O^i)`; biases are not included.
- Contribution cosine uses the flattened post-`W_O` contribution `c_i(x)` already captured by Experiment 0/1.5.

### Results

| Proxy | Mean Spearman with CFS | Median |
|---|---:|---:|
| Taylor importance | 0.238634 | 0.300000 |
| Attention-map similarity | 0.017522 | 0.000000 |
| Contribution cosine | 0.008670 | 0.000000 |
| Contribution proximity (negative normalized L2) | 0.000270 | 0.000000 |
| Parameter cosine | -0.005015 | 0.000000 |
| Contribution norm | -0.012795 | 0.000000 |

Files:

- `similarity_vs_cfs_summary.csv`
- `similarity_vs_cfs_records.csv`

## 3. Direct input-conditioned best-substitute statistics

For every fixed `(layer, source)` pair, define its modal-best-substitute coverage as

```text
max_j # {x : argmax_k S_(i->k)(x) = j} / N.
```

The statistic uses all 32 frozen Experiment 1.5 images. There are 72 fixed source units (`12 layers x 6 sources`).

| Metric | Result |
|---|---:|
| Mean modal coverage | 34.07% |
| Median modal coverage | 31.25% |
| Modal-coverage Q1 / Q3 | 28.13% / 37.50% |
| Mean switch rate | 65.93% |
| Median switch rate | 68.75% |
| Mean number of distinct best substitutes | 4.97 / 5 |
| Median number of distinct best substitutes | 5 / 5 |

Files:

- `input_conditioning_summary.csv`
- `input_conditioning_by_layer.csv`
- `input_conditioning_by_source.csv`

The previously collected perturbation control remains:

| Comparison | Median best-representative agreement |
|---|---:|
| Mild color perturbation | 83.3% |
| Horizontal flip | 50.0% |
| Different images | 16.7% |

Original files:

- `../stability0_initial8/stability_summary.txt`
- `../stability0_initial8/stability_records.csv`

## 4. Directionality

The existing Experiment 1.5 directionality audit contains 5,760 unordered head pairs:

| Metric | Result |
|---|---:|
| Median absolute directional gap | 0.078518 |
| 75th percentile absolute directional gap | 0.176679 |
| One side `S > 0.10`, reverse `S <= 0` | 3.941% |
| One side `S > 0.25`, reverse `S <= 0` | 1.615% |

Original files:

- `../experiment15_analysis/analysis_metadata.json`
- `../experiment15_analysis/substitution_scores.npz`

## 5. Importance, representative and calibration decomposition

All rows use the same held-out protocol. Policies marked conditional use per-input oracle information.

| Policy | Mean recovery S | Median S | Mean KL |
|---|---:|---:|---:|
| Direct drop | 0.000000 | 0.000000 | 0.011511 |
| Fixed representative + fixed alpha | 0.001513 | 0.018470 | 0.010745 |
| Current-L2 representative + fixed alpha | 0.003501 | 0.001472 | 0.010910 |
| Conditional representative + fixed alpha | 0.121325 | 0.082153 | 0.009504 |
| Random representative + conditional oracle alpha | 0.198779 | 0.152762 | 0.007150 |
| Current-L2 representative + conditional oracle alpha | 0.202665 | 0.120413 | 0.007136 |
| Fixed representative + conditional oracle alpha | 0.218580 | 0.149702 | 0.006626 |
| Conditional representative + conditional oracle alpha | 0.370124 | 0.317427 | 0.003750 |

Taylor representative selection, with the same per-input oracle-alpha permission, obtains:

| Policy | Mean recovery S | Median S | Best-representative accuracy | Mean KL |
|---|---:|---:|---:|---:|
| Taylor representative + conditional oracle alpha | 0.259199 | 0.178321 | 33.56% | 0.005511 |

Original files:

- `../experiment15_analysis/policy_summary.csv`
- `../experiment15_analysis/policy_decomposition.csv`
- `../experiment15_analysis/strong_baseline_summary.csv`
- `../experiment15_analysis/strong_baseline_policies.csv`

## 6. Exact joint-subset oracle

### Protocol

- Model: `vit_base_patch16_224.augreg_in21k_ft_in1k`.
- Layer: final attention layer, layer 11.
- Heads: 12.
- Evaluation: the existing frozen router test split of 448 images.
- Intervention: simultaneous hard gating of all omitted heads.
- Objective: for every input and budget, exhaustively select the retained set with minimum KL to the dense output.
- Budgets:
  - keep 3: all `C(12,3) = 220` subsets;
  - keep 6: all `C(12,6) = 924` subsets;
  - keep 9: all `C(12,9) = 220` subsets.
- Total evaluated intervention variants: `448 x 1,364 = 611,072`.
- Taylor is evaluated with its actual simultaneous-gating output under the same head budget.
- Dense accuracy on the frozen split: 81.92%.

### Results

| Heads retained | Computation removed | Exact oracle KL | Exact oracle accuracy | Dense-prediction fidelity | Taylor KL | Taylor accuracy | Taylor fidelity | Oracle KL reduction vs Taylor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 / 12 | 75% | 0.081187 | 81.47% | 96.88% | 0.211562 | 80.58% | 93.75% | 61.63% |
| 6 / 12 | 50% | 0.012900 | 82.37% | 99.55% | 0.051255 | 81.47% | 95.76% | 74.83% |
| 9 / 12 | 25% | 0.002578 | 81.70% | 99.55% | 0.013751 | 81.03% | 98.44% | 81.25% |

### Paired bootstrap comparisons

Bootstrap resampling uses images as the unit with 10,000 replicates.

| Heads retained | Oracle - Taylor mean KL | 95% CI | Oracle - Taylor fidelity | 95% CI |
|---:|---:|---:|---:|---:|
| 3 / 12 | -0.130568 | [-0.154879, -0.108789] | +0.031167 | [+0.008929, +0.053571] |
| 6 / 12 | -0.038394 | [-0.047692, -0.030606] | +0.037982 | [+0.022321, +0.055804] |
| 9 / 12 | -0.011167 | [-0.015441, -0.007882] | +0.011105 | [0.000000, +0.022321] |

Files:

- `../section4_exact_joint_subset_oracle_test448/exact_joint_oracle_summary.csv`
- `../section4_exact_joint_subset_oracle_test448/exact_joint_oracle_samples.csv`
- `../section4_exact_joint_subset_oracle_test448/exact_joint_oracle_bootstrap.csv`
- `../section4_exact_joint_subset_oracle_test448/metadata.json`
- `../section4_exact_joint_subset_oracle_test448/run.log`

## 7. Scripts

- `scripts/analyze_section4_validation_patches.py`
- `scripts/collect_exact_joint_subset_oracle.py`
- `scripts/analyze_exact_joint_subset_oracle.py`

