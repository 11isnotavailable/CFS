# Seven controlled ImageNet-1K ViTs: common 120-epoch initial budget

**Run stopped by the user on 2026-11-16. This frozen recipe is retained for audit;
do not resume it automatically. The current task is the finite
[BERT throughput benchmark](../results/bert_throughput_20261116/REPORT.md), with no
formal BERT training authorized.**

This is the active protocol, fixed before production training on 2026-11-16.
It supersedes both the initial 300-epoch proposal and the unexecuted individual
early-stopping proposal. The objective is adequately trained, functionally stable,
comparable models, not state-of-the-art ImageNet classification accuracy.

## Common training budget and checkpoints

All seven models train from scratch for **120 epochs** under one schedule:
5 epochs of linear warmup (learning rate 1e-6 to 1e-3), followed by the shared
timm cosine schedule reaching 1e-5 at epoch 120. There is no individual early
stopping, no CFS-based checkpoint selection and no automatic extension.

Save online-model checkpoints at epochs **40, 60, 80, 100 and 120**. `last.pt`
also stores optimizer, scheduler, EMA, RNG and history for epoch-boundary resume.
An interrupted partial epoch is replayed; its wall time remains in attempt logs.
The primary controlled comparison uses **epoch 120 for all seven models**.
Earlier comparisons, if reported, must use a common epoch across all seven.
EMA is maintained for reproducibility but does not replace the primary weights.

| ID | Width | Depth | Heads | Head dimension | Parameters |
|---|---:|---:|---:|---:|---:|
| V1 | 192 | 12 | 3 | 64 | 5,717,416 |
| V2 | 256 | 12 | 4 | 64 | 9,982,184 |
| V3 | 320 | 12 | 5 | 64 | 15,426,600 |
| V4 | 256 | 9 | 4 | 64 | 7,612,904 |
| V5 | 256 | 18 | 4 | 64 | 14,720,744 |
| V6 | 256 | 12 | 2 | 128 | 9,982,184 |
| V7 | 256 | 12 | 8 | 32 | 9,982,184 |

All use patch 16, input 224, MLP ratio 4, GELU, LayerNorm epsilon 1e-6,
one CLS token and learned absolute positional embeddings. V2/V6/V7 have
identical initial parameter tensors with seed 42; only head partitioning differs.
Width axis: V1/V2/V3. Depth axis: V4/V2/V5. Head-count axis: V6/V2/V7.
V3 and V5 are approximately, not exactly, parameter matched.

The executable recipe is `src/controlled_vit_imagenet.py`, with hashed JSON in
`configs/controlled_imagenet/`. Common settings: AdamW, effective batch 1024,
betas 0.9/0.999, epsilon 1e-8, decay 0.05 excluding biases/norm, stochastic depth
0.1, smoothing 0.1, Mixup 0.8, CutMix 1.0, RandAugment rand-m9-mstd0.5-inc1,
random erasing 0.25, repeated augmentation 3, BF16, seed 42. The two GPUs run
independent model queues; no production run depends on cross-GPU NCCL.

## Shared data preparation

Source: `benjamin-paine/imagenet-1k`, frozen revision
`ac6a45918fb830d6d1555b01090d7b991e8f8a4a`. All 310 source shards are SHA256
verified. Original image bytes, manifests and indices are retained in indexed
packs. The data audit found 831 training paths byte-identical to validation
images; these are excluded uniformly before training. This is exact-byte overlap
removal, not a claim of perceptual deduplication. The cleaned training set has
**1,280,336 images** and 1,000 classes; the full 50,000-image validation set and
historical 512-image CFS cohort are unchanged. RepeatAugSampler and drop-last
produce 1,250 updates / 1,280,000 sample visits per epoch for every model.

One-time deterministic processing reduces filesystem and decode overhead:

- Training: RGB, bicubic short-edge resize to 256 with aspect ratio preserved,
  cap exceptional long edges at 1024, JPEG quality 95 with subsampling 2.
  Record original and processed hashes and dimensions for every image. This
  uniformly changes the training image pipeline relative to full-resolution
  inputs and must be disclosed; it is not lossless preprocessing.
- Validation: cache the original bicubic resize / center crop as raw CHW uint8.
  A numerical test verifies normalization reproduces the original transform.
- Random resized crop, flip, RandAugment, Mixup/CutMix and erasing stay online.
  Decode reuse for repeated indices returns independent image copies; random
  augmentation is never cached.
- CFS always uses original validation files and the historical preprocessing.
  Source files and historical experiments are not modified.

Production waits for complete data verification, preprocessing, throughput pilots
and CFS integration checks. Pilot weights are discarded. The task-local Pillow
11.3 dependency resolves the observed Pillow 13 / timm posterize incompatibility
without modifying the environment used by unrelated experiments.

## CE and functional stability monitoring

Evaluate full validation CE/top-1/top-5 every epoch. At epochs **60, 80, 100, 120**,
measure CFS on the **same 64 images**: evenly spaced offsets within the original
fixed 512-image probe, identical across all models and epochs. Use each model's
last attention block, the original 31-point alpha grid on [0,3], original output
KL and valid-source threshold 1e-5. Sampling fewer inputs makes monitoring light;
the intervention or optimization precision is not reduced.

Retain complete directed matrices, drop losses, alpha error cubes and identities.
Monitoring reports Raw/Matched Oracle, Cover/H, Rank/H, paired input bootstrap
intervals, matrix MAE and rank correlation between successive checkpoints, and
the proportion of sources valid at both checkpoints. No head or pair is treated
as an independent bootstrap sample. Undefined correlations (especially H=2)
are not turned into evidence of stability. A zero-crossing change interval is
not an equivalence test, and stable aggregate means do not prove stable matrices.

After all models reach 120, `analyze_controlled_vit_stability.py` combines:

- CE trajectories, last-20-epoch slope, and mean CE at 91-100 versus 111-120;
- CFS changes at 60-80, 80-100 and 100-120 with paired 95% intervals;
- matrix changes, valid-source overlap and sampling uncertainty.

It writes a joint review, not an automatic stopping/extension verdict. If any
model still shows material CE improvement or functional drift, decide a **common
extension for all seven** after inspecting the full evidence. If CE worsens,
inspect overfitting rather than assuming that more epochs will help. No model
receives a bespoke endpoint chosen to improve a scaling trend. One training seed
and a 64-image monitoring cohort limit what stability can establish.

## Primary CFS analysis and implementation reuse

After each model completes 120 epochs, collect the original 512-image CFS cohort
with the same intervention definition and final block. Reuse
`collect_predictor_dataset.py`, `src/head_intervention.py`, `run_experiment0.py`
and `analyze_matched_scaling.py`; no historical code or results are overwritten.
The new checkpoint wrapper only constructs the controlled architecture and sets
its preprocessing metadata.

Matched-N uses the original exact combination expectation for available N=1/2/3.
Fixed-K uses K=2/3 where supported, 500 draws, seed 42, and K=H once. Bootstrap
uses 2,000 whole-input resamples. Comparisons only use jointly supported settings:
H=2 does not have Matched-2. Primary comparisons, input identities, raw/full and
controlled metrics, task performance, training costs and full learning curves
are saved as CSV/JSON, LaTeX, PDF/PNG and Markdown. No monotonic trend is assumed.

## Remote reproduction

Root: `/data/cfs_controlled_vit_20261116`. Python:
`/data/cfs_rebuttal_20261114/venv/bin/python`. In the task root:

```bash
export PYTHONPATH=/data/cfs_controlled_vit_20261116/deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
PY=/data/cfs_rebuttal_20261114/venv/bin/python
$PY scripts/preprocess_controlled_imagenet.py --data-root "$PWD/data" --workers 8
$PY scripts/benchmark_controlled_vit_imagenet.py --root "$PWD" --val-root /data/cfs_rebuttal_20261114/data/imagenet1k_val
# Lock once, after tests and configuration freeze. Never overwrite an active lock.
$PY scripts/run_controlled_vit_imagenet.py --root "$PWD" --lock
$PY -u scripts/run_controlled_vit_imagenet.py --root "$PWD"
$PY scripts/status_controlled_vit_imagenet.py --root "$PWD"
```

The queue handles per-model resume, fixed-time CFS monitoring, final 512-image
collection and both analysis reports. An interrupted queue may be relaunched
with the same command; hashes prevent silently changing its active recipe.
Checkpoints, datasets and raw matrices stay on the experiment server, not in Git.
