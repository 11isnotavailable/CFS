# Historical speed-first early-stopping proposal (not executed)

**Superseded before production training by the user's common 120-epoch budget.
Use [CONTROLLED_IMAGENET_VIT_120EPOCH.md](CONTROLLED_IMAGENET_VIT_120EPOCH.md).
The early-stopping rules below are retained only as proposal history.**

The user selected speed-first early stopping before any production training:
50-epoch cosine cycle and minimum60 epochs. This supersedes the initial300-epoch
proposal and its extension rule. The old protocol lock and benchmarks are retained.

## Unchanged architecture and CFS definitions

| ID | Width | Depth | Heads | Head dim | Parameters |
|---|---:|---:|---:|---:|---:|
| V1 | 192 | 12 | 3 | 64 | 5,717,416 |
| V2 | 256 | 12 | 4 | 64 | 9,982,184 |
| V3 | 320 | 12 | 5 | 64 | 15,426,600 |
| V4 | 256 | 9 | 4 | 64 | 7,612,904 |
| V5 | 256 | 18 | 4 | 64 | 14,720,744 |
| V6 | 256 | 12 | 2 | 128 | 9,982,184 |
| V7 | 256 | 12 | 8 | 32 | 9,982,184 |

Patch16,224 inputs, MLP ratio4, learned absolute positions, one CLS, GELU,
LayerNorm epsilon1e-6, common biases, no register/distillation tokens. Fixed-
capacity models have identical initialized parameters under seed42, verified
tensor by tensor. V2 is trained once and reused in all axes. V3 exceedsV5 by
705,856 parameters: approximate, not exact, capacity matching.

Repository audit: historical GPT-2 scripts are in `scripts/controlled_gpt2/`;
the older ViT training implementation is `scripts/train_controlled_vit_scaling.py`
and used ImageNet-100. Original interventions and KL are reused from
`src/head_intervention.py`, `scripts/collect_predictor_dataset.py` and
`scripts/run_experiment0.py`. Original Matched-N/Cover/Rank functions are reused
through `scripts/analyze_matched_scaling.py`. Old code and results are unchanged.

## Uniform data preprocessing

The original full-resolution source is `benjamin-paine/imagenet-1k`, revision
`ac6a45918fb830d6d1555b01090d7b991e8f8a4a`. All310 shards use LFS SHA256 checks;
original bytes and image manifests remain available in indexed packs.

The source contains831 training rows byte-identical to validation images. All
seven models exclude those training paths by the same hash-only rule before
training. No labels, losses or predictions affect this exclusion. Preserve the
full original index and report final counts in DATA_READY. This tests exact-byte
overlap, not perceptual similarity. The original50,000 validation images and512
CFS probe images remain unchanged.

To reduce the measured CPU and ZFS IO bottlenecks, the user authorized:

- **Training:** convert to RGB, bicubic resize to short edge256 while preserving
  aspect ratio, cap exceptionally long edges at1024, encode once as JPEG95 with
  subsampling2. Record original/processed hashes and dimensions. This uniformly
  changes the training pipeline relative to full-resolution DeiT and must be
  disclosed; it is not claimed to be pixel-identical to that training recipe.
- **Validation:** cache original bicubic resize256/center crop224 as CHW uint8,
  without lossy reencoding. Normalize on GPU. Tests compare the cache against
  the original deterministic transform. CFS uses the original validation files
  and exactly this same evaluation transform.
- Indexed byte packs avoid per-image pathname lookup. Adjacent repeated samples
  reuse JPEG decoding, but receive independent image copies and fresh random
  augmentation. A test verifies that one view cannot mutate the next view.
- RandomResizedCrop, RandAugment, Mixup/CutMix and erasing are never cached.

Initial preparation re-fetched pinned training parquet for sequential packing
because reading1.28M staging files was much slower. Only this experiment's
redundant staging copies are removed after verified packs exist, with a20GiB
free-space guard. Historical datasets and original experiment outputs stay intact.

## Shared training and stopping rules

`src/controlled_vit_imagenet.py` and the seven JSON configurations are the
executable protocol. Training reuses [DeiT's recipe](https://github.com/facebookresearch/deit/blob/main/main.py)
and [timm1.0.28 components](https://github.com/huggingface/pytorch-image-models/blob/v1.0.28/train.py).

- Random initialization, seed42; no pretrained backbone, teacher or CFS loss.
- AdamW beta(.9,.999), epsilon1e-8, decay.05 with timm no-decay grouping;
  global batch1024, common microbatch1024, LR.001.
- Warmup5 epochs from1e-6; shared cosine cycle50 epochs to1e-5; keep the floor
  afterwards. The LR at a given epoch is the same across all seven models.
- **Minimum60 epochs, maximum300 as a safety cap.** Track the5-epoch moving
  mean of monitor CE. Relative improvement >0.2% resets patience; otherwise
  increment it. Stop after the minimum once patience reaches15 epochs.
- Monitor CE uses49,488 validation images, excluding the512 fixed CFS images.
  Select the online checkpoint with lowest unsmoothed monitor CE. Report both
  its selected epoch and the actual stop epoch. Full50,000-image CE/top1 are
  reported task metrics, not the stopping monitor.
- Checkpoints retain optimizer, scheduler, EMA, all per-rank RNG states, monitor
  state, best weights, history and sample seed. Resume replays an incomplete
  epoch from the last atomic epoch checkpoint; at most3 automatic attempts.
  Same world size/microbatch required; no cross-hardware bitwise guarantee.
- Save common-budget model checkpoints every10 epochs and75/125/175. After all
  seven stop, choose the **largest saved epoch present for all seven**, solely
  from availability, and repeat CFS there. This epoch is at least60. Do not use
  CFS values to choose checkpoints or change stopping rules.
- No automatic extension and no additional training seeds in this first run.
  Tiny is not assumed to converge first; stopping depends on measured CE.

Other common constants: RandAugment `rand-m9-mstd0.5-inc1`, Mixup.8/CutMix1,
probability1/switch.5, batch mode, smoothing.1, erasing.25 pixel mode, flip.5,
bicubic random crop scale(.08,1), ratio(.75,4/3), drop path.1, no other dropout,
no clipping (nonfinite gradients fail), BF16 autocast and SDPA, FP32 parameters,
EMA.99996 as secondary state, ImageNet mean/std, repeated augmentation3.
Sample visits are rounded to whole1024-image updates after the common exclusion.
DATA_READY/run.json record exact minimum/maximum/actual budgets.

The existing Pillow13 was incompatible with timm zero-bit posterization. An
isolated Pillow11.3 dependency preserves augmentation behavior without changing
other experiments' environments.

## Statistical analysis and interpretation

Evaluate the last block on the original512 images (`linspace(0,49999,512)` in
sorted validation order). Original FP16 intervention protocol: dense-to-variant
KL,31 alpha points[0,3], valid drop KL>1e-5, strict cover threshold CFS>.5.
Save complete drop/replacement/optimal-alpha matrices and alpha error cubes.
No additional Oracle clipping; retain the original rank affinity clipping.

- Width: Matched1/2, FixedK3. Depth: Matched1/2/3, fullH4 and FixedK3.
- Fixed capacity: Matched1, FixedK2; Matched2 only for supported models, neverH2.
  K2 is a limited sensitivity control.
- FixedK500 draws/input/layer, seed42; K=H once. Report Monte Carlo SE separately.
- Preserve valid-source-weighted Oracle ratios and original graph aggregation;
  denominators use allK nodes. Bootstrap whole inputs2000 times; paired model
  differences use identical image identities and resample indices.
- One training seed: input CIs are not architecture-training uncertainty.
  Do not infer significance from three-point monotonicity alone.

`analysis/` evaluates selected checkpoints with variable budgets;
`analysis_common/` evaluates equal epochs/sample visits. Only the latter supports
strict equal-training-budget comparisons. Report actual epochs, sample visits,
GPU-wall hours, curves and convergence limits. Analytic matmul FLOPs use MAC=2
and omit elementwise/norm/softmax/activation costs. Parameter efficiency, compute
efficiency and functional organization are separate; lower CFS is not itself a
causal explanation for higher accuracy.

## Running the experiment

Remote root `/data/cfs_controlled_vit_20261116`; both GPUs are authorized. Two
single-GPU queues run concurrently. The original raw-input300-epoch estimate
(~651 GPU-hours) is obsolete; new preprocessed benchmarks report60/90/120/300-
epoch scenarios rather than promise a stopping time. Update ETA from epoch logs.

```bash
cd /data/cfs_controlled_vit_20261116
export PYTHONPATH="$PWD/deps"
PY=/data/cfs_rebuttal_20261114/venv/bin/python
$PY scripts/prepare_controlled_imagenet.py --root "$PWD/data"
$PY scripts/pack_controlled_imagenet.py --root "$PWD/data"
$PY scripts/finalize_controlled_imagenet_data.py --root "$PWD/data"
$PY scripts/preprocess_controlled_imagenet.py --data-root "$PWD/data" --workers 8
$PY scripts/audit_controlled_vit_imagenet.py --output audit
$PY scripts/smoke_controlled_vit_imagenet.py
$PY scripts/benchmark_controlled_vit_imagenet.py --root "$PWD" \
  --val-root /data/cfs_rebuttal_20261114/data/imagenet1k_val --micro-batch 1024
$PY scripts/run_controlled_vit_imagenet.py --root "$PWD" --lock
$PY scripts/run_controlled_vit_imagenet.py --root "$PWD"
$PY scripts/status_controlled_vit_imagenet.py --root "$PWD"
```

Preparation can overlap. The initial strict download check stops on831 overlaps;
the registered exclusion script creates DATA_READY after verification. Never
fabricate readiness markers. Preserve the old protocol lock before creating the
amended lock. The queue verifies source hashes, readiness and numerical checks,
then runs all models, both CFS cohorts, bootstrap, tables, figures and reports.
Pending training and untrained smoke outputs are never labeled scientific results.
