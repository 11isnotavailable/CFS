# Historical initial proposal — superseded before formal training

**The user replaced mandatory300 epochs with a common 120-epoch initial budget and
shared offline preprocessing. The active protocol is
[CONTROLLED_IMAGENET_VIT_120EPOCH.md](CONTROLLED_IMAGENET_VIT_120EPOCH.md).
The initial proposal below is retained for audit only; its fixed-epoch and
extension instructions must not be used for the active experiment.**

This is a new from-scratch experiment, independent of historical pretrained
ViT and ImageNet-100 trajectories. No old experimental files are overwritten.
The executable source of configuration is `src/controlled_vit_imagenet.py`;
`configs/controlled_imagenet/{shared,V1,...,V7}.json` records its protocol hash.

## Repository audit

- Historical GPT-2 control scripts: `scripts/controlled_gpt2/`; original local
  `_remote_train.py` and `_run_controlled.sh` implement fixed-token training,
  resume, and fixed QKV width across head decompositions. Their text-domain
  dataset and optimization recipe are not silently reused for images.
- `docs/protocols/PROTOCOL_03_SCALING_AND_GENERALIZATION.md` contains an older
  ImageNet-100/75-epoch ViT protocol. This does not satisfy the new request.
- `scripts/train_controlled_vit_scaling.py` supplies historical DeiT-style
  augmentation and from-scratch construction, but also optional CFS training
  losses. The new experiment uses **no CFS regularization or pretrained weights**.
- `src/head_intervention.py`, `scripts/collect_predictor_dataset.py`, and
  `scripts/run_experiment0.py` define the reused intervention and KL.
- `scripts/analyze_matched_scaling.py` reuses exact combinatorial Matched-N,
  original exact cover and spectral rank, and input-cluster bootstrap.
- Original natural ViT probe is `linspace(0,49999,512)` in sorted ImageFolder
  order. It is recovered from the original metadata and byte hashes.

## Architectures and performance accounting

| ID | Width | Depth | Heads | Head dimension | Parameters |
|---|---:|---:|---:|---:|---:|
| V1 | 192 | 12 | 3 | 64 | 5,717,416 |
| V2 | 256 | 12 | 4 | 64 | 9,982,184 |
| V3 | 320 | 12 | 5 | 64 | 15,426,600 |
| V4 | 256 | 9 | 4 | 64 | 7,612,904 |
| V5 | 256 | 18 | 4 | 64 | 14,720,744 |
| V6 | 256 | 12 | 2 | 128 | 9,982,184 |
| V7 | 256 | 12 | 8 | 32 | 9,982,184 |

All use patch16, resolution224, MLP ratio4, learned absolute positions, one CLS,
no register/distillation tokens, GELU, LayerNorm epsilon1e-6, QKV/output/MLP biases.
Fixed-capacity models have identical parameter tensors at initialization under
seed42; only the attention reshape and head-dimension scaling differ.
V2 is trained once and its identical checkpoint reused across all three axes.
V3 has 705,856 more parameters than V5 (about4.8% relative to V5).
Analytic FLOPs count dense matrix products with MAC=2, separately attention,
MLP, patch embedding and classifier. Elementwise operations, normalization,
activation and softmax are excluded; these are not complete hardware FLOPs.
GPU-hours are separately measured. Equal epochs do not imply equal compute.

## Fixed training recipe

Reference implementations: [official DeiT](https://github.com/facebookresearch/deit/blob/main/main.py)
and [timm 1.0.28 training](https://github.com/huggingface/pytorch-image-models/blob/v1.0.28/train.py).
The small runner reuses timm's transforms, FastCollateMixup, GPU prefetch/erasing,
RepeatAugSampler, optimizer parameter grouping, CosineLRScheduler and ModelEmaV2.
It adds explicit configuration checks and atomic epoch-boundary state recovery.

- Seven models, 300 epochs, seed42, no pretrained backbone or teacher.
- AdamW, beta(.9,.999), epsilon1e-8, decay.05 with timm no-decay grouping;
  effective batch1024, LR.001 (DeiT base.0005 scaled by1024/512), warmup5 epochs
  from1e-6, cosine to1e-5; no gradient clipping (nonfinite gradients fail).
- RandAugment rand-m9-mstd0.5-inc1, Mixup.8/CutMix1, probability1/switch.5,
  batch mode, smoothing.1, erasing.25 pixel mode, flip.5, bicubic crop
  scale(.08,1), ratio(.75,4/3), drop path.1, no other dropout.
- Repeated augmentation3 using the official-style timm sampler. Epoch visits
  are rounded to full effective batches **after byte-overlap exclusions**.
  `DATA_READY.json` records exact counts shared by all seven models. Repeats
  are augmented views, not extra dataset.
- BF16 autocast and SDPA. FP32 parameters, EMA decay.99996 retained as secondary
  training state. **Primary checkpoint: final epoch300 online weights**, not the
  best validation epoch or whichever checkpoint has preferred CFS.
- Validation all50,000 at224, bicubic resize256/center crop224, ImageNet mean/std.
  CFS uses these same transforms. This differs from some historical AugReg
  pretrained checkpoint normalization, as appropriate for this new recipe.
- The existing environment's Pillow13 rejected timm's zero-bit posterization.
  An isolated Pillow11.3 dependency preserves the original augmentation behavior;
  no existing environment was changed. Failed pilot logs are retained.
- Epoch checkpoints save model, optimizer, scheduler, EMA, per-rank Python/NumPy/
  Torch/CUDA RNG, epoch/sample seed, history and GPU-hours. A partial failed epoch
  is replayed from the last completed epoch. Workers restart and reseed every
  epoch. Same world size and microbatch required for resume; no bitwise promise
  across CUDA hardware/kernel changes. Only bounded automatic retries are allowed.

Full-resolution ImageNet training parquet source is pinned to the same revision
as the historical validation set: `benjamin-paine/imagenet-1k` at
`ac6a45918fb830d6d1555b01090d7b991e8f8a4a`. Download each of310 shards, check LFS
SHA256, preserve image bytes, record per-image hashes, validate1,281,167 training
images and1000 classes, and audit byte overlap with validation before launch.
Only verified disposable training shard downloads are removed after extraction.

The downloaded source contains **831 training rows byte-identical to validation
images**. Before any formal training, a uniform data-integrity amendment excludes
every such training path, using SHA256 only (not predictions or losses). All
50,000 validation images and original512 CFS inputs remain unchanged. Original
training bytes/manifests remain archived in packs; `train_full_index.npy` preserves
the original index and `train_index.npy` selects the cleaned common subset.
This is an explicitly documented ImageNet-1K training-set hygiene adjustment,
not an exact reproduction of the unfiltered official DeiT dataset.

Host ZFS per-image metadata lookup was the initial IO bottleneck. Indexed byte
packs preserve JPEG data and sorted label/path ordering without transcoding.
Adjacent repeats reuse only JPEG decoding; each view receives its own image copy
and fresh timm random augmentation. A unit test checks that mutating one view
cannot change the next view. No augmentation draws or labels are cached.
Existing validation parquet is reused. Training parquet is re-fetched with the
same pinned checksums for sequential packing because reading1.28M newly created
individual files was much slower; this extra preparation network cost is recorded
separately from GPU training. Only this experiment's duplicate staging files are
removed after their byte-verified packs are complete.

The four prespecified V1/V2/V3/V5 pilots train fresh disposable weights for30
updates. Validation images are used **only as a decode/augmentation throughput
workload**; no pilot score or weight enters production. Discard5 warmup steps.
No recipe, model inclusion or stopping decision depends on pilot CFS.
All seven production runs start from new seed42 initialization after data readiness.

## CFS and statistical analysis

Primary layer is always the last block; no auxiliary depth layers in this first
run. Original source deletion/representative replacement, dense-to-variant KL,
31 alpha points[0,3], valid source deletion KL>1e-5, strict coverage CFS>.5.
FP16 original collector, full saved per-input drop/replacement/best-alpha arrays
and alpha error cubes. No additional clipping of Oracle scores; the original
rank routine's positive affinity clipping remains unchanged.

- Width: Matched1/2, FixedK3. Depth: Matched1/2/3, fullH4 and FixedK3.
- Fixed-capacity: Matched1, FixedK2; Matched2 only where H>=3, missing forH2.
- Additional supported N/K values may appear in per-model files but comparisons
  use only the shared support above. No fabricated H2 Matched2.
- Reuse500 subset draws/input/layer/K with seed42; K=H once, and report Monte
  Carlo SE separately. Coverage denominator is allK nodes, not valid sources.
- Oracle averages valid sources within a layer; bootstrap recomputes the ratio
  of total recovery sum to valid-source count. Graph metrics average inputs.
- 2000 independent-input bootstrap draws. Shared identities and identical
  bootstrap indices permit paired differences to adjacent models andV2.
- Input bootstrap is conditional on a single trained checkpoint, not a CI for
  architecture training randomness. No significance claim from three-point
  monotonicity or correlation alone; no multiplicity correction is implied.

Report all training curves and last20-epoch trends. The primary study remains300
epochs. Before formal training, fix this convergence diagnostic: if **any** model
improves validation CE by >1% between mean epochs281–285 and296–300, extend **all
seven** models by60 epochs at the scheduler floor LR1e-5. Epoch300 checkpoints,
performance and CFS remain primary; epoch360 task metrics are supplementary and
do not silently replace primary CFS. The trigger is independent of CFS and is a
diagnostic, not proof of convergence when false. Save EXTENSION_DECISION.json.
This adds at most20% training; no additional training seeds in the first run.

## Deliverables and execution

Remote root `/data/cfs_controlled_vit_20261116`, existing interpreter
`/data/cfs_rebuttal_20261114/venv/bin/python`, isolated `deps` on PYTHONPATH.
Two single-GPU queues keep global batch1024; a common fitting microbatch is
chosen solely from throughput/memory benchmarks (256 vs1024, no learning-score
selection), with accumulation if smaller than1024. torchrun DDP is also
supported. Production starts only after benchmark/data/protocol/numerical checks.
Final dual-GPU pilots selected a common microbatch1024, safely below96GB/card.
The initial total estimate is about651 GPU-hours /15.3 queue days before full
validation/checkpoint overhead, with a possible20% convergence extension. These
are throughput projections, not completed training hours; update from epoch logs.
Training is multi-day; status must distinguish pending/running/completed results.
Final measurement, offline controls and report run automatically after training.
No interim validation or untrained CFS is represented as a completed experiment.

```bash
cd /data/cfs_controlled_vit_20261116
export PYTHONPATH="$PWD/deps"
PY=/data/cfs_rebuttal_20261114/venv/bin/python
# Preserve all commands' logs; preparation stages may run concurrently.
$PY scripts/prepare_controlled_imagenet.py --root "$PWD/data"
$PY scripts/pack_controlled_imagenet.py --root "$PWD/data"
$PY scripts/finalize_controlled_imagenet_data.py --root "$PWD/data"
$PY scripts/audit_controlled_vit_imagenet.py --output audit
$PY scripts/smoke_controlled_vit_imagenet.py
$PY scripts/benchmark_controlled_vit_imagenet.py --root "$PWD" \
  --val-root /data/cfs_rebuttal_20261114/data/imagenet1k_val --micro-batch 1024
$PY scripts/run_controlled_vit_imagenet.py --root "$PWD" --lock  # once only
$PY scripts/run_controlled_vit_imagenet.py --root "$PWD"
# Read-only progress, safe while running:
$PY scripts/status_controlled_vit_imagenet.py --root "$PWD"
```

The initial downloader intentionally fails its strict zero-overlap gate after
retaining the full download audit. The pretraining hygiene amendment is then
applied by `finalize_controlled_imagenet_data.py`, which creates DATA_READY only
after verified packs, the complete exclusion index and unique probe check.
Do not bypass that marker manually. The queue has an exclusive lock and verifies
source hashes; do not edit its source after PROTOCOL_LOCK. All production attempts
including replayed epochs are logged for reserved GPU-wall hours; the epoch-level
train/validation timer is also reported separately. Checkpoints remain on the
server with resume state and model-only epoch50/100/150/200/250/300 exports.
