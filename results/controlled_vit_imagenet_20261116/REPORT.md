# Controlled ImageNet-1K ViT run: protocol and launch record

Status recorded on 2026-11-16: **preprocessing and the persistent queue are running;
production training has not started yet. There are no trained CFS results to report.**

The active [120-epoch protocol](../../docs/CONTROLLED_IMAGENET_VIT_120EPOCH.md)
replaces the unexecuted 300-epoch and individual early-stopping proposals.

## Registered comparison

- Seven from-scratch ViTs, identical 120-epoch initial budget and 5-epoch warmup /
  cosine schedule. Same effective batch 1024, seed 42 and augmentation recipe.
- Online checkpoints at epochs 40, 60, 80, 100, 120. Primary comparison: epoch
  120 for every model. No per-model stopping or favorable CFS checkpoint selection.
- Identical 64-image probe at epochs 60, 80, 100, 120, original CFS definition and
  31-point alpha grid. Full 512-image analysis at epoch 120.
- Joint CE / CFS trajectory review after all seven reach 120. No automatic
  extension; any subsequent budget must be decided uniformly for all seven.

## Verified preparation

All 310 training shards have been downloaded, hash checked and packed. The audit
identified 831 training paths byte-identical to validation images. Uniform
pre-training exclusion leaves 1,280,336 training images, while all 50,000
validation images and the historical 512-image CFS cohort remain unchanged.

Indexed packs remove per-image filesystem lookup. Offline training resize and
JPEG encoding are shared by all models; original bytes and hashes are retained.
Validation caches preserve the original deterministic crop pixels. Online random
augmentation remains active. At the launch check, **173/310 training shards** had
completed offline processing. This is a dated snapshot, not a live progress counter.

Four disposable preprocessed pilots (V1/V2/V3/V5) successfully exercised training
on the two GPUs independently. Their first timings overlapped CPU preprocessing
and are unsuitable for a dependable completion estimate. A registered 100-update
pilot on the complete cleaned training index waits for preprocessing to finish.
No pilot weights are reused. The production queue waits for this pilot and full
data readiness, then starts V1 and V2 concurrently.

Cross-GPU NCCL tests failed on this server, including with P2P disabled. Production
uses independent GPU queues with the same effective batch and sample budgets.
The successful independent-card pilots validate this execution path; unrelated
server experiments and environments were not changed.

## Correctness evidence

`architectures.csv` records actual instantiated parameters, gate equivalence and
analytic matmul FLOPs (MAC=2; excluding elementwise/norm/softmax/activation costs).
`SMOKE_CHECKS.json` records untrained H=2 and H=8 integration checks, not results.
`PROTOCOL_LOCK.json` records the exact remote code/configuration hashes.

Final local check: **14 tests passed**, covering identical-capacity initialization,
supported N/K boundaries, exact Matched-N enumeration, checkpoint optimizer /
scheduler / RNG continuation, pixel cache equivalence, fixed 64-image selection,
and preservation of original unclipped CFS values / input order in stability
analysis. Source compilation also passed.

## Execution and pending deliverables

Remote root: `/data/cfs_controlled_vit_20261116`.

```bash
cd /data/cfs_controlled_vit_20261116
export PYTHONPATH="$PWD/deps" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
PY=/data/cfs_rebuttal_20261114/venv/bin/python
$PY scripts/status_controlled_vit_imagenet.py --root "$PWD"
# The queue is already running. Use this only to resume after it exits:
$PY -u scripts/run_controlled_vit_imagenet.py --root "$PWD"
```

Queues: GPU0 V1/V3/V4/V6; GPU1 V2/V5/V7. The queue owns a filesystem lock,
verifies the frozen source hashes and performs bounded epoch-resume retries.
Each model's log/history and five milestone checkpoints remain under `runs/`.
Per-epoch CFS matrices and statistics go to `runs/<model>/light_cfs/`.
Final raw CFS goes to `cfs/`; main figures/tables/report go to `analysis/`;
the convergence and functional stability review goes to `stability/`.

Training, final CFS comparisons, input-bootstrap results and the convergence
decision are **pending**. A common 120-epoch budget establishes comparability;
it does not by itself establish adequate convergence or functional stability.
