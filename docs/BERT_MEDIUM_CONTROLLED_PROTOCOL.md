# Frozen BERT-Medium controlled scaling protocol (2026-11-16)

The user authorized this new seven-model2B experiment after the small-family
throughput benchmark. Neither the stopped ViT run nor the old small BERT model
definitions are reused as scientific results. No pretrained weights are loaded.

## Audit and architecture

Original GPT-2 records: FineWeb sample-10BT, revision
`9bb295ddab0e05d785b879661af7260fed5140fc`, original15-shard manifest and seeded
order, BLAKE2b-64(seed42 + NUL + raw UTF-8 text), little endian, modulo1000:
bucket0 validation and all others training. There is no original test split.
See `results/bert_throughput_20261116/gpt2_audit/` for actual archived evidence.

| ID | Width | Layers | Heads | Head dimension | Actual parameters |
|---|---:|---:|---:|---:|---:|
| B1 | 384 | 8 | 6 | 64 | 26,293,434 |
| B2 | 512 | 8 | 8 | 64 | 41,404,730 |
| B3 | 640 | 8 | 10 | 64 | 59,694,522 |
| B4 | 512 | 6 | 8 | 64 | 35,099,962 |
| B5 | 512 | 12 | 8 | 64 | 54,014,266 |
| B6 | 512 | 8 | 4 | 128 | 41,404,730 |
| B7 | 512 | 8 | 16 | 32 | 41,404,730 |

B2/B6/B7 have identical initialized parameter tensors and hashes under seed42;
only attention head partitioning differs. B2 is trained once. All models use
HF BERT post-LayerNorm, GELU, FFN4x, biases, learned512-position embeddings,
two token-type embeddings (single segment), tied MLM decoder, vocabulary30,522,
dropout0.1 and no NSP. Actual input length is128. FLOPs are analytic forward
matmuls with MAC=2 and expected15% of126 MLM positions; they exclude activation,
normalization, softmax and backward. They are not measured end-to-end FLOPs.

## Data and token accounting

WordPiece tokenizer-only files reuse the frozen bert-base-uncased revision
`86b5e0934494bd15c9632b12f734a8a67f723594`; no model weights are loaded.
Download only as many original shards as needed, with full source SHA256 checks.
Preserve original seeded shard order and row order. Each training block contains
126 WordPiece tokens from one training document plus CLS/SEP, no padding;
document tails shorter than126 tokens are discarded. There is no cross-document
packing or reuse of GPT-2 BPE IDs. This differs from GPT-2's causal1024-token
packing and is disclosed, while remaining identical within the new family.

Freeze exactly15,625,000 blocks / **2,000,000,000 nonpadding input positions**
including special tokens (1,968,750,000 content tokens). Store per-block document
IDs and per-document provenance. Seven models consume this same stream once.
Log input tokens, content tokens, actual selected MLM targets and document visits.

Use2,048 distinct validation document hashes and one prefix block per document.
Their training-style15% masks and80/10/10 replacements are fixed once. Their
documents cannot enter training under the original hash rule. Freeze256 of these
documents for primary CFS, and64 evenly spaced indices within those256 for the
1B/2B stability comparison. The absence of an independent original test split
is retained, not disguised by calling these validation texts a test set.

## Training

Shared seed42, effective batch256 sequences, BF16 autocast / FP32 parameters and
optimizer, fused AdamW (beta0.9/0.999, epsilon1e-8, weight decay0.01 excluding
bias/norm), gradient norm clip1.0. Peak LR3e-4, linear token-based warmup for2%
(40M input positions), cosine to3e-5 at2B. No compilation or gradient scaling.
This is a common BERT recipe chosen before training, not a tuned performance claim.

Reuse checked sparse MLM projection; only supervised positions enter the
pointwise MLM transform, vocabulary projection and CE. The encoder sees all
128 positions with bidirectional attention. Training masks use a separate CUDA
generator seeded4242, isolating them from architecture-dependent dropout RNG.
Mask probability15%,80%MASK/10%random/10%unchanged, excluding special tokens.

Save0.5B/1B/1.5B/2B model checkpoints. A shared202-sequence tail batch at each
milestone makes the budget exact;61,036 total optimizer updates. The2B schedule
is fixed from update1. Atomic resume checkpoints every1,000 updates include
optimizer, model, CUDA/CPU RNG, independent mask RNG, stream cursor and history.
Interrupted partial work is replayed; attempts are separately costed. Validation
is run every1,000 updates and at each milestone; per-document checkpoint loss
sums, correct counts and target counts are retained for paired statistics.

No per-model early stopping or special learning rates.2B is the primary endpoint.
No automatic3B training. Any future1B extension must be a newly documented common
stage for all seven, after joint CE and CFS review.

## CFS definitions and numerical reuse

Reuse `collect_bert_mlm_scaling.selected_mlm_logits`, `variant_logits` and
`outcome_kl`: FP16 autocast, dense-to-intervened output KL averaged across exactly
8 fixed MASK positions,16 alpha values on[0,3], drop KL>1e-5, cover threshold>0.5.
The original parameterization sets source scale0 and candidate scalealpha;
alpha1 equals deletion-only, alpha0 deletes both. The shared output bias remains.

The compatibility adapter preserves pre-WO head gating across HF versions that
return different tuple lengths. Cache the prefix before the final block and
stream output KL per chunk to avoid retaining gigabytes of vocabulary logits.
Four-input untrained checks compare against the original full-backbone Oracle;
each trained checkpoint additionally checks its first input against an uncached
streamed Oracle. Full directed matrices, alpha choices and all alpha errors are
retained. Caching is a numerical implementation optimization, not a CFS redefinition.

Report input-mean Oracle aggregation as in original BERT MLM analysis. The final
paper's unified rank replaces the obsolete BERT-only no-self variant: use
`spectral_effective_rank` from `analyze_functional_capacity`, with clipped positive
affinity, unit self-affinity and invalid rows zeroed, as explicitly done in
`build_section5_delivery.bert_table`. Preserve raw CFS without Oracle clipping.

Matched-N uses the existing exact combination expectation. Width/depth compare
N2/4/5; head decomposition compares N1/2/3. Unsupported N is absent, never capped
silently. FixedK4 uses existing induced-subgraph functions,500 draws per input,
seed42, and a single full graph for H4. Save draws and100/250/500 convergence
checks. Bootstrap2,000 whole input documents, paired across models and checkpoints;
subgraph Monte Carlo error is separate. One training seed is a limitation.

## Execution and outputs

New remote root `/data/cfs_bert_medium_20261116`, Python
`/data/cfs_rebuttal_20261114/venv/bin/python`, two independent PRO6000 GPU queues.
Preflight: actual parameter/hash audit, new B1/B2/B3/B5 throughput, sparse MLM
loss/gradient equivalence, exact budget tests and cached/streamed intervention tests.
Original benchmark cache is used only for disposable throughput pilots.

```bash
cd /data/cfs_bert_medium_20261116
PY=/data/cfs_rebuttal_20261114/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
$PY scripts/audit_bert_medium.py --output audit
$PY scripts/smoke_bert_medium_cfs.py --root "$PWD"
# Prepare once; retains per-shard resume markers and source hashes.
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=12 $PY scripts/prepare_bert_medium_data.py \
  --root "$PWD" --metadata /data/cfs_bert_benchmark_20261116/gpt2_data_metadata.json \
  --tokenizer /data/cfs_bert_benchmark_20261116/tokenizer
# Freeze once after checks. Do not overwrite an existing lock.
$PY scripts/run_bert_medium.py --root "$PWD" --lock
$PY -u scripts/run_bert_medium.py --root "$PWD"
$PY scripts/status_bert_medium.py --root "$PWD"
```

`runs/` contains all logs/weights and validation records; `cfs/` contains1B64-input
and2B256-input raw matrices.2B stability reuses the common64 rows from full256
results. `analysis/` contains per-input metrics, bootstrap draws, paired contrasts,
curves, raw CSV/JSON, LaTeX, Markdown and an evidence-based English rebuttal draft.
Checkpoint comparisons are scheduled in advance and cannot select a preferred
scaling direction. Missing or failed tasks are reported as incomplete.
