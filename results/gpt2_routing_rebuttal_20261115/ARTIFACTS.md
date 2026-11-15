# Artifact inventory and reproduction

Both layers finished. All 18 main models (R2/Gating/A0 × seeds17/23/41 ×
layers11/8) were selected on validation data before their test evaluation.
Backbone training: none. Raw evaluation: 18,432 method/seed/budget rows over
the same 256 independent documents; repeated rows are not independent samples.

## Local and remote backups

- Full experiment archive:
  `D:/tries2/outputs/cfs_gpt2_routing_artifacts_20261115.tar.gz`
  (775,638,996 bytes).
- SHA256:
  `b927606c4e4b5c93dc77e00714daec88deb817a2882e510dff700897b791fc37`.
- Frozen public pretrained backbone:
  `D:/tries2/outputs/gpt2_pretrained_607a30d_model.safetensors`
  (548,105,171 bytes), SHA256
  `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
- Remote run root: `/data/cfs_gpt2_routing_20261115`.
- Remote full archive: `/data/cfs_gpt2_routing_artifacts_20261115.tar.gz`.
- Exact public checkpoint revision:
  `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`.

The archive includes every saved epoch and selected checkpoint for all main,
pilot and bootstrap routers, full logs/configurations, frozen input documents
and tokens, all pairwise drops/substitution/alpha/features/baselines, all joint
teacher masks/KLs, full validation suffix caches, raw test predictions, and
analysis/source files. Only train-cache residual/heads/reference tensors are
omitted from this local compact archive; their complete original NPZ files
remain remote, with full-file hashes in `full_cache_manifest.json`. No teacher
target is omitted. The public backbone is backed up separately as above.

Do not treat compact training caches as complete suffix caches when collecting
new joint labels. Use the preserved full remote cache or recapture prefixes;
existing joint teacher labels require no recomputation.

The repository contains summaries, complete per-input/per-seed metrics,
paired differences, bootstrap draws, figures, LaTeX tables, and audit metadata.
Raw text and large model weights are intentionally in the backup, not Git.

## Entry points

- [REPORT.md](REPORT.md): methods, all metrics and interpretation.
- [REBUTTAL_WCQn.md](REBUTTAL_WCQn.md): English response draft.
- [summary.csv](summary.csv): KL/NLL/PPL, input SD, seed SD and 95% intervals.
- [paired.csv](paired.csv), [paired_per_input.csv](paired_per_input.csv):
  document-paired R2 contrasts, without treating heads/seeds as extra samples.
- [per_seed.csv](per_seed.csv), [per_input.csv](per_input.csv): complete results.
- [main_kl_ci_table.tex](main_kl_ci_table.tex): paper-ready mean/CI table.
- [figure_routing_kl.pdf](figure_routing_kl.pdf): vector figure.
- [runtime.csv](runtime.csv): actual full-prefix forward latency.
- [verification.json](verification.json): identity, disjointness, selection,
  teacher, numerical and weight audits; all passed.
- [environment.json](environment.json): exact production package versions.
- [source_manifest.json](source_manifest.json): executed source hashes.
- [protocol](../../docs/GPT2_ROUTING_PROTOCOL.md): frozen study design.

The seven reused core source files were also hash-compared against this Git
checkout with no differences. There were 68 passing local tests and 5 passing
GPT-2 tests in the production environment.

## Reproduction

Use the frozen archive inputs or the exact public parquet SHA in
`provenance.json`; its historical dataset revision is unknown and was not
invented. The model revision is pinned by the preparation script. Run commands
are in REPORT.md; both layers use separate teachers and independent training.
The `RUN` directory must contain copies of `scripts/`, `src/`, `tests/` and
`docs/`. Never reuse a nonempty training output for a fresh run.

The online API is `GPT2RuntimeRouter.next_logits(ids)` in
`src/gpt2_routing.py`. Load the selected `model.pt` payload, use an eval-mode
frozen GPT-2 backbone, and call under `torch.inference_mode()`. It accepts a
complete unpadded prefix and returns only the next-token logits; there is no
KV-cache or sparse-kernel acceleration claim. Close the router to remove hooks.

R2-versus-A0 compares the full CFS-related auxiliary supervision bundle
(pairwise, deletion/importance, alpha and structured supervision). It does not
identify the isolated causal contribution of pairwise CFS labels alone.
