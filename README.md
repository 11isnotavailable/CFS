# Conditional Functional Substitutability (CFS)

Official research code and compact result artifacts for:

> **Effective Does Not Mean Useful: Conditional Functional Substitutability for Redundancy and Scaling in Transformers**<br>
> Jiaheng Chen, Jiaxing Li, Yucheng Xiao, Xinyong Cai, Juncheng Bu, Lan Yu, and Tinghe Zhang<br>
> [arXiv:2609.39259](https://arxiv.org/abs/2609.39259)

CFS asks whether the downstream effect of removing a Transformer component can
be recovered by another component for a particular input. Unlike importance or
representation similarity, CFS is functional, directional, and
input-conditioned.

## Definition

For source component `i`, substitute `j`, and input `x`,

```text
S(i -> j | x) = 1 - D*(i -> j | x) / D_drop(i | x),
```

where `D_drop` is the output discrepancy after removing `i`, and `D*` is the
smallest discrepancy obtained by removing `i` and recalibrating `j`. Normalized
scores are reported only when `D_drop > 1e-5` in the paper protocols.

## Repository map

- `src/`: head interventions for ViT, encoder models, GPT-style/GQA models,
  sparse attention, and the learned runtime router.
- `scripts/`: data collection, CFS analysis, controlled scaling, exact
  joint-subset oracles, router training/evaluation, and figure/table builders.
- `scripts/controlled_gpt2/`: controlled GPT-2 CFS collection and analysis.
- `tests/`: synthetic invariants for gating, graph metrics, GQA indexing,
  controlled-head analysis, and router behavior.
- `docs/protocols/`: paper-aligned experimental protocols and pseudocode.
- `results/`: compact CSV summaries underlying Sections 4--6. Large features,
  checkpoints, third-party datasets, and model weights are intentionally not
  redistributed.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your accelerator before installing
the remaining dependencies when CUDA-specific wheels are needed.

## Verification

```bash
python -m pytest -q
```

The tests require no datasets or pretrained checkpoints. Full experiments do:
each collector exposes its inputs and frozen protocol through `--help`, for
example:

```bash
python scripts/collect_predictor_dataset.py --help
python scripts/collect_bert_mlm_scaling.py --help
python scripts/collect_qwen_cfs.py --help
python scripts/controlled_gpt2/collect_gpt2_cfs.py --help
```

Start with [`docs/protocols/CFS_CORE_ALGORITHM_PSEUDOCODE.md`](docs/protocols/CFS_CORE_ALGORITHM_PSEUDOCODE.md)
for the canonical algorithm and
[`docs/protocols/REPRODUCIBILITY_AND_RELEASE.md`](docs/protocols/REPRODUCIBILITY_AND_RELEASE.md)
for protocol invariants and artifact boundaries.

## Scope and interpretation

- Raw Oracle measures the best available substitute and grows with the set of
  available paths; Matched-k is the candidate-count control.
- CFS values based on different output spaces (classification, masked-token,
  and next-token KL) should not be compared as identical absolute scales.
- Logical head budgets or masked dense execution are not wall-clock speedups.
  The online execution check establishes consistency, not a sparse-kernel
  systems claim.
- The fixed-capacity experiments establish an association between functional
  organization and language-modeling performance; they do not by themselves
  prove that directly optimizing CFS structure causes better performance.

## Citation

```bibtex
@article{chen2026effective,
  title   = {Effective Does Not Mean Useful: Conditional Functional
             Substitutability for Redundancy and Scaling in Transformers},
  author  = {Chen, Jiaheng and Li, Jiaxing and Xiao, Yucheng and Cai, Xinyong
             and Bu, Juncheng and Yu, Lan and Zhang, Tinghe},
  journal = {arXiv preprint arXiv:2609.39259},
  year    = {2026}
}
```

No software license is declared in this initial release. Third-party datasets
and checkpoints remain governed by their original licenses.
