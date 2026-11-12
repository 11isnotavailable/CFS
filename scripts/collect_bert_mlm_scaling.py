"""Evaluate pretrained BERT-family MLM loss and conditional head substitution.

No parameter is trained.  A deterministic set of ordinary tokens is replaced
by ``[MASK]`` in each frozen text pair.  MLM loss is evaluated on the original
tokens, while CFS uses the mean KL divergence across the same masked positions.
Head gates are applied to ``A_i V_i`` immediately before the shared output
projection, exactly as in the BERT classification experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from src.text_head_intervention import clear_bert_interventions, make_bert_intervenable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation_matched")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--oracle-count", type=int, default=32)
    parser.add_argument("--loss-count", type=int, default=512)
    parser.add_argument("--mask-positions", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--variant-batch-size", type=int, default=32)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-steps", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=20260810)
    parser.add_argument("--mask-seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def configure_cache() -> None:
    root = PROJECT_ROOT / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "datasets"))


def load_mlm_model(model_name: str):
    """Load MLM weights, with a trusted-bin fallback for Google miniatures.

    Some official Google BERT miniature repositories predate safetensors.
    Transformers 5 refuses their PyTorch bins under torch<2.6.  These are
    explicitly named first-party checkpoints, so we load them with
    ``weights_only=True`` instead of weakening the global safety policy.
    """
    try:
        return AutoModelForMaskedLM.from_pretrained(model_name)
    except ValueError as error:
        if "torch.load" not in str(error) or not model_name.startswith("google/"):
            raise
        directory = Path(
            snapshot_download(
                model_name,
                allow_patterns=["config.json", "pytorch_model.bin"],
            )
        )
        config = AutoConfig.from_pretrained(directory)
        model = AutoModelForMaskedLM.from_config(config)
        state = torch.load(
            directory / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        important_missing = [
            key
            for key in missing
            if not key.endswith("position_ids")
        ]
        if important_missing:
            raise RuntimeError(f"Missing trusted checkpoint weights: {important_missing}")
        print(
            json.dumps(
                {
                    "trusted_bin_fallback": model_name,
                    "unexpected_keys": unexpected,
                }
            ),
            flush=True,
        )
        return model


def masked_batch(
    tokenizer,
    rows: list[dict],
    sample_ids: list[int],
    max_length: int,
    positions_per_sample: int,
    mask_seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [row["premise"] for row in rows],
        [row["hypothesis"] for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    original = encoded["input_ids"].clone()
    position_rows = []
    target_rows = []
    special_ids = set(tokenizer.all_special_ids)
    for offset, sample_id in enumerate(sample_ids):
        valid = [
            index
            for index, (token, mask) in enumerate(
                zip(original[offset].tolist(), encoded["attention_mask"][offset].tolist())
            )
            if mask and token not in special_ids
        ]
        if len(valid) < positions_per_sample:
            raise ValueError(
                f"Sample {sample_id} has only {len(valid)} maskable tokens."
            )
        rng = np.random.default_rng(mask_seed + int(sample_id))
        chosen = np.sort(rng.choice(valid, size=positions_per_sample, replace=False))
        positions = torch.as_tensor(chosen, dtype=torch.long)
        targets = original[offset].index_select(0, positions)
        encoded["input_ids"][offset, positions] = tokenizer.mask_token_id
        position_rows.append(positions)
        target_rows.append(targets)
    inputs = {name: value.to(device) for name, value in encoded.items()}
    return inputs, torch.stack(position_rows).to(device), torch.stack(target_rows).to(device)


def choose_eligible_samples(
    raw,
    tokenizer,
    order: np.ndarray,
    required: int,
    max_length: int,
    positions_per_sample: int,
) -> list[int]:
    """Choose a deterministic common subset with enough ordinary tokens."""
    selected = []
    special_ids = set(tokenizer.all_special_ids)
    for index in order.tolist():
        row = raw[int(index)]
        encoded = tokenizer(
            row["premise"],
            row["hypothesis"],
            truncation=True,
            max_length=max_length,
        )
        maskable = sum(token not in special_ids for token in encoded["input_ids"])
        if maskable >= positions_per_sample:
            selected.append(int(index))
        if len(selected) == required:
            return selected
    raise RuntimeError(
        f"Only found {len(selected)} eligible samples; required {required}."
    )


def selected_mlm_logits(
    model,
    inputs: dict[str, torch.Tensor],
    positions: torch.Tensor,
) -> torch.Tensor:
    output = model.bert(**inputs, return_dict=True).last_hidden_state
    row = torch.arange(len(output), device=output.device)[:, None]
    selected = output[row, positions]
    return model.cls(selected).float()


def outcome_kl(reference: torch.Tensor, variant: torch.Tensor) -> torch.Tensor:
    reference_log = F.log_softmax(reference.float(), dim=-1)
    variant_log = F.log_softmax(variant.float(), dim=-1)
    token_kl = (reference_log.exp() * (reference_log - variant_log)).sum(dim=-1)
    return token_kl.clamp_min(0).mean(dim=-1)


def variant_logits(
    model,
    wrapper,
    inputs: dict[str, torch.Tensor],
    positions: torch.Tensor,
    sample_index: torch.Tensor,
    scales: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    results = []
    for begin in range(0, len(sample_index), chunk_size):
        end = min(begin + chunk_size, len(sample_index))
        index = sample_index[begin:end]
        wrapper.set_head_scales(scales[begin:end])
        batch = {name: value.index_select(0, index) for name, value in inputs.items()}
        batch_positions = positions.index_select(0, index)
        with torch.inference_mode(), torch.autocast(
            device_type=next(model.parameters()).device.type,
            dtype=torch.float16,
            enabled=next(model.parameters()).device.type == "cuda",
        ):
            results.append(selected_mlm_logits(model, batch, batch_positions).cpu())
    wrapper.set_head_scales(None)
    return torch.cat(results, dim=0)


def intervention_oracle(
    model,
    wrapper,
    inputs: dict[str, torch.Tensor],
    positions: torch.Tensor,
    reference_logits: torch.Tensor,
    alphas: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(reference_logits)
    heads = wrapper.num_heads
    device = next(model.parameters()).device

    sample_drop = torch.arange(batch, device=device).repeat_interleave(heads)
    source_drop = torch.arange(heads, device=device).repeat(batch)
    scales_drop = torch.ones(batch * heads, heads, device=device)
    scales_drop[torch.arange(batch * heads, device=device), source_drop] = 0
    drop_logits = variant_logits(
        model, wrapper, inputs, positions, sample_drop, scales_drop, chunk_size
    )
    drops = outcome_kl(reference_logits.index_select(0, sample_drop.cpu()), drop_logits)
    drops = drops.view(batch, heads)

    records = [
        (sample, source, candidate, alpha_index)
        for sample in range(batch)
        for source in range(heads)
        for candidate in range(heads)
        if source != candidate
        for alpha_index in range(len(alphas))
    ]
    metadata = torch.as_tensor(records, dtype=torch.long, device=device)
    sample_index = metadata[:, 0]
    scales = torch.ones(len(metadata), heads, device=device)
    row = torch.arange(len(metadata), device=device)
    scales[row, metadata[:, 1]] = 0
    scales[row, metadata[:, 2]] = alphas[metadata[:, 3]]
    logits = variant_logits(
        model, wrapper, inputs, positions, sample_index, scales, chunk_size
    )
    errors = outcome_kl(reference_logits.index_select(0, sample_index.cpu()), logits)
    errors = errors.view(batch, heads, heads - 1, len(alphas))
    best_error, best_index = errors.min(dim=-1)
    substitution = torch.full((batch, heads, heads), torch.nan)
    best_alpha = torch.full_like(substitution, torch.nan)
    alpha_cpu = alphas.cpu()
    for source in range(heads):
        candidates = [candidate for candidate in range(heads) if candidate != source]
        substitution[:, source, candidates] = best_error[:, source]
        best_alpha[:, source, candidates] = alpha_cpu[best_index[:, source]]
    return drops, substitution, best_alpha


def main() -> None:
    args = parse_args()
    configure_cache()
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = load_mlm_model(args.model).to(device).eval()
    wrappers = make_bert_intervenable(model)
    layer = args.layer if args.layer >= 0 else len(wrappers) - 1
    if not 0 <= layer < len(wrappers):
        raise ValueError(f"Layer {layer} is outside [0, {len(wrappers)-1}].")
    wrapper = wrappers[layer]
    alphas = torch.linspace(
        args.alpha_min, args.alpha_max, args.alpha_steps, device=device
    )

    raw = load_from_disk(str(args.dataset_path))[args.split]
    order = np.random.default_rng(args.sample_seed).permutation(len(raw))
    required = max(args.oracle_count, args.loss_count)
    selected_ids = choose_eligible_samples(
        raw,
        tokenizer,
        order,
        required,
        args.max_length,
        args.mask_positions,
    )

    loss_sum = 0.0
    token_count = 0
    for begin in range(0, args.loss_count, args.batch_size):
        ids = selected_ids[begin : min(begin + args.batch_size, args.loss_count)]
        rows = [raw[int(index)] for index in ids]
        inputs, positions, targets = masked_batch(
            tokenizer,
            rows,
            ids,
            args.max_length,
            args.mask_positions,
            args.mask_seed,
            device,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = selected_mlm_logits(model, inputs, positions)
        loss_sum += float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        )
        token_count += int(targets.numel())

    sample_ids_all = []
    drop_parts = []
    substitution_parts = []
    alpha_parts = []
    for begin in range(0, args.oracle_count, args.batch_size):
        ids = selected_ids[begin : min(begin + args.batch_size, args.oracle_count)]
        rows = [raw[int(index)] for index in ids]
        inputs, positions, _ = masked_batch(
            tokenizer,
            rows,
            ids,
            args.max_length,
            args.mask_positions,
            args.mask_seed,
            device,
        )
        clear_bert_interventions(wrappers)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            reference = selected_mlm_logits(model, inputs, positions).cpu()
        drops, substitution, best_alpha = intervention_oracle(
            model,
            wrapper,
            inputs,
            positions,
            reference,
            alphas,
            args.variant_batch_size,
        )
        sample_ids_all.extend(ids)
        drop_parts.append(drops.numpy())
        substitution_parts.append(substitution.numpy())
        alpha_parts.append(best_alpha.numpy())
        print(
            f"oracle={min(begin + len(ids), args.oracle_count)}/{args.oracle_count}",
            flush=True,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = model.config
    metadata = {
        "model": args.model,
        "parameters": parameter_count,
        "depth": int(config.num_hidden_layers),
        "heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_attention_heads),
        "head_dim": int(config.hidden_size // config.num_attention_heads),
        "hidden_size": int(config.hidden_size),
        "layers": [layer],
        "depth_fractions": [(layer + 1) / config.num_hidden_layers],
        "dataset": str(args.dataset_path),
        "split": args.split,
        "oracle_count": args.oracle_count,
        "loss_count": args.loss_count,
        "mask_positions": args.mask_positions,
        "max_length": args.max_length,
        "sample_seed": args.sample_seed,
        "mask_seed": args.mask_seed,
        "sample_ids": sample_ids_all,
        "alpha_grid": alphas.cpu().tolist(),
        "intervention": "query-head A_iV_i gate before shared W_O",
        "outcome": "mean KL across deterministic masked-token positions",
        "training": "none; pretrained MLM checkpoint evaluated as-is",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "shard_00000.npz",
        sample_id=np.asarray(sample_ids_all, dtype=np.int64),
        single_drop=np.concatenate(drop_parts)[:, None],
        substitution=np.concatenate(substitution_parts)[:, None],
        best_alpha=np.concatenate(alpha_parts)[:, None],
        loss_sum=np.asarray([loss_sum], dtype=np.float64),
        token_count=np.asarray([token_count], dtype=np.int64),
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "layer": layer,
                "mlm_loss": loss_sum / token_count,
                "pseudo_perplexity": math.exp(min(loss_sum / token_count, 30)),
                "oracle_samples": args.oracle_count,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
