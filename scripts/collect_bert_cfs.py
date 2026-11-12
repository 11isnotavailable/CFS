"""Collect BERT CFS oracle labels and compact observability features.

The output intentionally matches the vision collectors closely enough to use
``train_pair_score_predictor.py`` unchanged.  Interventions are exact head
gates before BERT's shared output projection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding

from src.text_head_intervention import (
    clear_bert_interventions,
    make_bert_intervenable,
    post_wo_contributions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--task", choices=("sst2", "mnli"), default="sst2")
    parser.add_argument("--split", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=20260801)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant-batch-size", type=int, default=128)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-steps", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def configure_cache() -> None:
    root = PROJECT_ROOT / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "datasets"))


def masked_moments(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Token moments for [B,T,D] or [B,H,T,D]."""
    if value.ndim == 3:
        weight = mask[:, :, None].to(value.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (value * weight).sum(dim=1) / count
        var = ((value - mean[:, None]) ** 2 * weight).sum(dim=1) / count
    else:
        weight = mask[:, None, :, None].to(value.dtype)
        count = weight.sum(dim=2).clamp_min(1.0)
        mean = (value * weight).sum(dim=2) / count
        var = ((value - mean[:, :, None]) ** 2 * weight).sum(dim=2) / count
    return mean, var.clamp_min(0).sqrt()


def global_features(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean, std = masked_moments(hidden.float(), mask)
    return torch.cat([hidden[:, 0].float(), mean, std], dim=-1)


def qk_features(query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    query = query.float()
    key = key.float()
    q_mean, q_std = masked_moments(query, mask)
    k_mean, k_std = masked_moments(key, mask)
    q_norm = query.norm(dim=-1)
    k_norm = key.norm(dim=-1)
    weight = mask[:, None].to(q_norm.dtype)
    count = weight.sum(dim=-1).clamp_min(1.0)

    def norm_stats(norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (norm * weight).sum(dim=-1) / count
        var = ((norm - mean[:, :, None]) ** 2 * weight).sum(dim=-1) / count
        return mean, var.clamp_min(0).sqrt()

    qn_mean, qn_std = norm_stats(q_norm)
    kn_mean, kn_std = norm_stats(k_norm)
    scalars = torch.stack([qn_mean, qn_std, kn_mean, kn_std], dim=-1)
    return torch.cat(
        [query[:, :, 0], key[:, :, 0], q_mean, k_mean, q_std, k_std, scalars],
        dim=-1,
    )


def contribution_geometry(
    contribution: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask[:, None, :, None].to(contribution.dtype)
    count = weight.sum(dim=(2, 3)).clamp_min(1.0)
    norm = ((contribution.square() * weight).sum(dim=(2, 3)) / count).sqrt()
    difference = contribution[:, :, None] - contribution[:, None, :]
    pair_weight = mask[:, None, None, :, None].to(difference.dtype)
    pair_count = pair_weight.sum(dim=(3, 4)).clamp_min(1.0)
    distance = ((difference.square() * pair_weight).sum(dim=(3, 4)) / pair_count).sqrt()
    return distance, norm


def head_tensor_summary(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CLS, masked token moments, and norm moments for [B,H,T,D]."""
    mean, std = masked_moments(value.float(), mask)
    norm = value.float().norm(dim=-1)
    weight = mask[:, None].to(norm.dtype)
    count = weight.sum(dim=-1).clamp_min(1.0)
    norm_mean = (norm * weight).sum(dim=-1) / count
    norm_var = ((norm - norm_mean[:, :, None]) ** 2 * weight).sum(dim=-1) / count
    return torch.cat(
        [
            value[:, :, 0].float(),
            mean,
            std,
            norm_mean[:, :, None],
            norm_var.clamp_min(0).sqrt()[:, :, None],
        ],
        dim=-1,
    )


def kl_from_logits(reference: torch.Tensor, variant: torch.Tensor) -> torch.Tensor:
    reference_log = F.log_softmax(reference.float(), dim=-1)
    reference_prob = reference_log.exp()
    variant_log = F.log_softmax(variant.float(), dim=-1)
    return (reference_prob * (reference_log - variant_log)).sum(dim=-1).clamp_min(0.0)


def variant_logits(
    model,
    wrapper,
    inputs: dict[str, torch.Tensor],
    sample_index: torch.Tensor,
    scales: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    results = []
    for begin in range(0, len(sample_index), chunk_size):
        end = min(begin + chunk_size, len(sample_index))
        index = sample_index[begin:end]
        wrapper.set_head_scales(scales[begin:end])
        batch = {key: value.index_select(0, index) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=next(model.parameters()).device.type,
            dtype=torch.float16,
            enabled=next(model.parameters()).device.type == "cuda",
        ):
            results.append(model(**batch).logits.float())
    wrapper.set_head_scales(None)
    return torch.cat(results, dim=0)


def intervention_oracle(
    model,
    wrapper,
    inputs: dict[str, torch.Tensor],
    reference_logits: torch.Tensor,
    alphas: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = reference_logits.shape[0]
    heads = wrapper.num_heads
    device = reference_logits.device
    sample_drop = torch.arange(batch, device=device).repeat_interleave(heads)
    source_drop = torch.arange(heads, device=device).repeat(batch)
    scales_drop = torch.ones(batch * heads, heads, device=device)
    scales_drop[torch.arange(batch * heads, device=device), source_drop] = 0.0
    drop_logits = variant_logits(
        model, wrapper, inputs, sample_drop, scales_drop, chunk_size
    )
    single_drop = kl_from_logits(
        reference_logits.index_select(0, sample_drop), drop_logits
    ).view(batch, heads)

    records = []
    for sample in range(batch):
        for source in range(heads):
            for candidate in range(heads):
                if source == candidate:
                    continue
                for alpha_index in range(len(alphas)):
                    records.append((sample, source, candidate, alpha_index))
    metadata = torch.tensor(records, device=device, dtype=torch.long)
    sample_index = metadata[:, 0]
    scales = torch.ones(len(metadata), heads, device=device)
    row = torch.arange(len(metadata), device=device)
    scales[row, metadata[:, 1]] = 0.0
    scales[row, metadata[:, 2]] = alphas[metadata[:, 3]]
    logits = variant_logits(
        model, wrapper, inputs, sample_index, scales, chunk_size
    )
    errors = kl_from_logits(reference_logits.index_select(0, sample_index), logits)
    errors = errors.view(batch, heads, heads - 1, len(alphas))
    best_error, best_index = errors.min(dim=-1)
    substitution = torch.full((batch, heads, heads), torch.nan, device=device)
    best_alpha = torch.full_like(substitution, torch.nan)
    for source in range(heads):
        candidates = [candidate for candidate in range(heads) if candidate != source]
        substitution[:, source, candidates] = best_error[:, source]
        best_alpha[:, source, candidates] = alphas[best_index[:, source]]
    return single_drop, substitution, best_alpha


def stack_records(records: list[dict], key: str) -> np.ndarray:
    return np.stack([record[key] for record in records]).astype(np.float32)


def save_oracle_shard(directory: Path, shard_index: int, records: list[dict]) -> None:
    output = directory / f"shard_{shard_index:05d}.npz"
    temporary = output.with_suffix(".tmp.npz")
    arrays = {
        "dataset_index": np.asarray([r["dataset_index"] for r in records], dtype=np.int64),
        "dataset_label": np.asarray([r["dataset_label"] for r in records], dtype=np.int64),
        "prediction": np.asarray([r["prediction"] for r in records], dtype=np.int64),
        "confidence": np.asarray([r["confidence"] for r in records], dtype=np.float32),
        "path": np.asarray([r["path"] for r in records]),
        "prehead": stack_records(records, "prehead")[:, None],
        "taylor": stack_records(records, "taylor")[:, None],
        "contribution_l2": stack_records(records, "contribution_l2")[:, None],
        "contribution_norm": stack_records(records, "contribution_norm")[:, None],
        "substitution": stack_records(records, "substitution")[:, None],
        "best_alpha": stack_records(records, "best_alpha")[:, None],
        "single_drop": stack_records(records, "single_drop")[:, None],
    }
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    configure_cache()
    args.oracle_dir.mkdir(parents=True, exist_ok=True)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device).eval()
    wrappers = make_bert_intervenable(model)
    if not 0 <= args.layer < len(wrappers):
        raise ValueError(f"Layer {args.layer} is outside [0, {len(wrappers) - 1}].")
    wrapper = wrappers[args.layer]

    dataset = (
        load_from_disk(str(args.dataset_path))
        if args.dataset_path is not None
        else load_dataset("nyu-mll/glue", args.task)
    )
    raw = dataset[args.split].shuffle(seed=args.shuffle_seed)
    stop = min(args.start + args.count, len(raw))
    selected = raw.select(range(args.start, stop))

    def tokenize(batch):
        if args.task == "sst2":
            result = tokenizer(
                batch["sentence"],
                truncation=True,
                max_length=args.max_length,
            )
        else:
            result = tokenizer(
                batch["premise"],
                batch["hypothesis"],
                truncation=True,
                max_length=args.max_length,
            )
        result["labels"] = batch["label"]
        result["dataset_index"] = batch["local_index"]
        return result

    selected = selected.add_column("local_index", list(range(args.start, stop)))
    tokenized = selected.map(
        tokenize,
        batched=True,
        remove_columns=selected.column_names,
    )
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(tokenized, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    alphas = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_steps, device=device)

    all_records: list[dict] = []
    feature_path: list[str] = []
    feature_global: list[np.ndarray] = []
    feature_level2: list[np.ndarray] = []
    feature_level6: list[np.ndarray] = []
    feature_level7: list[np.ndarray] = []
    shard_buffer: list[dict] = []
    shard_index = 0
    processed = 0

    for raw_batch in loader:
        dataset_index = raw_batch.pop("dataset_index").numpy()
        labels = raw_batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in raw_batch.items()}
        mask = inputs["attention_mask"]
        clear_bert_interventions(wrappers)
        wrapper.set_capture(True)
        scales = torch.ones(
            len(labels), wrapper.num_heads, device=device, requires_grad=True
        )
        wrapper.set_head_scales(scales)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
        gradient_ce = torch.autograd.grad(loss, scales, retain_graph=True)[0]
        predicted = logits.argmax(dim=-1)
        selected_logit = logits.float().gather(1, predicted[:, None]).sum()
        gradient_logit = torch.autograd.grad(selected_logit, scales)[0]
        capture = wrapper.capture
        if capture is None:
            raise RuntimeError("BERT attention capture failed.")
        hidden = capture.hidden.detach().float()
        query = capture.query.detach().float()
        key = capture.key.detach().float()
        aggregated = capture.aggregated.detach().float()
        wrapper.set_capture(False)
        wrapper.set_head_scales(None)
        contribution = post_wo_contributions(model, args.layer, aggregated).float()
        distances, norms = contribution_geometry(contribution, mask)
        global_value = global_features(hidden, mask)
        qk_value = qk_features(query, key, mask)
        contribution_value = head_tensor_summary(contribution, mask)
        level6_value = torch.cat([qk_value, contribution_value], dim=-1)
        gradients = torch.stack(
            [gradient_ce, gradient_ce.abs(), gradient_logit, gradient_logit.abs()], dim=-1
        ).float()
        single_drop, substitution, best_alpha = intervention_oracle(
            model,
            wrapper,
            inputs,
            logits.detach().float(),
            alphas,
            args.variant_batch_size,
        )
        confidence = logits.float().softmax(dim=-1).max(dim=-1).values

        for row in range(len(labels)):
            identifier = (
                f"{args.task}:{args.split}:{int(dataset_index[row])}"
            )
            record = {
                "dataset_index": int(dataset_index[row]),
                "dataset_label": int(labels[row]),
                "prediction": int(predicted[row]),
                "confidence": float(confidence[row].detach()),
                "path": identifier,
                "prehead": global_value[row].detach().cpu().numpy(),
                "taylor": gradient_ce[row].abs().detach().cpu().numpy(),
                "contribution_l2": distances[row].detach().cpu().numpy(),
                "contribution_norm": norms[row].detach().cpu().numpy(),
                "substitution": substitution[row].detach().cpu().numpy(),
                "best_alpha": best_alpha[row].detach().cpu().numpy(),
                "single_drop": single_drop[row].detach().cpu().numpy(),
            }
            all_records.append(record)
            shard_buffer.append(record)
            feature_path.append(identifier)
            feature_global.append(global_value[row].detach().cpu().numpy())
            feature_level2.append(qk_value[row].detach().cpu().numpy())
            feature_level6.append(level6_value[row].detach().cpu().numpy())
            feature_level7.append(
                torch.cat([level6_value[row], gradients[row]], dim=-1).detach().cpu().numpy()
            )
            if len(shard_buffer) == args.shard_size:
                save_oracle_shard(args.oracle_dir, shard_index, shard_buffer)
                shard_buffer = []
                shard_index += 1
        processed += len(labels)
        print(f"processed={processed}/{len(selected)}", flush=True)

    if shard_buffer:
        save_oracle_shard(args.oracle_dir, shard_index, shard_buffer)
    level2 = np.stack(feature_level2).astype(np.float32)
    level6 = np.stack(feature_level6).astype(np.float32)
    np.savez_compressed(
        args.features,
        paths=np.asarray(feature_path),
        global_features=np.stack(feature_global).astype(np.float32),
        level2=level2,
        level5=level2,
        level6=level6,
        level7=np.stack(feature_level7).astype(np.float32),
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "task": f"glue/{args.task}",
        "split": args.split,
        "start": args.start,
        "count": len(selected),
        "shuffle_seed": args.shuffle_seed,
        "layer": args.layer,
        "heads": wrapper.num_heads,
        "max_length": args.max_length,
        "alpha_grid": alphas.detach().cpu().tolist(),
        "intervention": "per-head gate after A@V and before shared W_O",
    }
    (args.oracle_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    args.features.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
