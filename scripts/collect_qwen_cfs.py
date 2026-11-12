"""Collect next-token CFS oracles and causal-LM loss for Qwen2.5 base models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.qwen_head_intervention import QwenHeadGate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-steps", type=int, default=16)
    parser.add_argument("--variant-batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=4)
    parser.add_argument(
        "--prediction-positions",
        default="final",
        help=(
            "Comma-separated zero-based positions, 'final', or 'random:N'. "
            "Random schedules are deterministic per context and shared across models."
        ),
    )
    parser.add_argument("--position-seed", type=int, default=20260802)
    parser.add_argument("--minimum-position", type=int, default=7)
    parser.add_argument(
        "--functional-units", type=int, default=None,
        help="Equal-width contiguous o_proj-input units; defaults to query heads.",
    )
    parser.add_argument(
        "--exclude-same-query-head", action="store_true",
        help="For split-head units, exclude representatives from the same original query head.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def selected_logits(
    model, input_ids: torch.Tensor, prediction_positions: torch.Tensor
) -> torch.Tensor:
    hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
    return model.lm_head(hidden.index_select(1, prediction_positions))


def token_loss(model, input_ids: torch.Tensor, chunk: int = 32) -> tuple[float, int]:
    hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
    total, count = 0.0, 0
    for start in range(0, hidden.shape[1] - 1, chunk):
        stop = min(start + chunk, hidden.shape[1] - 1)
        logits = model.lm_head(hidden[:, start:stop]).float()
        targets = input_ids[:, start + 1 : stop + 1]
        total += float(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum"))
        count += targets.numel()
    return total, count


def kl(reference: torch.Tensor, variants: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(reference.float(), dim=-1)
    p = log_p.exp()
    per_position = (p * (log_p - F.log_softmax(variants.float(), dim=-1))).sum(-1)
    return per_position.clamp_min(0).mean(-1)


def masks(
    heads: int, alphas: torch.Tensor, excluded: np.ndarray | None = None
) -> tuple[torch.Tensor, list[tuple[int,int,int]]]:
    rows, metadata = [], []
    for source in range(heads):
        for representative in range(heads):
            if representative == source:
                continue
            if excluded is not None and excluded[source, representative]:
                continue
            for alpha_index, alpha in enumerate(alphas):
                row = torch.ones(heads)
                row[source] = 0
                row[representative] = alpha
                rows.append(row)
                metadata.append((source, representative, alpha_index))
    return torch.stack(rows), metadata


def run_masked(
    model, gate, ids, scale_rows, reference, batch_size, prediction_positions
):
    errors = []
    for start in range(0, len(scale_rows), batch_size):
        rows = scale_rows[start : start + batch_size].to(ids.device)
        gate.set_scales(rows)
        batch = ids.expand(len(rows), -1)
        with torch.inference_mode():
            logits = selected_logits(model, batch, prediction_positions)
            errors.append(kl(reference, logits).cpu())
    gate.set_scales(None)
    return torch.cat(errors)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().to(device)
    model.requires_grad_(False)
    depth = int(model.config.num_hidden_layers)
    selected = [int(value) for value in args.layers.split(",")]
    if any(layer < 0 or layer >= depth for layer in selected):
        raise ValueError(f"Invalid layers {selected} for depth {depth}")
    query_heads = int(model.config.num_attention_heads)
    heads = query_heads if args.functional_units is None else args.functional_units
    if int(model.config.hidden_size) % heads:
        raise ValueError(f"hidden size is not divisible by {heads} functional units")
    excluded = None
    if args.exclude_same_query_head:
        if heads <= query_heads or heads % query_heads:
            raise ValueError("same-query exclusion requires an integer split of query heads")
        units_per_query = heads // query_heads
        query_owner = np.arange(heads) // units_per_query
        excluded = query_owner[:, None] == query_owner[None, :]
        np.fill_diagonal(excluded, False)
    alpha_grid = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    substitution_masks, substitution_metadata = masks(heads, alpha_grid, excluded)
    deletion_masks = torch.ones(heads, heads)
    deletion_masks[torch.arange(heads), torch.arange(heads)] = 0
    contexts = np.load(args.contexts, allow_pickle=False)["input_ids"][: args.num_samples]
    context_length = int(contexts.shape[1] - 1)
    if args.prediction_positions == "final":
        position_schedule = np.full(
            (len(contexts), 1), context_length - 1, dtype=np.int64
        )
    elif args.prediction_positions.startswith("random:"):
        position_count = int(args.prediction_positions.split(":", 1)[1])
        candidates = np.arange(args.minimum_position, context_length, dtype=np.int64)
        if position_count <= 0 or position_count > len(candidates):
            raise ValueError(
                f"Cannot sample {position_count} positions from {len(candidates)} candidates"
            )
        rng = np.random.default_rng(args.position_seed)
        position_schedule = np.stack([
            np.sort(rng.choice(candidates, size=position_count, replace=False))
            for _ in range(len(contexts))
        ])
    else:
        positions = [int(value) for value in args.prediction_positions.split(",")]
        if not positions or any(position < 0 or position >= context_length for position in positions):
            raise ValueError(f"Invalid prediction positions {positions} for length {context_length}")
        if len(set(positions)) != len(positions):
            raise ValueError(f"Duplicate prediction positions: {positions}")
        position_schedule = np.tile(np.asarray(positions, dtype=np.int64), (len(contexts), 1))
    schedule_digest = hashlib.sha256(position_schedule.tobytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model, "parameters": sum(p.numel() for p in model.parameters()),
        "layers": selected, "depth": depth, "heads": heads,
        "query_heads": query_heads,
        "functional_units": heads,
        "unit_width": int(model.config.hidden_size) // heads,
        "base_head_dim": int(model.config.hidden_size) // query_heads,
        "exclude_same_query_head": args.exclude_same_query_head,
        "kv_heads": int(model.config.num_key_value_heads),
        "samples": len(contexts), "sequence_length": context_length,
        "alpha_grid": alpha_grid.tolist(),
        "intervention": "equal-width contribution units at query-head AV before o_proj",
        "prediction_position_spec": args.prediction_positions,
        "position_seed": args.position_seed,
        "minimum_position": args.minimum_position,
        "prediction_position_schedule": position_schedule.tolist(),
        "prediction_position_schedule_sha256": schedule_digest,
        "cfs_output": "mean next-token-distribution KL at fixed context positions",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    for shard_start in range(0, len(contexts), args.shard_size):
        shard_index = shard_start // args.shard_size
        output = args.output_dir / f"shard_{shard_index:05d}.npz"
        if output.exists():
            continue
        records = []
        for sample_index in range(shard_start, min(shard_start + args.shard_size, len(contexts))):
            ids = torch.from_numpy(contexts[sample_index, :-1].astype(np.int64)).unsqueeze(0).to(device)
            prediction_positions = torch.from_numpy(position_schedule[sample_index]).to(device)
            target = int(contexts[sample_index, -1])
            with torch.inference_mode():
                reference = selected_logits(model, ids, prediction_positions)
                loss_sum, token_count = token_loss(model, torch.from_numpy(contexts[sample_index].astype(np.int64)).unsqueeze(0).to(device))
            layer_drop, layer_sub, layer_alpha = [], [], []
            for layer in selected:
                gate = QwenHeadGate(model, layer, num_units=heads)
                with torch.inference_mode():
                    dense = selected_logits(model, ids, prediction_positions)
                gate.set_scales(torch.ones(heads, device=device))
                with torch.inference_mode():
                    ones = selected_logits(model, ids, prediction_positions)
                dense_error = float(torch.max(torch.abs(reference.float() - dense.float())))
                ones_error = float(torch.max(torch.abs(reference.float() - ones.float())))
                if max(dense_error, ones_error) > 2e-2:
                    raise RuntimeError(f"Dense equivalence failed: {dense_error}, {ones_error}")
                gate.set_scales(None)
                drops = run_masked(
                    model, gate, ids, deletion_masks, reference,
                    args.variant_batch_size, prediction_positions,
                )
                errors = run_masked(
                    model, gate, ids, substitution_masks, reference,
                    args.variant_batch_size, prediction_positions,
                )
                matrix = torch.full((heads, heads), float("nan"))
                best_alpha = torch.full((heads, heads), float("nan"))
                cursor = 0
                for source in range(heads):
                    for representative in range(heads):
                        if representative == source:
                            continue
                        if excluded is not None and excluded[source, representative]:
                            continue
                        values = errors[cursor : cursor + len(alpha_grid)]
                        index = int(values.argmin())
                        matrix[source, representative] = values[index]
                        best_alpha[source, representative] = alpha_grid[index]
                        cursor += len(alpha_grid)
                layer_drop.append(drops.numpy())
                layer_sub.append(matrix.numpy())
                layer_alpha.append(best_alpha.numpy())
                gate.close()
            records.append({
                "sample": sample_index, "target": target,
                "prediction": int(reference[:, -1].argmax(-1)), "loss_sum": loss_sum,
                "token_count": token_count, "single_drop": np.stack(layer_drop),
                "substitution": np.stack(layer_sub), "best_alpha": np.stack(layer_alpha),
            })
            elapsed = (time.perf_counter() - started) / 60
            print(f"sample={sample_index + 1}/{len(contexts)} elapsed={elapsed:.1f}m", flush=True)
        np.savez_compressed(
            output,
            sample=np.asarray([r["sample"] for r in records]),
            target=np.asarray([r["target"] for r in records]),
            prediction=np.asarray([r["prediction"] for r in records]),
            loss_sum=np.asarray([r["loss_sum"] for r in records]),
            token_count=np.asarray([r["token_count"] for r in records]),
            single_drop=np.stack([r["single_drop"] for r in records]),
            substitution=np.stack([r["substitution"] for r in records]),
            best_alpha=np.stack([r["best_alpha"] for r in records]),
        )
        print(f"saved={output}", flush=True)
    print(f"peak_memory_gib={torch.cuda.max_memory_allocated()/2**30:.3f}")


if __name__ == "__main__":
    main()
