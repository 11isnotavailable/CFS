"""Collect multi-position CFS interventions for controlled GPT-2 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from gpt2_head_intervention import GPT2HeadGate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", required=True)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--alpha-min", type=float, default=0.0)
    p.add_argument("--alpha-max", type=float, default=3.0)
    p.add_argument("--alpha-steps", type=int, default=16)
    p.add_argument("--variant-batch-size", type=int, default=64)
    p.add_argument("--shard-size", type=int, default=2)
    p.add_argument("--prediction-positions", default="random:8")
    p.add_argument("--position-seed", type=int, default=20260802)
    p.add_argument("--minimum-position", type=int, default=7)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def selected_logits(model, ids, positions):
    hidden = model.transformer(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    return model.lm_head(hidden.index_select(1, positions))


def token_loss(model, ids, chunk=32):
    hidden = model.transformer(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    total, count = 0.0, 0
    for start in range(0, hidden.shape[1] - 1, chunk):
        stop = min(start + chunk, hidden.shape[1] - 1)
        logits = model.lm_head(hidden[:, start:stop]).float()
        target = ids[:, start + 1:stop + 1]
        total += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum"))
        count += target.numel()
    return total, count


def kl(reference, variants):
    log_p = F.log_softmax(reference.float(), dim=-1)
    p = log_p.exp()
    per_position = (p * (log_p - F.log_softmax(variants.float(), dim=-1))).sum(-1)
    return per_position.clamp_min(0).mean(-1)


def make_masks(heads, alphas):
    rows, metadata = [], []
    for source in range(heads):
        for representative in range(heads):
            if representative == source:
                continue
            for alpha_index, alpha in enumerate(alphas):
                row = torch.ones(heads)
                row[source] = 0
                row[representative] = alpha
                rows.append(row)
                metadata.append((source, representative, alpha_index))
    return torch.stack(rows), metadata


def run_masked(model, gate, ids, rows, reference, batch_size, positions):
    errors = []
    for start in range(0, len(rows), batch_size):
        local = rows[start:start + batch_size].to(ids.device)
        gate.set_scales(local)
        batch = ids.expand(len(local), -1)
        with torch.inference_mode():
            logits = selected_logits(model, batch, positions)
            errors.append(kl(reference, logits).cpu())
    gate.set_scales(None)
    return torch.cat(errors)


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True,
    ).eval().to(device)
    model.requires_grad_(False)
    depth, heads = int(model.config.n_layer), int(model.config.n_head)
    selected = [int(x) for x in args.layers.split(",")]
    if any(x < 0 or x >= depth for x in selected):
        raise ValueError(f"Invalid layers {selected} for depth {depth}")
    alpha_grid = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    substitution_masks, _ = make_masks(heads, alpha_grid)
    deletion_masks = torch.ones(heads, heads)
    deletion_masks[torch.arange(heads), torch.arange(heads)] = 0
    contexts = np.load(args.contexts, allow_pickle=False)["input_ids"][:args.num_samples]
    context_length = int(contexts.shape[1] - 1)
    if args.prediction_positions.startswith("random:"):
        count = int(args.prediction_positions.split(":", 1)[1])
        candidates = np.arange(args.minimum_position, context_length, dtype=np.int64)
        rng = np.random.default_rng(args.position_seed)
        schedule = np.stack([
            np.sort(rng.choice(candidates, size=count, replace=False))
            for _ in range(len(contexts))
        ])
    else:
        positions = [int(x) for x in args.prediction_positions.split(",")]
        schedule = np.tile(np.asarray(positions, dtype=np.int64), (len(contexts), 1))
    completion_path = args.model.parent / "completion.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    metadata = {
        "name": args.name, "model": str(args.model),
        "parameters": sum(p.numel() for p in model.parameters()),
        "depth": depth, "heads": heads,
        "hidden_size": int(model.config.n_embd),
        "head_dim": int(model.config.n_embd) // heads,
        "layers": selected, "samples": len(contexts),
        "sequence_length": context_length,
        "alpha_grid": alpha_grid.tolist(),
        "prediction_position_spec": args.prediction_positions,
        "position_seed": args.position_seed,
        "minimum_position": args.minimum_position,
        "prediction_position_schedule": schedule.tolist(),
        "prediction_position_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "contexts_sha256": hashlib.sha256(args.contexts.read_bytes()).hexdigest(),
        "intervention": "GPT-2 attention head AV branch immediately before c_proj",
        "cfs_output": "mean next-token-distribution KL over fixed random-eight positions",
        "training_final_validation_loss": completion.get("final_validation_loss"),
        "training_tokens": completion.get("tokens_trained"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    started = time.perf_counter()
    for shard_start in range(0, len(contexts), args.shard_size):
        shard_index = shard_start // args.shard_size
        output = args.output_dir / f"shard_{shard_index:05d}.npz"
        if output.exists():
            continue
        records = []
        for sample_index in range(shard_start, min(shard_start + args.shard_size, len(contexts))):
            full_ids = torch.from_numpy(contexts[sample_index].astype(np.int64)).unsqueeze(0).to(device)
            ids = full_ids[:, :-1]
            positions = torch.from_numpy(schedule[sample_index]).to(device)
            with torch.inference_mode():
                reference = selected_logits(model, ids, positions)
                loss_sum, token_count = token_loss(model, full_ids)
            layer_drop, layer_sub, layer_alpha = [], [], []
            for layer in selected:
                gate = GPT2HeadGate(model, layer)
                with torch.inference_mode():
                    dense = selected_logits(model, ids, positions)
                gate.set_scales(torch.ones(heads, device=device))
                with torch.inference_mode():
                    ones = selected_logits(model, ids, positions)
                dense_error = float(torch.max(torch.abs(reference.float() - dense.float())))
                ones_error = float(torch.max(torch.abs(reference.float() - ones.float())))
                if max(dense_error, ones_error) > 2e-2:
                    raise RuntimeError(f"Dense equivalence failed: {dense_error}, {ones_error}")
                gate.set_scales(None)
                drops = run_masked(model, gate, ids, deletion_masks, reference,
                                   args.variant_batch_size, positions)
                errors = run_masked(model, gate, ids, substitution_masks, reference,
                                    args.variant_batch_size, positions)
                matrix = torch.full((heads, heads), float("nan"))
                best_alpha = torch.full((heads, heads), float("nan"))
                cursor = 0
                for source in range(heads):
                    for representative in range(heads):
                        if source == representative:
                            continue
                        values = errors[cursor:cursor + len(alpha_grid)]
                        index = int(values.argmin())
                        matrix[source, representative] = values[index]
                        best_alpha[source, representative] = alpha_grid[index]
                        cursor += len(alpha_grid)
                layer_drop.append(drops.numpy())
                layer_sub.append(matrix.numpy())
                layer_alpha.append(best_alpha.numpy())
                gate.close()
            records.append({
                "sample": sample_index, "loss_sum": loss_sum, "token_count": token_count,
                "single_drop": np.stack(layer_drop),
                "substitution": np.stack(layer_sub),
                "best_alpha": np.stack(layer_alpha),
            })
            print(f"sample={sample_index + 1}/{len(contexts)} elapsed_min={(time.perf_counter()-started)/60:.2f}", flush=True)
        np.savez_compressed(
            output,
            sample=np.asarray([r["sample"] for r in records]),
            loss_sum=np.asarray([r["loss_sum"] for r in records]),
            token_count=np.asarray([r["token_count"] for r in records]),
            single_drop=np.stack([r["single_drop"] for r in records]),
            substitution=np.stack([r["substitution"] for r in records]),
            best_alpha=np.stack([r["best_alpha"] for r in records]),
        )
        print(f"saved={output}", flush=True)
    print(f"peak_memory_gib={torch.cuda.max_memory_allocated()/2**30:.3f}", flush=True)


if __name__ == "__main__":
    main()
