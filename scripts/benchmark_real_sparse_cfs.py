"""Benchmark real selected-head execution on a local CUDA GPU.

This differs from logical masking: unselected QKV rows, attention heads, and
output-projection columns are not multiplied. Timings exclude data loading and
host-to-device transfer and include the forward-only router when requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

import torch

from src.model_loading import create_timm_model
from src.runtime_cfs_router import RuntimeCFSRouter
from src.sparse_head_attention import (
    PackedStaticHeadAttention,
    SelectedHeadAttention,
    StaticHeadSelector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="vit_base_patch16_224.augreg_in21k_ft_in1k"
    )
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--budgets", default="3,6,9")
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def cuda_benchmark(fn, warmup: int, iterations: int, repeats: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                fn()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)) / iterations)
    return values


def summarize(
    scope: str,
    policy: str,
    batch: int,
    keep: int,
    values: list[float],
    dense_ms: float,
    unique_subsets: int | None = None,
) -> dict[str, object]:
    median = statistics.median(values)
    return {
        "scope": scope,
        "policy": policy,
        "batch_size": batch,
        "keep_heads": keep,
        "median_ms": median,
        "min_ms": min(values),
        "max_ms": max(values),
        "relative_runtime": median / dense_ms,
        "speedup": dense_ms / median,
        "unique_subsets_last_batch": unique_subsets,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This benchmark requires CUDA events on a CUDA GPU")
    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32
    budgets = [int(value) for value in args.budgets.split(",")]
    batches = [int(value) for value in args.batch_sizes.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model, _ = create_timm_model(args.model)
    model.eval().to(device=device, dtype=dtype)
    block = model.blocks[args.layer]
    dense_attention = block.attn
    heads = int(dense_attention.num_heads)
    dim = int(dense_attention.qkv.in_features)
    tokens = int(model.patch_embed.num_patches) + int(model.num_prefix_tokens)
    image_size = model.patch_embed.img_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    rows: list[dict[str, object]] = []
    correctness = {}
    for batch in batches:
        hidden = torch.randn(batch, tokens, dim, device=device, dtype=dtype)
        images = torch.randn(
            batch, 3, image_size[0], image_size[1], device=device, dtype=dtype
        )

        block.attn = dense_attention
        dense_micro_values = cuda_benchmark(
            lambda: dense_attention(hidden), args.warmup, args.iterations, args.repeats
        )
        dense_micro = statistics.median(dense_micro_values)
        rows.append(summarize("attention", "dense", batch, heads,
                              dense_micro_values, dense_micro))
        dense_e2e_values = cuda_benchmark(
            lambda: model(images), args.warmup, args.iterations, args.repeats
        )
        dense_e2e = statistics.median(dense_e2e_values)
        rows.append(summarize("end_to_end", "dense", batch, heads,
                              dense_e2e_values, dense_e2e))

        all_heads = SelectedHeadAttention(
            dense_attention,
            StaticHeadSelector(torch.ones(heads, dtype=torch.bool, device=device)),
        ).eval().to(device)
        with torch.inference_mode():
            dense_value = dense_attention(hidden)
            selected_value = all_heads(hidden)
        correctness[f"batch{batch}_all_head_max_abs"] = float(
            (dense_value - selected_value).abs().max()
        )

        for keep in budgets:
            static_mask = torch.zeros(heads, dtype=torch.bool, device=device)
            static_mask[:keep] = True
            static_sparse = SelectedHeadAttention(
                dense_attention, StaticHeadSelector(static_mask)
            ).eval().to(device)
            static_micro_values = cuda_benchmark(
                lambda: static_sparse(hidden), args.warmup,
                args.iterations, args.repeats,
            )
            rows.append(summarize(
                "attention", "real_sparse_static", batch, keep,
                static_micro_values, dense_micro, static_sparse.last_unique_subsets,
            ))
            block.attn = static_sparse
            static_e2e_values = cuda_benchmark(
                lambda: model(images), args.warmup, args.iterations, args.repeats
            )
            rows.append(summarize(
                "end_to_end", "real_sparse_static", batch, keep,
                static_e2e_values, dense_e2e, static_sparse.last_unique_subsets,
            ))

            packed_static = PackedStaticHeadAttention(
                dense_attention, static_mask
            ).eval().to(device)
            packed_micro_values = cuda_benchmark(
                lambda: packed_static(hidden), args.warmup,
                args.iterations, args.repeats,
            )
            rows.append(summarize(
                "attention", "real_sparse_prepacked_ceiling", batch, keep,
                packed_micro_values, dense_micro, 1,
            ))
            block.attn = packed_static
            packed_e2e_values = cuda_benchmark(
                lambda: model(images), args.warmup, args.iterations, args.repeats
            )
            rows.append(summarize(
                "end_to_end", "real_sparse_prepacked_ceiling", batch, keep,
                packed_e2e_values, dense_e2e, 1,
            ))

            if args.router_checkpoint is not None:
                router = RuntimeCFSRouter.from_checkpoint(
                    args.router_checkpoint, keep=keep, map_location="cpu"
                ).eval().to(device)
                dynamic_sparse = SelectedHeadAttention(dense_attention, router).eval().to(device)
                dynamic_micro_values = cuda_benchmark(
                    lambda: dynamic_sparse(hidden), args.warmup,
                    args.iterations, args.repeats,
                )
                rows.append(summarize(
                    "attention", "real_sparse_dynamic_router", batch, keep,
                    dynamic_micro_values, dense_micro,
                    dynamic_sparse.last_unique_subsets,
                ))
                block.attn = dynamic_sparse
                dynamic_e2e_values = cuda_benchmark(
                    lambda: model(images), args.warmup,
                    args.iterations, args.repeats,
                )
                rows.append(summarize(
                    "end_to_end", "real_sparse_dynamic_router", batch, keep,
                    dynamic_e2e_values, dense_e2e,
                    dynamic_sparse.last_unique_subsets,
                ))

        block.attn = dense_attention

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "wallclock_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "timing_scope": "CUDA compute only; data loading and H2D excluded",
        "real_sparse_definition": (
            "unselected QKV rows, SDPA heads, and W_O input columns are not computed"
        ),
        "correctness": correctness,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
