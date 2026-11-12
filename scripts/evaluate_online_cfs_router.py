"""Evaluate CFS routing computed online from the routed layer's own input."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache" / "torch")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader

from scripts.evaluate_dynamic_cfs_execution import (
    FrozenPathDataset,
    dense_kl,
    load_targets,
    remap_paths,
)
from scripts.run_experiment0 import configure_cache
from src.head_intervention import make_intervenable
from src.model_loading import create_timm_model
from src.runtime_cfs_router import RuntimeCFSRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--keep-counts", default="2,3,4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--expected-predictions",
        type=Path,
        help="Optional offline predictions used to verify exact mask equivalence.",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    configure_cache()
    device = torch.device(args.device)
    paths_frozen, targets = load_targets(args.oracle_dir)
    expected = None
    if args.expected_predictions is not None:
        expected = np.load(args.expected_predictions)
        if not np.array_equal(paths_frozen.astype(str), expected["paths"].astype(str)):
            raise ValueError("Expected predictions do not align with oracle paths")
    paths = remap_paths(paths_frozen, args.path_root)
    model, _ = create_timm_model(args.model, args.checkpoint)
    model.eval().to(device)
    if args.fp16:
        model.half()
    wrappers = make_intervenable(model)
    wrapper = wrappers[args.layer]
    keep_counts = [int(value) for value in args.keep_counts.split(",")]
    controllers = {
        keep: RuntimeCFSRouter.from_checkpoint(args.router_checkpoint, keep).to(device)
        for keep in keep_counts
    }
    transform = create_transform(
        **resolve_model_data_config(model), is_training=False
    )
    loader = DataLoader(
        FrozenPathDataset(paths, targets, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    dtype = torch.float16 if args.fp16 else torch.float32
    totals = {
        keep: {"correct": 0, "agreement": 0, "kl": 0.0, "samples": 0,
               "route_seconds": 0.0, "mask_mismatch": 0}
        for keep in keep_counts
    }
    dense_correct = 0
    dense_seconds = 0.0
    rows = []
    for images, batch_targets, indices in loader:
        images = images.to(device, dtype=dtype, non_blocking=True)
        batch_targets = batch_targets.to(device)
        wrapper.set_runtime_router(None)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            reference = model(images)
        synchronize(device)
        dense_seconds += time.perf_counter() - started
        dense_prediction = reference.argmax(dim=-1)
        dense_correct += int((dense_prediction == batch_targets).sum())
        for keep, controller in controllers.items():
            wrapper.set_runtime_router(controller)
            synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                variant = model(images)
            synchronize(device)
            elapsed = time.perf_counter() - started
            prediction = variant.argmax(dim=-1)
            kl = dense_kl(reference, variant)
            state = totals[keep]
            state["correct"] += int((prediction == batch_targets).sum())
            state["agreement"] += int((prediction == dense_prediction).sum())
            state["kl"] += float(kl.sum())
            state["samples"] += len(images)
            state["route_seconds"] += elapsed
            if expected is not None:
                expected_mask = torch.from_numpy(
                    expected[f"keep{keep}_dynamic_cfs_router"][indices.numpy()]
                ).to(device)
                state["mask_mismatch"] += int(
                    (controller.last_mask != expected_mask).any(dim=-1).sum()
                )
            for local, sample_index in enumerate(indices.tolist()):
                rows.append(
                    {
                        "sample": sample_index,
                        "keep": keep,
                        "correct": int(prediction[local] == batch_targets[local]),
                        "dense_agreement": int(
                            prediction[local] == dense_prediction[local]
                        ),
                        "kl_to_dense": float(kl[local]),
                    }
                )
    wrapper.set_runtime_router(None)
    count = len(paths)
    summary = [
        {
            "keep": len(controllers[next(iter(controllers))].masks[0]),
            "policy": "dense",
            "accuracy": dense_correct / count,
            "dense_agreement": 1.0,
            "mean_kl_to_dense": 0.0,
            "seconds": dense_seconds,
            "relative_runtime": 1.0,
            "mask_mismatch": 0,
        }
    ]
    # Correct dense keep count is the model head count, not subset mask length.
    summary[0]["keep"] = wrapper.num_heads
    for keep, state in totals.items():
        summary.append(
            {
                "keep": keep,
                "policy": "online_dynamic_cfs_router",
                "accuracy": state["correct"] / count,
                "dense_agreement": state["agreement"] / count,
                "mean_kl_to_dense": state["kl"] / count,
                "seconds": state["route_seconds"],
                "relative_runtime": state["route_seconds"] / dense_seconds,
                "mask_mismatch": state["mask_mismatch"],
            }
        )
    frame = pd.DataFrame(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "online_execution_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "online_execution_samples.csv", index=False
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "runtime_input": "same-layer pre-attention residual",
                "logical_budget_only": True,
                "timing_interpretation": (
                    "Dense fused QKV remains active; relative runtime measures router "
                    "overhead, not sparse-kernel acceleration."
                ),
                "images": count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
