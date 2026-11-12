"""Collect scalable pre-head features and functional-substitution labels.

Each output shard is self-contained and written atomically. Existing shards
are skipped, making long RTX 4060 collection runs resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from timm.data import ImageNetInfo, create_transform, resolve_model_data_config

from scripts.run_experiment0 import (
    build_substitution_masks,
    configure_cache,
    contribution_norms,
    current_distance_matrices,
    make_dataset,
    run_variants,
    substitution_error_cube,
    summarize_substitutions,
    verification_input,
    verify_dense_equivalence,
)
from src.head_intervention import clear_interventions, make_intervenable
from src.model_loading import create_timm_model, repartition_timm_attention_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--override-num-heads", type=int,
        help=(
            "Diagnostic only: repartition the fixed QKV/WO dimensions into this "
            "many heads. This is not a checkpoint trained at the new head count."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--indices-metadata",
        type=Path,
        help="Use the exact sample_indices from a previous collector metadata.",
    )
    parser.add_argument("--selection", choices=("contiguous", "spread"), default="spread")
    parser.add_argument(
        "--exclude-metadata",
        type=Path,
        action="append",
        help="Optional collector metadata whose sample_indices must be excluded.",
    )
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--variant-batch-size", type=int, default=128)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-steps", type=int, default=16)
    parser.add_argument(
        "--save-alpha-error-cube", action="store_true",
        help="Retain replacement KL for every source, representative, and alpha.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help='Comma-separated zero-based layers, e.g. "10", or "all".',
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def sample_indices(
    length: int,
    count: int,
    start: int,
    selection: str,
    excluded: set[int] | None = None,
) -> np.ndarray:
    if start < 0 or start >= length:
        raise ValueError(f"start-index {start} is outside dataset length {length}")
    excluded = excluded or set()
    candidates = np.asarray(
        [index for index in range(start, length) if index not in excluded],
        dtype=np.int64,
    )
    count = min(count, len(candidates))
    if selection == "contiguous":
        return candidates[:count]
    offsets = np.linspace(0, len(candidates) - 1, num=count, dtype=int)
    return candidates[offsets]


def evenly_subsample_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Take an order-preserving spread subset from a frozen index protocol."""
    count = min(count, len(indices))
    if count == len(indices):
        return indices
    offsets = np.linspace(0, len(indices) - 1, num=count, dtype=int)
    return indices[offsets]


def prehead_statistics(hidden: torch.Tensor) -> np.ndarray:
    """CLS, non-CLS token mean, and non-CLS token variance."""
    value = hidden[0].float()
    patches = value[1:]
    feature = torch.cat(
        [
            value[0],
            patches.mean(dim=0),
            patches.var(dim=0, unbiased=False),
        ]
    )
    return feature.cpu().numpy().astype(np.float16)


def target_for_path(path: str, synset_to_index: dict[str, int]) -> int:
    return int(synset_to_index.get(Path(path).parent.name, -1))


def parse_layers(value: str, depth: int) -> list[int]:
    if value == "all":
        return list(range(depth))
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or any(layer < 0 or layer >= depth for layer in layers):
        raise ValueError(f"Invalid layers {value!r} for depth {depth}")
    return layers


def collect_one(
    model: torch.nn.Module,
    wrappers,
    image: torch.Tensor,
    label: int,
    path: str,
    dataset_index: int,
    alphas: torch.Tensor,
    substitution_masks: torch.Tensor,
    substitution_metadata,
    variant_batch_size: int,
    synset_to_index: dict[str, int],
    selected_layers: list[int],
    save_alpha_error_cube: bool,
) -> dict[str, np.ndarray | int | float | str]:
    layers = len(selected_layers)
    heads = wrappers[0].num_heads

    clear_interventions(wrappers)
    for layer in selected_layers:
        wrapper = wrappers[layer]
        wrapper.capture_input = True
        wrapper.capture_contributions = True
    with torch.inference_mode():
        reference_logits = model(image)

    prehead = np.stack(
        [prehead_statistics(wrappers[layer].last_input) for layer in selected_layers]
    )
    contribution_l2 = np.empty((layers, heads, heads), dtype=np.float32)
    contribution_norm = np.empty((layers, heads), dtype=np.float32)
    for output_layer, layer in enumerate(selected_layers):
        wrapper = wrappers[layer]
        if wrapper.last_contributions is None:
            raise RuntimeError("Contribution capture failed.")
        contribution_l2[output_layer] = current_distance_matrices(
            wrapper.last_contributions
        )[0]
        contribution_norm[output_layer] = contribution_norms(
            wrapper.last_contributions
        )

    # A single backward pass supplies the strongest known cheap baseline.
    clear_interventions(wrappers)
    scales = [
        torch.ones(
            heads,
            device=image.device,
            dtype=image.dtype,
            requires_grad=True,
        )
        for _ in wrappers
    ]
    for wrapper, scale in zip(wrappers, scales):
        wrapper.set_head_scales(scale)
    logits = model(image)
    target = target_for_path(path, synset_to_index)
    loss_target = target if target >= 0 else int(reference_logits.argmax(dim=-1))
    loss = F.cross_entropy(
        logits.float(),
        torch.tensor([loss_target], device=image.device),
    )
    gradients = torch.autograd.grad(loss, scales)
    taylor_all = np.stack(
        [
            (gradient * scale).abs().float().detach().cpu().numpy()
            for gradient, scale in zip(gradients, scales)
        ]
    ).astype(np.float32)
    taylor = taylor_all[selected_layers]

    substitution = np.empty((layers, heads, heads), dtype=np.float32)
    best_alpha = np.empty((layers, heads, heads), dtype=np.float32)
    single_drop = np.empty((layers, heads), dtype=np.float32)
    alpha_error_cube = (
        np.empty((layers, heads, heads, len(alphas)), dtype=np.float32)
        if save_alpha_error_cube else None
    )

    for output_layer, layer in enumerate(selected_layers):
        wrapper = wrappers[layer]
        clear_interventions(wrappers)
        deletion_masks = torch.ones(heads, heads)
        deletion_masks[torch.arange(heads), torch.arange(heads)] = 0
        _, drop_errors = run_variants(
            model,
            wrapper,
            image,
            deletion_masks,
            reference_logits,
            variant_batch_size,
            1.0,
        )
        _, substitution_errors = run_variants(
            model,
            wrapper,
            image,
            substitution_masks,
            reference_logits,
            variant_batch_size,
            1.0,
        )
        matrix, alpha = summarize_substitutions(
            heads,
            substitution_metadata,
            substitution_errors,
        )
        substitution[output_layer] = matrix
        best_alpha[output_layer] = alpha
        single_drop[output_layer] = drop_errors.numpy()
        if alpha_error_cube is not None:
            alpha_error_cube[output_layer] = substitution_error_cube(
                heads, len(alphas), substitution_metadata, substitution_errors
            )

    clear_interventions(wrappers)
    record = {
        "dataset_index": dataset_index,
        "dataset_label": label,
        "imagenet_target": target,
        "prediction": int(reference_logits.argmax(dim=-1)),
        "confidence": float(reference_logits.float().softmax(dim=-1).max()),
        "path": path,
        "prehead": prehead,
        "taylor": taylor,
        "contribution_l2": contribution_l2,
        "contribution_norm": contribution_norm,
        "substitution": substitution,
        "best_alpha": best_alpha,
        "single_drop": single_drop,
    }
    if alpha_error_cube is not None:
        record["alpha_error_cube"] = alpha_error_cube
    return record


def save_shard(
    output: Path,
    records: list[dict],
    alpha_grid: np.ndarray,
) -> None:
    arrays = {
        "dataset_index": np.asarray([row["dataset_index"] for row in records]),
        "dataset_label": np.asarray([row["dataset_label"] for row in records]),
        "imagenet_target": np.asarray([row["imagenet_target"] for row in records]),
        "prediction": np.asarray([row["prediction"] for row in records]),
        "confidence": np.asarray([row["confidence"] for row in records]),
        "path": np.asarray([row["path"] for row in records]),
        "alpha_grid": alpha_grid,
    }
    fields = [
        "prehead",
        "taylor",
        "contribution_l2",
        "contribution_norm",
        "substitution",
        "best_alpha",
        "single_drop",
    ]
    if records and "alpha_error_cube" in records[0]:
        fields.append("alpha_error_cube")
    for name in fields:
        arrays[name] = np.stack([row[name] for row in records])
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    configure_cache()
    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 else torch.float32
    loaded_model, custom_class_to_idx = create_timm_model(
        args.model,
        args.checkpoint,
    )
    original_num_heads = int(loaded_model.blocks[0].attn.num_heads)
    checkpoint_trained_heads = None
    if args.checkpoint is not None:
        checkpoint_payload = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        if checkpoint_payload.get("trained_num_heads") is not None:
            checkpoint_trained_heads = int(checkpoint_payload["trained_num_heads"])
    if args.override_num_heads is not None:
        repartition_timm_attention_heads(loaded_model, args.override_num_heads)
    model = (
        loaded_model
        .eval()
        .to(device=device, dtype=dtype)
    )
    verify_image = verification_input(str(device), dtype)
    with torch.inference_mode():
        original_logits = model(verify_image)
    wrappers = make_intervenable(model)
    dense_error = verify_dense_equivalence(
        model, wrappers, verify_image, original_logits
    )
    tolerance = 2e-3 if args.fp16 else 2e-5
    if dense_error > tolerance:
        raise RuntimeError(f"Dense equivalence failed: {dense_error}")
    model.requires_grad_(False)

    transform = create_transform(
        **resolve_model_data_config(model),
        is_training=False,
    )
    dataset = make_dataset(args.data_root, transform)
    excluded: set[int] = set()
    if args.indices_metadata is not None:
        prior = json.loads(
            args.indices_metadata.read_text(encoding="utf-8")
        )
        indices = np.asarray(prior["sample_indices"], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(dataset)):
            raise ValueError("indices-metadata is incompatible with dataset")
        indices = evenly_subsample_indices(indices, args.num_samples)
    else:
        if args.exclude_metadata is not None:
            for metadata_path in args.exclude_metadata:
                prior = json.loads(metadata_path.read_text(encoding="utf-8"))
                excluded.update(int(index) for index in prior["sample_indices"])
        indices = sample_indices(
            len(dataset),
            args.num_samples,
            args.start_index,
            args.selection,
            excluded,
        )
    heads = wrappers[0].num_heads
    selected_layers = parse_layers(args.layers, len(wrappers))
    alphas = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    substitution_masks, substitution_metadata = build_substitution_masks(
        heads, alphas
    )
    if custom_class_to_idx is None:
        labels = ImageNetInfo().label_names()
        synset_to_index = {
            synset: index for index, synset in enumerate(labels)
        }
    else:
        synset_to_index = custom_class_to_idx

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "checkpoint": (
            str(args.checkpoint.resolve())
            if args.checkpoint is not None
            else None
        ),
        "data_root": str(args.data_root.resolve()),
        "dataset_size": len(dataset),
        "requested_samples": int(len(indices)),
        "sample_indices": indices.tolist(),
        "selection": args.selection,
        "excluded_indices": sorted(excluded),
        "indices_metadata": (
            str(args.indices_metadata.resolve())
            if args.indices_metadata is not None
            else None
        ),
        "layers": len(wrappers),
        "selected_layers": selected_layers,
        "heads": heads,
        "original_checkpoint_heads": original_num_heads,
        "checkpoint_trained_heads": (
            checkpoint_trained_heads
            if checkpoint_trained_heads is not None
            else original_num_heads
        ),
        "head_repartition_diagnostic": (
            args.override_num_heads is not None
            and args.override_num_heads
            != (
                checkpoint_trained_heads
                if checkpoint_trained_heads is not None
                else original_num_heads
            )
        ),
        "prehead_feature": "CLS + patch mean + patch variance",
        "alpha_grid": alphas.tolist(),
        "alpha_error_cube_saved": args.save_alpha_error_cube,
        "alpha_error_cube_shape": (
            [len(selected_layers), heads, heads, len(alphas)]
            if args.save_alpha_error_cube else None
        ),
        "dtype": str(dtype),
        "dense_equivalence_max_abs_error": dense_error,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    started = time.perf_counter()
    total_shards = int(np.ceil(len(indices) / args.shard_size))
    for shard_index in range(total_shards):
        begin = shard_index * args.shard_size
        stop = min(begin + args.shard_size, len(indices))
        output = args.output_dir / f"shard_{shard_index:05d}.npz"
        if output.exists():
            print(f"skip existing {output.name}")
            continue
        records = []
        for position in range(begin, stop):
            dataset_index = int(indices[position])
            image, label, path = dataset[dataset_index]
            record = collect_one(
                model=model,
                wrappers=wrappers,
                image=image.unsqueeze(0).to(device=device, dtype=dtype),
                label=int(label),
                path=str(path),
                dataset_index=dataset_index,
                alphas=alphas,
                substitution_masks=substitution_masks,
                substitution_metadata=substitution_metadata,
                variant_batch_size=args.variant_batch_size,
                synset_to_index=synset_to_index,
                selected_layers=selected_layers,
                save_alpha_error_cube=args.save_alpha_error_cube,
            )
            records.append(record)
            elapsed = time.perf_counter() - started
            completed = position + 1
            eta = elapsed / completed * (len(indices) - completed)
            print(
                f"sample={completed}/{len(indices)} index={dataset_index} "
                f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
            )
        save_shard(output, records, alphas.numpy())
        print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
