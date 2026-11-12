"""Collect matched non-functional head baselines for Experiment 1.5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.run_experiment0 import (
    configure_cache,
    current_distance_matrices,
    verification_input,
    verify_dense_equivalence,
)
from src.head_intervention import clear_interventions, make_intervenable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deit_small_patch16_224")
    return parser.parse_args()


def attention_statistics(attention: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return pairwise cosine distance and normalized entropy per head."""
    # [1, H, query, key]
    maps = attention[0].float()
    flat = maps.flatten(1)
    normalized = F.normalize(flat, dim=1)
    cosine_distance = 1.0 - normalized @ normalized.T
    entropy = -(maps.clamp_min(1e-12) * maps.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy.mean(dim=-1) / np.log(maps.shape[-1])
    return cosine_distance.cpu().numpy(), entropy.cpu().numpy()


def main() -> None:
    args = parse_args()
    configure_cache()
    import timm
    from timm.data import create_transform, resolve_model_data_config

    device = torch.device("cuda")
    dtype = torch.float16
    model = (
        timm.create_model(args.model, pretrained=True)
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
    if dense_error > 2e-3:
        raise RuntimeError(f"Dense equivalence failed: {dense_error}.")
    del verify_image, original_logits
    model.requires_grad_(False)

    transform = create_transform(
        **resolve_model_data_config(model),
        is_training=False,
    )
    samples = pd.read_csv(args.experiment_dir / "sample_records.csv")
    layers = len(wrappers)
    heads = wrappers[0].num_heads

    map_distance = np.zeros((len(samples), layers, heads, heads), dtype=np.float32)
    entropy = np.zeros((len(samples), layers, heads), dtype=np.float32)
    contribution_norm = np.zeros((len(samples), layers, heads), dtype=np.float32)
    contribution_l2 = np.zeros((len(samples), layers, heads, heads), dtype=np.float32)
    taylor = np.zeros((len(samples), layers, heads), dtype=np.float32)

    for sample_index, row in samples.iterrows():
        with Image.open(row["path"]) as loaded:
            image = transform(loaded.convert("RGB"))
        image = image.unsqueeze(0).to(device=device, dtype=dtype)

        # Capture one layer at a time so all preceding layers use the original
        # fused attention path.
        for layer, wrapper in enumerate(wrappers):
            clear_interventions(wrappers)
            wrapper.capture_attention_map = True
            wrapper.capture_contributions = True
            with torch.inference_mode():
                model(image)
            if wrapper.last_attention_map is None or wrapper.last_contributions is None:
                raise RuntimeError("Attention capture failed.")
            distances, head_entropy = attention_statistics(
                wrapper.last_attention_map
            )
            l2, _ = current_distance_matrices(wrapper.last_contributions)
            norms = (
                wrapper.last_contributions[0]
                .float()
                .flatten(1)
                .norm(dim=1)
                .cpu()
                .numpy()
            )
            map_distance[sample_index, layer] = distances
            entropy[sample_index, layer] = head_entropy
            contribution_l2[sample_index, layer] = l2
            contribution_norm[sample_index, layer] = norms

        # One backward pass yields gate Taylor scores for every layer.
        clear_interventions(wrappers)
        scales = [
            torch.ones(heads, device=device, dtype=dtype, requires_grad=True)
            for _ in wrappers
        ]
        for wrapper, scale in zip(wrappers, scales):
            wrapper.set_head_scales(scale)
        logits = model(image)
        target = torch.tensor(
            [int(row["imagenet_target"])],
            device=device,
            dtype=torch.long,
        )
        loss = F.cross_entropy(logits.float(), target)
        gradients = torch.autograd.grad(loss, scales)
        for layer, gradient in enumerate(gradients):
            taylor[sample_index, layer] = (
                (gradient * scales[layer])
                .abs()
                .float()
                .detach()
                .cpu()
                .numpy()
            )
        clear_interventions(wrappers)
        print(f"sample={sample_index + 1}/{len(samples)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        attention_map_distance=map_distance,
        attention_entropy=entropy,
        contribution_norm=contribution_norm,
        contribution_l2=contribution_l2,
        taylor=taylor,
        sample=samples["sample"].to_numpy(),
    )
    metadata = {
        "model": args.model,
        "samples": len(samples),
        "layers": layers,
        "heads": heads,
        "dense_equivalence_max_abs_error": dense_error,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"results={args.output.resolve()}")


if __name__ == "__main__":
    main()
