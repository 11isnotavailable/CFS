"""Check whether functional substitution structure is stable, not random.

For the same images, this script compares scalar-substitution matrices under
small semantic-preserving input transformations. It also computes a
between-image baseline at the same layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageEnhance, ImageOps
from scipy.stats import spearmanr
from torchvision.datasets import ImageFolder

from scripts.run_experiment0 import (
    build_substitution_masks,
    configure_cache,
    output_kl,
    parse_layers,
    run_variants,
    summarize_substitutions,
    verification_input,
    verify_dense_equivalence,
)
from src.head_intervention import clear_interventions, make_intervenable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", default="deit_small_patch16_224")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stability0"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--layers", default="0,1,6,11")
    parser.add_argument("--variant-batch-size", type=int, default=128)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-steps", type=int, default=31)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def transformed_images(image: Image.Image, transform) -> dict[str, torch.Tensor]:
    color = ImageEnhance.Brightness(image).enhance(1.05)
    color = ImageEnhance.Contrast(color).enhance(1.05)
    return {
        "original": transform(image),
        "horizontal_flip": transform(ImageOps.mirror(image)),
        "mild_color": transform(color),
    }


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def row_normalized_values(matrix: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    heads = matrix.shape[0]
    for source in range(heads):
        selector = np.arange(heads) != source
        denominator = np.median(matrix[source, selector])
        result[source, selector] /= max(float(denominator), 1e-12)
    return off_diagonal_values(result)


def top_indices(matrix: np.ndarray, count: int = 5) -> set[int]:
    values = off_diagonal_values(matrix)
    count = min(count, len(values))
    return set(np.argpartition(values, count - 1)[:count].tolist())


def matrix_comparison(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    left_values = off_diagonal_values(left)
    right_values = off_diagonal_values(right)
    heads = left.shape[0]
    left_best = np.argmin(np.where(np.eye(heads), np.inf, left), axis=1)
    right_best = np.argmin(np.where(np.eye(heads), np.inf, right), axis=1)
    left_global = int(np.argmin(np.where(np.eye(heads), np.inf, left)))
    right_global = int(np.argmin(np.where(np.eye(heads), np.inf, right)))
    left_top, right_top = top_indices(left), top_indices(right)
    return {
        "matrix_spearman": float(spearmanr(left_values, right_values).statistic),
        "row_normalized_spearman": float(
            spearmanr(
                row_normalized_values(left),
                row_normalized_values(right),
            ).statistic
        ),
        "best_representative_agreement": float(np.mean(left_best == right_best)),
        "global_best_pair_agreement": float(left_global == right_global),
        "top5_jaccard": float(
            len(left_top & right_top) / max(len(left_top | right_top), 1)
        ),
    }


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
    error = verify_dense_equivalence(model, wrappers, verify_image, original_logits)
    if error > 2e-3:
        raise RuntimeError(f"Dense equivalence failed: {error}.")
    del verify_image, original_logits

    layers = parse_layers(args.layers, len(wrappers))
    heads = wrappers[0].num_heads
    alphas = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    masks, mask_metadata = build_substitution_masks(heads, alphas)
    transform = create_transform(
        **resolve_model_data_config(model),
        is_training=False,
    )
    dataset = ImageFolder(args.data_root)
    sample_count = min(args.num_samples, len(dataset))
    sample_indices = np.linspace(
        0, len(dataset) - 1, sample_count, dtype=int
    ).tolist()

    records: dict[tuple[int, int, str], np.ndarray] = {}
    for sample, dataset_index in enumerate(sample_indices):
        path, _ = dataset.samples[dataset_index]
        with Image.open(path) as loaded:
            images = transformed_images(loaded.convert("RGB"), transform)
        for transform_name, tensor in images.items():
            image = tensor.unsqueeze(0).to(device=device, dtype=dtype)
            clear_interventions(wrappers)
            with torch.inference_mode():
                reference_logits = model(image)
            for layer in layers:
                _, errors = run_variants(
                    model,
                    wrappers[layer],
                    image,
                    masks,
                    reference_logits,
                    args.variant_batch_size,
                    args.temperature,
                )
                matrix, _ = summarize_substitutions(heads, mask_metadata, errors)
                records[(sample, layer, transform_name)] = matrix
            print(f"sample={sample + 1}/{sample_count} transform={transform_name}")

    rows = []
    for sample in range(sample_count):
        for layer in layers:
            original = records[(sample, layer, "original")]
            for transform_name in ("horizontal_flip", "mild_color"):
                comparison = matrix_comparison(
                    original,
                    records[(sample, layer, transform_name)],
                )
                comparison.update(
                    {
                        "comparison": transform_name,
                        "sample": sample,
                        "layer": layer,
                    }
                )
                rows.append(comparison)

    # Compare each original image with the next image at the same layer.
    for sample in range(sample_count):
        other = (sample + 1) % sample_count
        for layer in layers:
            comparison = matrix_comparison(
                records[(sample, layer, "original")],
                records[(other, layer, "original")],
            )
            comparison.update(
                {
                    "comparison": "between_images",
                    "sample": sample,
                    "layer": layer,
                }
            )
            rows.append(comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "stability_records.csv", index=False)
    np.savez_compressed(
        args.output_dir / "stability_matrices.npz",
        matrices=np.stack(list(records.values())),
        keys=np.array([str(key) for key in records]),
    )

    metric_columns = [
        "matrix_spearman",
        "row_normalized_spearman",
        "best_representative_agreement",
        "top5_jaccard",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for metric, axis in zip(metric_columns, axes.flatten()):
        sns.boxplot(data=frame, x="comparison", y=metric, ax=axis)
        axis.tick_params(axis="x", rotation=18)
        axis.set_title(metric)
    fig.tight_layout()
    fig.savefig(args.output_dir / "stability_summary.png", dpi=180)
    plt.close(fig)

    summary = (
        frame.groupby("comparison")[metric_columns]
        .agg(["mean", "median"])
        .round(4)
    )
    (args.output_dir / "stability_summary.txt").write_text(
        summary.to_string(), encoding="utf-8"
    )
    metadata = {
        "model": args.model,
        "samples": sample_count,
        "sample_indices": sample_indices,
        "layers": layers,
        "transforms": ["original", "horizontal_flip", "mild_color"],
        "dense_equivalence_max_abs_error": error,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary)
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
