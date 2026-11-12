"""Run the no-training head-intervention falsification experiment.

The script performs, for each selected image and transformer layer:
  * all 2^H deletion masks;
  * directed scalar substitution over an alpha grid;
  * post-W_O current-distance measurements;
  * exact subset, magnitude, greedy, and random budget summaries.

It writes raw CSV/NPZ records plus the three initial figures described in the
research plan.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


sys.path.insert(0, str(PROJECT_ROOT))

from src.head_intervention import clear_interventions, make_intervenable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnlabeledImages(Dataset):
    def __init__(self, root: Path, transform) -> None:
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found below {root}.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, -1, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--model", default="deit_small_patch16_224", help="timm model name"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment0"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--layers",
        default="all",
        help='Comma-separated zero-based layers, e.g. "3,6,9", or "all".',
    )
    parser.add_argument("--variant-batch-size", type=int, default=128)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-steps", type=int, default=31)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify dense equivalence on random input.",
    )
    return parser.parse_args()


def configure_cache() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))


def parse_layers(value: str, depth: int) -> list[int]:
    if value.lower() == "all":
        return list(range(depth))
    layers = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not layers or any(layer < 0 or layer >= depth for layer in layers):
        raise ValueError(f"Layers must lie in [0,{depth - 1}], received {layers}.")
    return layers


def all_binary_masks(heads: int) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product([0.0, 1.0], repeat=heads)),
        dtype=torch.float32,
    )


def output_kl(
    reference_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    reference = reference_logits.float() / temperature
    variants = variant_logits.float() / temperature
    log_p = F.log_softmax(reference, dim=-1)
    p = log_p.exp()
    log_q = F.log_softmax(variants, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1) * (temperature**2)
    # FP16 and batch-size-dependent kernels can yield tiny negative estimates.
    return kl.clamp_min(0.0)


def run_variants(
    model: torch.nn.Module,
    wrapper,
    image: torch.Tensor,
    masks: torch.Tensor,
    reference_logits: torch.Tensor,
    chunk_size: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_parts = []
    error_parts = []
    for start in range(0, len(masks), chunk_size):
        stop = min(start + chunk_size, len(masks))
        chunk = masks[start:stop].to(image.device)
        batch = image.expand(len(chunk), -1, -1, -1)
        wrapper.set_head_scales(chunk)
        with torch.inference_mode():
            logits = model(batch)
        errors = output_kl(reference_logits, logits, temperature)
        logits_parts.append(logits.float().cpu())
        error_parts.append(errors.cpu())
    wrapper.set_head_scales(None)
    return torch.cat(logits_parts), torch.cat(error_parts)


def current_distance_matrices(contributions: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    # contributions: [1, H, N, C]
    flat = contributions[0].float().flatten(1)
    cosine = 1.0 - F.normalize(flat, dim=1) @ F.normalize(flat, dim=1).T
    difference = flat[:, None, :] - flat[None, :, :]
    denominator = flat.norm(dim=1)[:, None] + flat.norm(dim=1)[None, :]
    normalized_l2 = difference.norm(dim=-1) / denominator.clamp_min(1e-12)
    return normalized_l2.cpu().numpy(), cosine.cpu().numpy()


def contribution_norms(contributions: torch.Tensor) -> torch.Tensor:
    return contributions[0].float().flatten(1).norm(dim=1).cpu()


def mask_index_lookup(masks: torch.Tensor) -> dict[tuple[int, ...], int]:
    return {
        tuple(int(value) for value in mask.tolist()): index
        for index, mask in enumerate(masks)
    }


def budget_statistics(
    masks: torch.Tensor,
    errors: torch.Tensor,
    norms: torch.Tensor,
) -> list[dict[str, float]]:
    heads = masks.shape[1]
    lookup = mask_index_lookup(masks)
    rows: list[dict[str, float]] = []

    # Greedy path uses the actual downstream error after every deletion.
    greedy_masks: dict[int, tuple[int, ...]] = {heads: tuple([1] * heads)}
    current = [1] * heads
    for kept in range(heads - 1, -1, -1):
        candidates = []
        for head in range(heads):
            if current[head]:
                candidate = current.copy()
                candidate[head] = 0
                candidates.append((float(errors[lookup[tuple(candidate)]]), candidate))
        _, current = min(candidates, key=lambda item: item[0])
        greedy_masks[kept] = tuple(current)

    descending_norm = torch.argsort(norms, descending=True).tolist()
    for kept in range(heads + 1):
        selector = masks.sum(dim=1) == kept
        selected_errors = errors[selector]
        magnitude_mask = [0] * heads
        for head in descending_norm[:kept]:
            magnitude_mask[head] = 1
        rows.append(
            {
                "kept_heads": kept,
                "exact_best_kl": float(selected_errors.min()),
                "random_mean_kl": float(selected_errors.mean()),
                "magnitude_kl": float(errors[lookup[tuple(magnitude_mask)]]),
                "greedy_kl": float(errors[lookup[greedy_masks[kept]]]),
            }
        )
    return rows


def build_substitution_masks(
    heads: int, alphas: torch.Tensor
) -> tuple[torch.Tensor, list[tuple[int, int, float]]]:
    masks = []
    metadata = []
    for source in range(heads):
        for representative in range(heads):
            if source == representative:
                continue
            for alpha in alphas.tolist():
                mask = torch.ones(heads)
                mask[source] = 0.0
                mask[representative] = alpha
                masks.append(mask)
                metadata.append((source, representative, float(alpha)))
    return torch.stack(masks), metadata


def summarize_substitutions(
    heads: int,
    metadata: list[tuple[int, int, float]],
    errors: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((heads, heads), np.nan, dtype=np.float64)
    alpha_matrix = np.full((heads, heads), np.nan, dtype=np.float64)
    grouped: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for (source, representative, alpha), error in zip(metadata, errors.tolist()):
        grouped.setdefault((source, representative), []).append((error, alpha))
    for (source, representative), values in grouped.items():
        best_error, best_alpha = min(values, key=lambda item: item[0])
        matrix[source, representative] = best_error
        alpha_matrix[source, representative] = best_alpha
    np.fill_diagonal(matrix, 0.0)
    np.fill_diagonal(alpha_matrix, 1.0)
    return matrix, alpha_matrix


def substitution_error_cube(
    heads: int,
    alpha_count: int,
    metadata: list[tuple[int, int, float]],
    errors: torch.Tensor,
) -> np.ndarray:
    """Return errors shaped [source, representative, alpha_index]."""
    cube = np.full((heads, heads, alpha_count), np.nan, dtype=np.float32)
    counters: dict[tuple[int, int], int] = {}
    for (source, representative, _), error in zip(metadata, errors.tolist()):
        key = (source, representative)
        alpha_index = counters.get(key, 0)
        cube[source, representative, alpha_index] = float(error)
        counters[key] = alpha_index + 1
    return cube


def verification_input(device: str, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(0)
    return torch.randn(
        2, 3, 224, 224, generator=generator, device=device, dtype=dtype
    )


def verify_dense_equivalence(
    model,
    wrappers,
    image: torch.Tensor,
    original_logits: torch.Tensor,
) -> float:
    clear_interventions(wrappers)
    with torch.inference_mode():
        wrapped_dense = model(image)
    for wrapper in wrappers:
        wrapper.set_head_scales(torch.ones(wrapper.num_heads, device=image.device))
    with torch.inference_mode():
        wrapped_ones = model(image)
    clear_interventions(wrappers)
    dense_error = (original_logits.float() - wrapped_dense.float()).abs().max()
    ones_error = (original_logits.float() - wrapped_ones.float()).abs().max()
    return float(torch.maximum(dense_error, ones_error))


def make_dataset(root: Path, transform):
    if any(path.is_dir() for path in root.iterdir()):
        try:
            dataset = ImageFolder(root, transform=transform)

            class WithPath(Dataset):
                def __len__(self):
                    return len(dataset)

                def __getitem__(self, index):
                    image, label = dataset[index]
                    return image, label, dataset.samples[index][0]

            return WithPath()
        except FileNotFoundError:
            pass
    return UnlabeledImages(root, transform)


def save_figures(
    output_dir: Path,
    example: dict[str, object],
    pair_rows: list[dict[str, float]],
    budget_rows: list[dict[str, float]],
    layer_summary_rows: list[dict[str, float]],
) -> None:
    matrix = example["substitution_matrix"]
    single_drop = np.asarray(example["single_drop_vector"])
    normalized = matrix / np.maximum(single_drop[:, None], 1e-10)
    np.fill_diagonal(normalized, np.nan)
    layer = example["layer"]
    sample = example["sample"]
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    sns.heatmap(matrix, annot=True, fmt=".2g", cmap="mako_r", ax=axes[0])
    axes[0].set_title("Substitution KL")
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=1.0,
        ax=axes[1],
    )
    axes[1].set_title("KL / direct-drop KL")
    for axis in axes:
        axis.set_xlabel("representative head j")
        axis.set_ylabel("replaced head i")
    figure.suptitle(
        f"Functional substitution matrix · sample {sample} · layer {layer}"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "figure1_substitution_matrix.png", dpi=180)
    plt.close(figure)

    pairs = pd.DataFrame(pair_rows)
    plt.figure(figsize=(6.4, 5.2))
    sns.scatterplot(
        data=pairs,
        x="current_l2",
        y="substitution_kl",
        hue="single_drop_kl",
        palette="viridis",
        s=24,
        linewidth=0,
    )
    plt.yscale("log")
    plt.title("Current representation distance vs functional substitution")
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_current_vs_functional.png", dpi=180)
    plt.close()

    budgets = pd.DataFrame(budget_rows)
    aggregate = budgets.groupby("kept_heads", as_index=False).mean(numeric_only=True)
    long = aggregate.melt(
        id_vars="kept_heads",
        value_vars=["exact_best_kl", "greedy_kl", "magnitude_kl", "random_mean_kl"],
        var_name="method",
        value_name="kl",
    )
    plt.figure(figsize=(6.4, 5.2))
    sns.lineplot(data=long, x="kept_heads", y="kl", hue="method", marker="o")
    positive_floor = 1e-10
    long["kl"] = long["kl"].clip(lower=positive_floor)
    plt.yscale("log")
    summaries = pd.DataFrame(layer_summary_rows)
    marker_x = int(summaries["heads"].iloc[0]) - 1
    plt.scatter(
        [marker_x],
        [max(float(summaries["best_functional_kl"].mean()), positive_floor)],
        marker="*",
        s=170,
        label="functional pair substitution",
        zorder=5,
    )
    plt.scatter(
        [marker_x],
        [max(float(summaries["similarity_pair_kl"].mean()), positive_floor)],
        marker="X",
        s=90,
        label="current-similarity pair",
        zorder=5,
    )
    plt.legend()
    plt.title("Head-budget oracle curve")
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_budget_curve.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    configure_cache()
    import timm
    from timm.data import create_transform, resolve_model_data_config

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    model = (
        timm.create_model(args.model, pretrained=True)
        .eval()
        .to(device=device, dtype=dtype)
    )
    verify_image = verification_input(str(device), dtype)
    with torch.inference_mode():
        original_logits = model(verify_image)
    wrappers = make_intervenable(model)
    layers = parse_layers(args.layers, len(wrappers))

    max_error = verify_dense_equivalence(
        model, wrappers, verify_image, original_logits
    )
    del verify_image, original_logits
    print(f"dense_equivalence_max_abs_error={max_error:.3e}")
    tolerance = 2e-3 if dtype == torch.float16 else 2e-5
    if max_error > tolerance:
        raise RuntimeError(
            f"Intervention wrapper failed dense equivalence: {max_error} > {tolerance}."
        )
    if args.verify_only:
        return
    if args.data_root is None:
        raise ValueError("--data-root is required unless --verify-only is set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    dataset = make_dataset(args.data_root, transform)
    sample_count = min(args.num_samples, len(dataset))
    sample_indices = np.linspace(
        0, len(dataset) - 1, num=sample_count, dtype=int
    ).tolist()

    heads = wrappers[0].num_heads
    if any(wrapper.num_heads != heads for wrapper in wrappers):
        raise ValueError("Experiment 0 expects the same head count in every layer.")
    subset_masks = all_binary_masks(heads)
    alphas = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    substitution_masks, substitution_metadata = build_substitution_masks(heads, alphas)
    subset_lookup = mask_index_lookup(subset_masks)

    pair_rows: list[dict[str, float]] = []
    budget_rows: list[dict[str, float]] = []
    layer_summary_rows: list[dict[str, float]] = []
    sample_rows: list[dict[str, object]] = []
    matrix_records = []
    subset_error_records = []
    substitution_curve_records = []
    example = None
    example_score = -math.inf

    for sample_idx, dataset_idx in enumerate(sample_indices):
        image, label, path = dataset[dataset_idx]
        image = image.unsqueeze(0).to(device=device, dtype=dtype)
        clear_interventions(wrappers)
        with torch.inference_mode():
            reference_logits = model(image)
        reference_class = int(reference_logits.argmax(dim=-1).item())
        confidence = float(reference_logits.float().softmax(dim=-1).max().item())
        synset = Path(path).parent.name
        target_index = -1
        try:
            from timm.data import ImageNetInfo

            imagenet_labels = ImageNetInfo().label_names()
            if synset in imagenet_labels:
                target_index = imagenet_labels.index(synset)
        except (FileNotFoundError, ValueError):
            pass
        sample_rows.append(
            {
                "sample": sample_idx,
                "dataset_index": dataset_idx,
                "path": str(path),
                "dataset_label": int(label),
                "imagenet_target": target_index,
                "prediction": reference_class,
                "correct": (
                    int(reference_class == target_index) if target_index >= 0 else -1
                ),
                "confidence": confidence,
            }
        )

        for layer in layers:
            wrapper = wrappers[layer]
            wrapper.capture_contributions = True
            with torch.inference_mode():
                _ = model(image)
            contributions = wrapper.last_contributions
            wrapper.capture_contributions = False
            if contributions is None:
                raise RuntimeError("Failed to capture head contributions.")
            l2_matrix, cosine_matrix = current_distance_matrices(contributions)
            norms = contribution_norms(contributions)
            wrapper.last_contributions = None

            _, subset_errors = run_variants(
                model,
                wrapper,
                image,
                subset_masks,
                reference_logits,
                args.variant_batch_size,
                args.temperature,
            )
            subset_error_records.append(subset_errors.numpy())
            _, substitution_errors = run_variants(
                model,
                wrapper,
                image,
                substitution_masks,
                reference_logits,
                args.variant_batch_size,
                args.temperature,
            )
            substitution_matrix, alpha_matrix = summarize_substitutions(
                heads, substitution_metadata, substitution_errors
            )
            substitution_curve_records.append(
                substitution_error_cube(
                    heads,
                    len(alphas),
                    substitution_metadata,
                    substitution_errors,
                )
            )

            matrix_records.append(
                {
                    "sample": sample_idx,
                    "dataset_index": dataset_idx,
                    "layer": layer,
                    "path": str(path),
                    "label": int(label),
                    "prediction": reference_class,
                    "substitution_matrix": substitution_matrix,
                    "alpha_matrix": alpha_matrix,
                    "current_l2_matrix": l2_matrix,
                    "current_cosine_matrix": cosine_matrix,
                }
            )
            single_errors = []
            for source in range(heads):
                single_mask = [1] * heads
                single_mask[source] = 0
                single_error = float(subset_errors[subset_lookup[tuple(single_mask)]])
                single_errors.append(single_error)
                for representative in range(heads):
                    if source == representative:
                        continue
                    substitution_error = float(
                        substitution_matrix[source, representative]
                    )
                    pair_rows.append(
                        {
                            "sample": sample_idx,
                            "layer": layer,
                            "source": source,
                            "representative": representative,
                            "current_l2": float(l2_matrix[source, representative]),
                            "current_cosine": float(
                                cosine_matrix[source, representative]
                            ),
                            "substitution_kl": substitution_error,
                            "best_alpha": float(alpha_matrix[source, representative]),
                            "single_drop_kl": single_error,
                            "substitution_gain": single_error - substitution_error,
                            "relative_gain": (
                                (single_error - substitution_error) / single_error
                                if single_error > 1e-10
                                else 0.0
                            ),
                        }
                    )

            off_diagonal_mask = ~np.eye(heads, dtype=bool)
            best_functional_kl = float(
                np.nanmin(substitution_matrix[off_diagonal_mask])
            )
            best_functional_flat = np.nanargmin(
                np.where(off_diagonal_mask, substitution_matrix, np.nan)
            )
            best_source, best_representative = np.unravel_index(
                best_functional_flat, substitution_matrix.shape
            )
            upper_triangle = np.triu(np.ones((heads, heads), dtype=bool), k=1)
            closest_flat = np.argmin(np.where(upper_triangle, l2_matrix, np.inf))
            closest_i, closest_j = np.unravel_index(
                closest_flat, l2_matrix.shape
            )
            similarity_pair_kl = float(
                min(
                    substitution_matrix[closest_i, closest_j],
                    substitution_matrix[closest_j, closest_i],
                )
            )
            gain_matrix = np.asarray(single_errors)[:, None] - substitution_matrix
            best_gain = float(np.nanmax(gain_matrix[off_diagonal_mask]))
            matrix_records[-1]["single_drop_vector"] = np.asarray(single_errors)
            layer_summary_rows.append(
                {
                    "sample": sample_idx,
                    "layer": layer,
                    "heads": heads,
                    "best_functional_kl": best_functional_kl,
                    "best_functional_source": best_source,
                    "best_functional_representative": best_representative,
                    "closest_current_i": closest_i,
                    "closest_current_j": closest_j,
                    "similarity_pair_kl": similarity_pair_kl,
                    "best_single_drop_kl": min(single_errors),
                    "mean_single_drop_kl": float(np.mean(single_errors)),
                }
            )
            if example is None or best_gain > example_score:
                example = matrix_records[-1]
                example_score = best_gain

            for row in budget_statistics(subset_masks, subset_errors, norms):
                row.update({"sample": sample_idx, "layer": layer})
                budget_rows.append(row)

            off_diagonal = substitution_matrix[~np.eye(heads, dtype=bool)]
            print(
                f"sample={sample_idx + 1}/{args.num_samples} layer={layer} "
                f"best_sub_kl={np.nanmin(off_diagonal):.3e}"
            )

    if example is None:
        raise RuntimeError("No samples were processed.")

    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pair_records.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(args.output_dir / "budget_records.csv", index=False)
    pd.DataFrame(layer_summary_rows).to_csv(
        args.output_dir / "layer_summary.csv", index=False
    )
    pd.DataFrame(sample_rows).to_csv(
        args.output_dir / "sample_records.csv", index=False
    )
    np.savez_compressed(
        args.output_dir / "matrix_records.npz",
        substitution=np.stack([record["substitution_matrix"] for record in matrix_records]),
        alpha=np.stack([record["alpha_matrix"] for record in matrix_records]),
        current_l2=np.stack([record["current_l2_matrix"] for record in matrix_records]),
        current_cosine=np.stack(
            [record["current_cosine_matrix"] for record in matrix_records]
        ),
        single_drop=np.stack(
            [record["single_drop_vector"] for record in matrix_records]
        ),
        subset_errors=np.stack(subset_error_records),
        subset_masks=subset_masks.numpy(),
        substitution_curves=np.stack(substitution_curve_records),
        alpha_grid=alphas.numpy(),
        sample=np.array([record["sample"] for record in matrix_records]),
        layer=np.array([record["layer"] for record in matrix_records]),
    )
    save_figures(
        args.output_dir, example, pair_rows, budget_rows, layer_summary_rows
    )
    metadata = {
        "model": args.model,
        "num_samples": sample_count,
        "sample_indices": sample_indices,
        "layers": layers,
        "heads": heads,
        "alpha_grid": alphas.tolist(),
        "temperature": args.temperature,
        "dtype": str(dtype),
        "dense_equivalence_max_abs_error": max_error,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
