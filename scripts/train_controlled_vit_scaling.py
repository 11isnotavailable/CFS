"""Train a controlled ViT configuration from scratch on a frozen data split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from timm.data import Mixup, create_transform, resolve_model_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer
from timm.utils import ModelEmaV2
from torchvision.datasets import ImageFolder

from src.head_intervention import make_intervenable

DEVELOPMENTAL_POLICIES = {
    "functional_basis",
    "safe_adversarial",
    "soft_all",
    "coverage_cfs",
    "sensitivity_cfs",
    "head_exchange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--embed-dim", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help="Enable a same-trajectory EMA checkpoint when greater than zero.",
    )
    parser.add_argument(
        "--sam-rho", type=float, default=0.0,
        help="Enable non-adaptive SAM with the given neighborhood radius.",
    )
    parser.add_argument(
        "--milestone-epochs", default="",
        help="Comma-separated epochs at which online and EMA states are saved.",
    )
    parser.add_argument(
        "--save-every-epoch", action="store_true",
        help="Save a model-only checkpoint after every completed epoch.",
    )
    parser.add_argument(
        "--save-final-training-state", action="store_true",
        help="Save final optimizer, scheduler, and GradScaler state for continuation.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=20260805)
    parser.add_argument("--tune-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Canonical controlled-ViT checkpoint used to initialize post-training.",
    )
    parser.add_argument(
        "--aux-policy",
        choices=(
            "none", "head_drop", "random_substitution", "self_cfs",
            *sorted(DEVELOPMENTAL_POLICIES),
        ),
        default="none",
    )
    parser.add_argument("--aux-samples", type=int, default=16)
    parser.add_argument("--aux-task-weight", type=float, default=0.25)
    parser.add_argument("--aux-kl-weight", type=float, default=0.25)
    parser.add_argument("--substitution-alpha", type=float, default=2.0)
    parser.add_argument("--aux-layers", default="3,7,11")
    parser.add_argument("--aux-structure-weight", type=float, default=0.05)
    parser.add_argument("--cfs-safe-ratio", type=float, default=0.75)
    parser.add_argument("--cfs-min-drop-kl", type=float, default=1e-5)
    parser.add_argument("--cfs-temperature-start", type=float, default=1.0)
    parser.add_argument("--cfs-temperature-end", type=float, default=0.1)
    parser.add_argument("--cfs-start-epoch", type=int, default=5)
    parser.add_argument("--cfs-ramp-end-epoch", type=int, default=20)
    parser.add_argument("--cfs-hold-end-epoch", type=int, default=55)
    parser.add_argument("--cfs-final-strength", type=float, default=0.1)
    parser.add_argument("--functional-basis-epsilon", type=float, default=1e-3)
    parser.add_argument("--functional-energy-floor-ratio", type=float, default=0.5)
    parser.add_argument("--functional-energy-floor-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(
    targets: list[int], tune_fraction: float, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.asarray(targets)
    train, tune, test = [], [], []
    for label in np.unique(values):
        indices = np.flatnonzero(values == label)
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_fraction)))
        tune_count = max(1, int(round(len(indices) * tune_fraction)))
        test.extend(indices[:test_count])
        tune.extend(indices[test_count : test_count + tune_count])
        train.extend(indices[test_count + tune_count :])
    for result in (train, tune, test):
        rng.shuffle(result)
    return tuple(np.asarray(value, dtype=np.int64) for value in (train, tune, test))


def load_or_create_split(args: argparse.Namespace, dataset: ImageFolder):
    target_hash = hashlib.sha256(np.asarray(dataset.targets, dtype=np.int16).tobytes()).hexdigest()
    if args.split_manifest.exists():
        archive = np.load(args.split_manifest)
        if str(archive["target_hash"]) != target_hash:
            raise ValueError("Frozen split manifest does not match this dataset")
        return archive["train"], archive["tune"], archive["test"], target_hash
    train, tune, test = split_indices(
        dataset.targets, args.tune_fraction, args.test_fraction, args.split_seed
    )
    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.split_manifest,
        train=train,
        tune=tune,
        test=test,
        target_hash=np.asarray(target_hash),
        split_seed=np.asarray(args.split_seed),
    )
    return train, tune, test, target_hash


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        loss_sum += float(loss) * len(targets)
        correct += int((logits.argmax(dim=-1) == targets).sum())
        count += len(targets)
    return loss_sum / count, correct / count


def canonical_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a checkpoint loadable as an ordinary, unwrapped timm ViT."""
    return {
        key.replace(".attn.attention.", ".attn."): value
        for key, value in model.state_dict().items()
    }


def load_canonical_state_dict(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Load an ordinary ViT state into either a dense or wrapped model."""
    wrapped = any(".attn.attention." in key for key in model.state_dict())
    if wrapped:
        state = {
            key.replace(".attn.", ".attn.attention."): value
            for key, value in state.items()
        }
    model.load_state_dict(state, strict=True)


def auxiliary_intervention_loss(
    model: torch.nn.Module,
    wrapper,
    images: torch.Tensor,
    mixed_targets: torch.Tensor,
    criterion,
    args: argparse.Namespace,
    sampling_rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a compute-matched stochastic intervention objective.

    All non-baseline policies evaluate two variants. ``self_cfs`` selects the
    variant with lower downstream KL, whereas ``random_substitution`` selects
    one uniformly. ``head_drop`` repeats the same deletion variant so its
    forward budget matches the substitution policies.
    """
    count = min(args.aux_samples, len(images))
    local_images = images[:count]
    local_targets = mixed_targets[:count]
    heads = wrapper.num_heads
    rows = torch.arange(count, device=images.device)
    source = torch.as_tensor(
        sampling_rng.integers(0, heads, size=count),
        device=images.device,
        dtype=torch.long,
    )
    candidate_values = []
    for sample, source_value in enumerate(source.cpu().tolist()):
        available = [head for head in range(heads) if head != source_value]
        if args.aux_policy == "head_drop":
            candidate_values.append((available[0], available[0]))
        else:
            chosen = sampling_rng.choice(available, size=2, replace=False)
            candidate_values.append((int(chosen[0]), int(chosen[1])))
    candidates = torch.as_tensor(
        candidate_values, device=images.device, dtype=torch.long
    )

    # Recompute a clean reference on the same subbatch. Resetting CUDA RNG
    # before every variant makes DropPath masks identical across comparisons.
    cuda_state = torch.cuda.get_rng_state(images.device)
    wrapper.set_head_scales(None)
    with torch.no_grad(), torch.autocast(
        device_type=images.device.type, dtype=torch.float16
    ):
        reference = model(local_images).detach()

    variant_logits = []
    for candidate_column in range(2):
        torch.cuda.set_rng_state(cuda_state, images.device)
        scales = torch.ones(
            count, heads, device=images.device, dtype=images.dtype
        )
        scales[rows, source] = 0.0
        if args.aux_policy != "head_drop":
            scales[rows, candidates[:, candidate_column]] = args.substitution_alpha
        wrapper.set_head_scales(scales)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16):
            variant_logits.append(model(local_images))
    wrapper.set_head_scales(None)

    stacked = torch.stack(variant_logits, dim=1)
    reference_probability = reference.float().softmax(dim=-1)[:, None, :]
    candidate_kl = F.kl_div(
        stacked.float().log_softmax(dim=-1),
        reference_probability.expand_as(stacked),
        reduction="none",
    ).sum(dim=-1)
    if args.aux_policy == "self_cfs":
        selection = candidate_kl.detach().argmin(dim=1)
    else:
        selection = torch.as_tensor(
            sampling_rng.integers(0, 2, size=count),
            device=images.device,
            dtype=torch.long,
        )
    selected_logits = stacked[rows, selection]
    selected_kl = candidate_kl[rows, selection].mean()
    selected_task = criterion(selected_logits, local_targets)
    return selected_task, selected_kl


def developmental_strength(epoch: int, args: argparse.Namespace) -> float:
    """Curriculum that shapes branch organization, then permits consolidation."""
    if epoch <= args.cfs_start_epoch:
        return 0.0
    if epoch < args.cfs_ramp_end_epoch:
        width = max(1, args.cfs_ramp_end_epoch - args.cfs_start_epoch)
        return (epoch - args.cfs_start_epoch) / width
    if epoch <= args.cfs_hold_end_epoch:
        return 1.0
    width = max(1, args.epochs - args.cfs_hold_end_epoch)
    progress = min(1.0, (epoch - args.cfs_hold_end_epoch) / width)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.cfs_final_strength + (1.0 - args.cfs_final_strength) * cosine


def developmental_temperature(epoch: int, args: argparse.Namespace) -> float:
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, args.epochs - 1)))
    return (
        args.cfs_temperature_start * (1.0 - progress)
        + args.cfs_temperature_end * progress
    )


def soft_target_ce_per_sample(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return -(targets.float() * logits.float().log_softmax(dim=-1)).sum(dim=-1)


def kl_to_reference(
    logits: torch.Tensor, reference_probability: torch.Tensor
) -> torch.Tensor:
    return F.kl_div(
        logits.float().log_softmax(dim=-1),
        reference_probability,
        reduction="none",
    ).sum(dim=-1)


def candidate_table(source: torch.Tensor, heads: int) -> torch.Tensor:
    all_heads = torch.arange(heads, device=source.device).expand(len(source), heads)
    return all_heads[all_heads != source[:, None]].reshape(len(source), heads - 1)


def intervention_scales(
    source: torch.Tensor,
    candidate: torch.Tensor | None,
    heads: int,
    dtype: torch.dtype,
    alpha: float,
) -> torch.Tensor:
    rows = torch.arange(len(source), device=source.device)
    scales = torch.ones(len(source), heads, device=source.device, dtype=dtype)
    scales[rows, source] = 0.0
    if candidate is not None:
        scales[rows, candidate] = alpha
    return scales


def sensitivity_developmental_loss(
    model: torch.nn.Module,
    wrapper,
    images: torch.Tensor,
    mixed_targets: torch.Tensor,
    criterion,
    args: argparse.Namespace,
    sampling_rng: np.random.Generator,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Task-gradient-aligned substitution without second-order gradients."""
    count = min(args.aux_samples, len(images))
    local_images, local_targets = images[:count], mixed_targets[:count]
    heads = wrapper.num_heads
    source = torch.as_tensor(
        sampling_rng.integers(0, heads, size=count),
        device=images.device, dtype=torch.long,
    )
    candidates = candidate_table(source, heads)
    rows = torch.arange(count, device=images.device)
    saved_cuda_state = torch.cuda.get_rng_state(images.device)

    wrapper.set_head_scales(None)
    wrapper.capture_heads_with_grad = True
    with torch.autocast(device_type=images.device.type, dtype=torch.float16):
        reference = model(local_images)
        reference_task = criterion(reference, local_targets)
    captured = wrapper.last_heads_with_grad
    if captured is None:
        raise RuntimeError("Failed to capture differentiable attention heads")
    head_gradient = torch.autograd.grad(
        reference_task, captured, retain_graph=True, create_graph=False
    )[0].detach()
    wrapper.capture_heads_with_grad = False
    wrapper.last_heads_with_grad = None

    signed_effect = (head_gradient.float() * captured.float()).sum(dim=(1, 3))
    source_effect = signed_effect[rows, source]
    candidate_effect = signed_effect.gather(1, candidates)
    energy = (
        -source_effect[:, None]
        + (args.substitution_alpha - 1.0) * candidate_effect
    ).square()
    normalized = energy / energy.detach().mean(dim=1, keepdim=True).clamp_min(1e-8)
    weights = torch.softmax(-normalized.detach() / max(temperature, 1e-3), dim=1)
    structural = (weights * normalized).sum(dim=1).mean()
    selection = normalized.detach().argmin(dim=1)
    selected_candidate = candidates[rows, selection]

    torch.cuda.set_rng_state(saved_cuda_state, images.device)
    wrapper.set_head_scales(intervention_scales(
        source, selected_candidate, heads, images.dtype, args.substitution_alpha
    ))
    with torch.autocast(device_type=images.device.type, dtype=torch.float16):
        selected_logits = model(local_images)
    wrapper.set_head_scales(None)
    torch.cuda.set_rng_state(saved_cuda_state, images.device)
    reference_probability = reference.detach().float().softmax(dim=-1)
    task = criterion(selected_logits, local_targets)
    kl = kl_to_reference(selected_logits, reference_probability).mean()
    return task, kl, structural


def developmental_intervention_loss(
    model: torch.nn.Module,
    wrapper,
    images: torch.Tensor,
    mixed_targets: torch.Tensor,
    criterion,
    args: argparse.Namespace,
    sampling_rng: np.random.Generator,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Develop CFS topology using all alternative heads at a sampled layer."""
    if args.aux_policy == "functional_basis":
        count = min(args.aux_samples, len(images))
        local_images = images[:count]
        heads = wrapper.num_heads
        saved_cuda_state = torch.cuda.get_rng_state(images.device)
        wrapper.set_head_scales(None)
        torch.cuda.set_rng_state(saved_cuda_state, images.device)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16):
            reference = model(local_images)

        drop_logits = []
        for head in range(heads):
            torch.cuda.set_rng_state(saved_cuda_state, images.device)
            source = torch.full(
                (count,), head, device=images.device, dtype=torch.long
            )
            wrapper.set_head_scales(intervention_scales(
                source, None, heads, images.dtype, args.substitution_alpha
            ))
            with torch.autocast(device_type=images.device.type, dtype=torch.float16):
                drop_logits.append(model(local_images))
        wrapper.set_head_scales(None)
        torch.cuda.set_rng_state(saved_cuda_state, images.device)

        # U[h] is the downstream functional signature of deleting head h.
        # Remove per-example uniform logit shifts, which do not change the
        # predictive distribution, before measuring functional dimension.
        dropped = torch.stack(drop_logits, dim=1)
        signatures = reference[:, None, :].float() - dropped.float()
        signatures = signatures - signatures.mean(dim=-1, keepdim=True)
        functional = signatures.permute(1, 0, 2).reshape(heads, -1).float()
        # The enclosing training forward runs under autocast. Disable it for
        # the small HxH spectral computation so eig/logdet stay in FP32.
        with torch.autocast(device_type=images.device.type, enabled=False):
            gram = functional @ functional.T / functional.shape[1]
        diagonal = torch.diagonal(gram)
        mean_energy = diagonal.detach().mean()
        anchor = getattr(wrapper, "functional_basis_energy_anchor", None)
        if anchor is None:
            anchor = diagonal.detach().clamp_min(1e-8).clone()
            wrapper.functional_basis_energy_anchor = anchor
        eigenvalues = torch.linalg.eigvalsh(gram.detach()).clamp_min(1e-12)
        probability = eigenvalues / eigenvalues.sum()
        wrapper.functional_basis_last_energy = float(mean_energy)
        wrapper.functional_basis_last_effective_rank = float(torch.exp(
            -(probability * probability.log()).sum()
        ))
        inverse_std = torch.rsqrt(diagonal.clamp_min(1e-12))
        correlation = inverse_std[:, None] * gram * inverse_std[None, :]
        identity = torch.eye(heads, device=gram.device, dtype=gram.dtype)
        stabilized = correlation + args.functional_basis_epsilon * identity
        sign, logdet = torch.linalg.slogdet(stabilized)
        if not bool(sign.detach() > 0):
            raise RuntimeError("Functional Gram matrix lost positive definiteness")
        # Correlation normalization makes the rank term invariant to each
        # head's contribution norm. A separate one-sided floor prevents the
        # degenerate solution observed in sensitivity training, without
        # rewarding unbounded norm growth.
        rank_loss = -logdet / heads + math.log1p(args.functional_basis_epsilon)
        energy_ratio = diagonal / anchor
        energy_floor = F.relu(
            args.functional_energy_floor_ratio - energy_ratio
        ).square().mean()
        structural = rank_loss + args.functional_energy_floor_weight * energy_floor
        zero = structural.new_zeros(())
        return zero, zero, structural
    if args.aux_policy == "sensitivity_cfs":
        return sensitivity_developmental_loss(
            model, wrapper, images, mixed_targets, criterion, args,
            sampling_rng, temperature,
        )

    count = min(args.aux_samples, len(images))
    local_images, local_targets = images[:count], mixed_targets[:count]
    heads = wrapper.num_heads
    rows = torch.arange(count, device=images.device)
    source = torch.as_tensor(
        sampling_rng.integers(0, heads, size=count),
        device=images.device, dtype=torch.long,
    )
    candidates = candidate_table(source, heads)
    saved_cuda_state = torch.cuda.get_rng_state(images.device)

    wrapper.set_head_scales(None)
    with torch.no_grad(), torch.autocast(
        device_type=images.device.type, dtype=torch.float16
    ):
        reference = model(local_images).detach()
    reference_probability = reference.float().softmax(dim=-1)

    torch.cuda.set_rng_state(saved_cuda_state, images.device)
    wrapper.set_head_scales(intervention_scales(
        source, None, heads, images.dtype, args.substitution_alpha
    ))
    with torch.autocast(device_type=images.device.type, dtype=torch.float16):
        drop_logits = model(local_images)
    drop_kl = kl_to_reference(drop_logits, reference_probability)

    variant_logits = []
    for column in range(heads - 1):
        torch.cuda.set_rng_state(saved_cuda_state, images.device)
        wrapper.set_head_scales(intervention_scales(
            source, candidates[:, column], heads, images.dtype,
            args.substitution_alpha,
        ))
        with torch.autocast(device_type=images.device.type, dtype=torch.float16):
            variant_logits.append(model(local_images))
    wrapper.set_head_scales(None)
    torch.cuda.set_rng_state(saved_cuda_state, images.device)

    stacked = torch.stack(variant_logits, dim=1)
    expanded_reference = reference_probability[:, None, :].expand_as(stacked)
    candidate_kl = kl_to_reference(stacked, expanded_reference)
    candidate_task = soft_target_ce_per_sample(
        stacked, local_targets[:, None, :].expand_as(stacked)
    )
    normalized = candidate_kl / candidate_kl.detach().mean(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    soft_weights = torch.softmax(
        -normalized.detach() / max(temperature, 1e-3), dim=1
    )
    zero = candidate_kl.new_zeros(())

    if args.aux_policy == "safe_adversarial":
        safe = (
            candidate_kl.detach()
            <= args.cfs_safe_ratio * drop_kl.detach()[:, None]
        ) & (drop_kl.detach()[:, None] > args.cfs_min_drop_kl)
        safe_task = candidate_task.detach().masked_fill(~safe, -torch.inf)
        hard_safe = safe_task.argmax(dim=1)
        fallback = candidate_kl.detach().argmin(dim=1)
        selection = torch.where(safe.any(dim=1), hard_safe, fallback)
        task = candidate_task[rows, selection].mean()
        kl = candidate_kl[rows, selection].mean()
        structural = zero
    elif args.aux_policy == "soft_all":
        task = (soft_weights * candidate_task).sum(dim=1).mean()
        kl = (soft_weights * candidate_kl).sum(dim=1).mean()
        structural = zero
    elif args.aux_policy == "coverage_cfs":
        best_two = candidate_kl.detach().topk(2, largest=False, dim=1).indices
        task = candidate_task.gather(1, best_two).mean()
        kl = candidate_kl.gather(1, best_two).mean()
        second = candidate_kl.gather(1, best_two[:, 1:2]).squeeze(1)
        important = drop_kl.detach() > args.cfs_min_drop_kl
        margin = F.relu(
            second / drop_kl.detach().clamp_min(args.cfs_min_drop_kl)
            - args.cfs_safe_ratio
        )
        structural = margin[important].mean() if important.any() else zero
    elif args.aux_policy == "head_exchange":
        # Redistribute one removed branch's unit gate mass across all remaining
        # heads according to detached functional affinity.
        exchange_scales = torch.ones(
            count, heads, device=images.device, dtype=images.dtype
        )
        exchange_scales[rows, source] = 0.0
        exchange_scales.scatter_add_(1, candidates, soft_weights.to(images.dtype))
        torch.cuda.set_rng_state(saved_cuda_state, images.device)
        wrapper.set_head_scales(exchange_scales)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16):
            exchange_logits = model(local_images)
        wrapper.set_head_scales(None)
        torch.cuda.set_rng_state(saved_cuda_state, images.device)
        task = criterion(exchange_logits, local_targets)
        kl = kl_to_reference(exchange_logits, reference_probability).mean()
        structural = zero
    else:
        raise ValueError(f"Unsupported developmental policy: {args.aux_policy}")
    return task, kl, structural


def main() -> None:
    args = parse_args()
    if args.embed_dim % args.num_heads:
        raise ValueError("embed-dim must be divisible by num-heads")
    if args.effective_batch_size % args.batch_size:
        raise ValueError("effective-batch-size must be divisible by batch-size")
    seed_everything(args.seed)
    device = torch.device(args.device)
    indexing = ImageFolder(args.data_root)
    train_indices, tune_indices, test_indices, target_hash = load_or_create_split(
        args, indexing
    )
    num_classes = len(indexing.classes)
    model_config = {
        "img_size": 224,
        "patch_size": 16,
        "num_classes": num_classes,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_path_rate": args.drop_path_rate,
    }
    model = VisionTransformer(**model_config)
    if args.init_checkpoint is not None:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        checkpoint_config = payload.get("model_config", {})
        for key in ("depth", "embed_dim", "num_heads", "num_classes"):
            expected = model_config[key]
            observed = checkpoint_config.get(key, expected)
            if observed != expected:
                raise ValueError(
                    f"Initialization checkpoint {key}={observed}, expected {expected}"
                )
        checkpoint_classes = payload.get("class_to_idx")
        if checkpoint_classes is not None and checkpoint_classes != indexing.class_to_idx:
            raise ValueError("Initialization checkpoint class mapping does not match dataset")
        model.load_state_dict(state, strict=True)
    wrappers = make_intervenable(model) if args.aux_policy != "none" else []
    aux_layer_indices = []
    if wrappers:
        aux_layer_indices = [
            int(value.strip()) for value in args.aux_layers.split(",") if value.strip()
        ]
        if not aux_layer_indices:
            raise ValueError("At least one --aux-layers index is required")
        if any(index < 0 or index >= len(wrappers) for index in aux_layer_indices):
            raise ValueError(
                f"Auxiliary layer indices {aux_layer_indices} exceed depth {len(wrappers)}"
            )
    config = resolve_model_data_config(model)
    train_transform = create_transform(
        **config,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )
    eval_transform = create_transform(**config, is_training=False)
    # Reuse the single filesystem index. Reconstructing ImageFolder for every
    # transform repeatedly scans 126k files and adds avoidable startup time.
    train_base = copy.copy(indexing)
    train_base.transform = train_transform
    eval_base = copy.copy(indexing)
    eval_base.transform = eval_transform
    train_dataset = torch.utils.data.Subset(train_base, train_indices)
    tune_dataset = torch.utils.data.Subset(eval_base, tune_indices)
    test_dataset = torch.utils.data.Subset(eval_base, test_indices)
    if args.max_eval_samples is not None:
        tune_dataset = torch.utils.data.Subset(
            tune_dataset, range(min(args.max_eval_samples, len(tune_dataset)))
        )
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(min(args.max_eval_samples, len(test_dataset)))
        )
    generator = torch.Generator().manual_seed(args.seed)
    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        drop_last=True, **common,
    )
    tune_loader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=args.batch_size * 2, shuffle=False, **common,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False, **common,
    )
    use_mixing = args.mixup_alpha > 0 or args.cutmix_alpha > 0
    mixup = (
        Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=num_classes,
        )
        if use_mixing else None
    )
    criterion = (
        SoftTargetCrossEntropy()
        if use_mixing
        else LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        if args.label_smoothing > 0
        else torch.nn.CrossEntropyLoss()
    )
    model.to(device, memory_format=torch.channels_last)
    model_ema = (
        ModelEmaV2(model, decay=args.ema_decay, device=None)
        if args.ema_decay > 0 else None
    )
    initial_tune_loss, initial_tune_accuracy = evaluate(model, tune_loader, device)
    print(json.dumps({
        "stage": "initialization",
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "tune_loss": initial_tune_loss,
        "tune_accuracy": initial_tune_accuracy,
    }), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    accumulation = args.effective_batch_size // args.batch_size
    if args.sam_rho > 0 and accumulation != 1:
        raise ValueError("SAM requires effective batch size == batch size")
    if args.sam_rho > 0 and wrappers:
        raise ValueError("SAM audit requires aux-policy=none")
    micro_steps = min(len(train_loader), args.max_steps or len(train_loader))
    optimizer_steps_per_epoch = math.ceil(micro_steps / accumulation)
    total_steps = max(1, args.epochs * optimizer_steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_epochs * optimizer_steps_per_epoch))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_tune = -1.0
    best_ema_tune = -1.0
    history = []
    milestone_epochs = {
        int(value.strip()) for value in args.milestone_epochs.split(",")
        if value.strip()
    }
    sampling_rng = np.random.default_rng(args.seed + 81023)
    started = time.perf_counter()

    def save_auxiliary_checkpoint(
        checkpoint_model: torch.nn.Module, path: Path, role: str
    ) -> None:
        torch.save(
            {
                "model": "controlled_vit",
                "checkpoint_role": role,
                "model_config": model_config,
                "num_classes": num_classes,
                "class_to_idx": indexing.class_to_idx,
                "state_dict": canonical_state_dict(checkpoint_model),
                "trained_num_heads": args.num_heads,
                "seed": args.seed,
                "split_manifest": str(args.split_manifest.resolve()),
                "training_protocol": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in (vars(args) | {
                        "data_root": str(args.data_root.resolve()),
                        "output": str(args.output.resolve()),
                    }).items()
                },
                "history": history,
            },
            path,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        aux_task_sum = 0.0
        aux_kl_sum = 0.0
        aux_structure_sum = 0.0
        aux_batches = 0
        sam_overflow_batches = 0
        seen = 0
        cfs_strength = (
            developmental_strength(epoch, args)
            if args.aux_policy in DEVELOPMENTAL_POLICIES else 1.0
        )
        cfs_temperature = developmental_temperature(epoch, args)
        for micro_step, (images, targets) in enumerate(train_loader):
            if micro_step >= micro_steps:
                break
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            if mixup is None:
                mixed_targets = targets
            else:
                images, mixed_targets = mixup(images, targets)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                raw_loss = criterion(logits, mixed_targets)
                if wrappers and cfs_strength > 0:
                    if args.aux_policy in DEVELOPMENTAL_POLICIES:
                        layer_index = int(sampling_rng.choice(aux_layer_indices))
                        aux_task, aux_kl, aux_structure = (
                            developmental_intervention_loss(
                                model, wrappers[layer_index], images, mixed_targets,
                                criterion, args, sampling_rng, cfs_temperature,
                            )
                        )
                    else:
                        aux_task, aux_kl = auxiliary_intervention_loss(
                            model, wrappers[-1], images, mixed_targets, criterion,
                            args, sampling_rng,
                        )
                        aux_structure = aux_kl.new_zeros(())
                    raw_loss = (
                        raw_loss
                        + cfs_strength * (
                            args.aux_task_weight * aux_task
                            + args.aux_kl_weight * aux_kl
                            + args.aux_structure_weight * aux_structure
                        )
                    )
                    aux_task_sum += float(aux_task.detach())
                    aux_kl_sum += float(aux_kl.detach())
                    aux_structure_sum += float(aux_structure.detach())
                    aux_batches += 1
                loss = raw_loss / accumulation
            if args.sam_rho > 0:
                # The scale factor applied by GradScaler cancels between the
                # gradient and its global norm, so the SAM perturbation can be
                # formed before the single allowed unscale_ call. Only the
                # second-pass gradient is used by AdamW.
                scaler.scale(raw_loss).backward()
                grad_parameters = [
                    parameter for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                grad_norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
                    for parameter in grad_parameters
                ]), ord=2)
                if not torch.isfinite(grad_norm):
                    # Match GradScaler's normal overflow recovery. The first
                    # SAM pass is not an optimizer step, so it must explicitly
                    # back off the scale before skipping this batch.
                    sam_overflow_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update(new_scale=max(1.0, scaler.get_scale() / 2.0))
                    continue
                perturbation_scale = args.sam_rho / (grad_norm + 1e-12)
                perturbations = []
                with torch.no_grad():
                    for parameter in grad_parameters:
                        perturbation = parameter.grad * perturbation_scale
                        parameter.add_(perturbation)
                        perturbations.append((parameter, perturbation))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    sam_logits = model(images)
                    sam_loss = criterion(sam_logits, mixed_targets)
                scaler.scale(sam_loss).backward()
                with torch.no_grad():
                    for parameter, perturbation in perturbations:
                        parameter.sub_(perturbation)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    if model_ema is not None:
                        model_ema.update(model)
                    scheduler.step()
                loss_sum += float(raw_loss.detach()) * len(targets)
                seen += len(targets)
                continue
            scaler.scale(loss).backward()
            loss_sum += float(raw_loss.detach()) * len(targets)
            seen += len(targets)
            boundary = (micro_step + 1) % accumulation == 0
            final = micro_step + 1 == micro_steps
            if boundary or final:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    if model_ema is not None:
                        model_ema.update(model)
                    scheduler.step()
        tune_loss, tune_accuracy = evaluate(model, tune_loader, device)
        if model_ema is None:
            ema_tune_loss, ema_tune_accuracy = None, None
        else:
            ema_tune_loss, ema_tune_accuracy = evaluate(
                model_ema.module, tune_loader, device
            )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "tune_loss": tune_loss,
            "tune_accuracy": tune_accuracy,
            "ema_tune_loss": ema_tune_loss,
            "ema_tune_accuracy": ema_tune_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.perf_counter() - started) / 60,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "aux_policy": args.aux_policy,
            "cfs_strength": cfs_strength if wrappers else None,
            "cfs_temperature": cfs_temperature if wrappers else None,
            "aux_task_loss": aux_task_sum / aux_batches if aux_batches else None,
            "aux_kl": aux_kl_sum / aux_batches if aux_batches else None,
            "aux_structure": (
                aux_structure_sum / aux_batches if aux_batches else None
            ),
            "functional_basis_energy": (
                float(np.mean([
                    item.functional_basis_last_energy for item in wrappers
                    if hasattr(item, "functional_basis_last_energy")
                ]))
                if args.aux_policy == "functional_basis" and any(
                    hasattr(item, "functional_basis_last_energy") for item in wrappers
                ) else None
            ),
            "functional_basis_effective_rank": (
                float(np.mean([
                    item.functional_basis_last_effective_rank for item in wrappers
                    if hasattr(item, "functional_basis_last_effective_rank")
                ]))
                if args.aux_policy == "functional_basis" and any(
                    hasattr(item, "functional_basis_last_effective_rank")
                    for item in wrappers
                ) else None
            ),
            "sam_overflow_batches": sam_overflow_batches,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if tune_accuracy > best_tune:
            best_tune = tune_accuracy
            torch.save(
                {
                    "model": "controlled_vit",
                    "model_config": model_config,
                    "num_classes": num_classes,
                    "class_to_idx": indexing.class_to_idx,
                    "state_dict": canonical_state_dict(model),
                    "trained_num_heads": args.num_heads,
                    "seed": args.seed,
                    "split_manifest": str(args.split_manifest.resolve()),
                    "training_protocol": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in (vars(args) | {
                        "data_root": str(args.data_root.resolve()),
                        "output": str(args.output.resolve()),
                        }).items()
                    },
                    "history": history,
                },
                args.output,
            )
        if model_ema is not None and ema_tune_accuracy > best_ema_tune:
            best_ema_tune = ema_tune_accuracy
            ema_best_output = args.output.with_name(
                f"{args.output.stem}.ema{args.output.suffix}"
            )
            save_auxiliary_checkpoint(
                model_ema.module, ema_best_output, "ema_best"
            )
        if args.save_every_epoch or epoch in milestone_epochs:
            milestone_output = args.output.with_name(
                f"{args.output.stem}.epoch{epoch:03d}{args.output.suffix}"
            )
            save_auxiliary_checkpoint(model, milestone_output, f"epoch_{epoch}")
            if model_ema is not None:
                ema_milestone_output = args.output.with_name(
                    f"{args.output.stem}.ema.epoch{epoch:03d}{args.output.suffix}"
                )
                save_auxiliary_checkpoint(
                    model_ema.module, ema_milestone_output, f"ema_epoch_{epoch}"
                )
    # Preserve the terminal optimization state separately from the
    # validation-selected recognition checkpoint. Functional analysis uses the
    # final state so that checkpoint selection noise cannot select a functional
    # topology post hoc.
    if args.skip_test:
        final_test_loss, final_test_accuracy = None, None
    else:
        final_test_loss, final_test_accuracy = evaluate(model, test_loader, device)
    final_output = args.output.with_name(
        f"{args.output.stem}.final{args.output.suffix}"
    )
    torch.save(
        {
            "model": "controlled_vit",
            "checkpoint_role": "final",
            "model_config": model_config,
            "num_classes": num_classes,
            "class_to_idx": indexing.class_to_idx,
            "state_dict": canonical_state_dict(model),
            "trained_num_heads": args.num_heads,
            "seed": args.seed,
            "split_manifest": str(args.split_manifest.resolve()),
            "training_protocol": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in (vars(args) | {
                    "data_root": str(args.data_root.resolve()),
                    "output": str(args.output.resolve()),
                }).items()
            },
            "history": history,
        },
        final_output,
    )
    if args.save_final_training_state:
        training_state_output = args.output.with_name(
            f"{args.output.stem}.trainstate.final{args.output.suffix}"
        )
        torch.save(
            {
                "model": "controlled_vit",
                "checkpoint_role": "final_training_state",
                "epoch": args.epochs,
                "model_config": model_config,
                "num_classes": num_classes,
                "class_to_idx": indexing.class_to_idx,
                "state_dict": canonical_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "grad_scaler_state_dict": scaler.state_dict(),
                "trained_num_heads": args.num_heads,
                "seed": args.seed,
                "split_manifest": str(args.split_manifest.resolve()),
                "training_protocol": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in (vars(args) | {
                        "data_root": str(args.data_root.resolve()),
                        "output": str(args.output.resolve()),
                    }).items()
                },
                "history": history,
            },
            training_state_output,
        )
    else:
        training_state_output = None
    if model_ema is None:
        ema_final_output = None
    else:
        ema_final_output = args.output.with_name(
            f"{args.output.stem}.ema.final{args.output.suffix}"
        )
        save_auxiliary_checkpoint(
            model_ema.module, ema_final_output, "ema_final"
        )
    payload = torch.load(args.output, map_location="cpu", weights_only=True)
    load_canonical_state_dict(model, payload["state_dict"])
    if args.skip_test:
        test_loss, test_accuracy = None, None
    else:
        test_loss, test_accuracy = evaluate(model, test_loader, device)
    summary = {
        "best_tune_accuracy": best_tune,
        "best_ema_tune_accuracy": (
            best_ema_tune if model_ema is not None else None
        ),
        "initial_tune_loss": initial_tune_loss,
        "initial_tune_accuracy": initial_tune_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
        "best_checkpoint": str(args.output),
        "final_checkpoint": str(final_output),
        "ema_best_checkpoint": (
            str(args.output.with_name(
                f"{args.output.stem}.ema{args.output.suffix}"
            )) if model_ema is not None else None
        ),
        "ema_final_checkpoint": (
            str(ema_final_output) if ema_final_output is not None else None
        ),
        "final_training_state": (
            str(training_state_output)
            if training_state_output is not None else None
        ),
        "history": history,
        "train_examples": len(train_dataset),
        "tune_examples": len(tune_dataset),
        "test_examples": len(test_dataset),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": model_config,
        "head_dim": args.embed_dim // args.num_heads,
        "target_hash": target_hash,
        "scientific_scope": (
            "fixed-checkpoint CFS post-training screen"
            if args.init_checkpoint is not None
            else "random-initialized one-axis controlled ViT scaling"
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"saved": str(args.output), **summary | {"history": "omitted"}}), flush=True)


if __name__ == "__main__":
    main()
