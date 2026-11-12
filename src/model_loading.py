"""Shared timm model loading for pretrained and probed checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def repartition_timm_attention_heads(
    model: torch.nn.Module,
    num_heads: int,
) -> None:
    """Reinterpret fixed ViT QKV/WO weights using a new head partition.

    This changes the attention computation and is intended only for controlled
    diagnostic pilots.  It does not create a checkpoint trained with the new
    head count and must not be described as an architectural scaling run.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Expected a timm Vision Transformer with model.blocks.")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    for block in model.blocks:
        attention = block.attn
        attention_dim = int(attention.attn_dim)
        if attention_dim % num_heads:
            raise ValueError(
                f"Attention dimension {attention_dim} is not divisible by {num_heads}."
            )
        if not isinstance(attention.q_norm, torch.nn.Identity) or not isinstance(
            attention.k_norm, torch.nn.Identity
        ):
            raise ValueError("Head repartition requires identity Q/K normalization.")
        attention.num_heads = int(num_heads)
        attention.head_dim = attention_dim // num_heads
        attention.scale = attention.head_dim ** -0.5


def create_timm_model(
    model_name: str,
    checkpoint: Path | None = None,
) -> tuple[torch.nn.Module, dict[str, int] | None]:
    import timm

    if checkpoint is None:
        return timm.create_model(model_name, pretrained=True), None

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if payload.get("model_config") is not None:
        from timm.models.vision_transformer import VisionTransformer

        model = VisionTransformer(**payload["model_config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        raw_mapping = payload.get("class_to_idx")
        mapping = (
            {str(key): int(value) for key, value in raw_mapping.items()}
            if raw_mapping is not None
            else None
        )
        return model, mapping
    checkpoint_model = str(payload["model"])
    if checkpoint_model != model_name:
        raise ValueError(
            f"Checkpoint model {checkpoint_model!r} does not match "
            f"--model {model_name!r}."
        )
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=int(payload["num_classes"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    raw_mapping = payload.get("class_to_idx")
    mapping = (
        {
            str(key): int(value)
            for key, value in raw_mapping.items()
        }
        if raw_mapping is not None
        else None
    )
    return model, mapping
