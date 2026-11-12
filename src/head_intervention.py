"""Per-sample attention-head interventions for timm Vision Transformers."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntervenableAttention(nn.Module):
    """Wrap a timm Attention module with per-sample head scaling and capture.

    Scaling is applied to each head immediately before the output projection.
    Consequently, a scale of zero removes both the head activation and its
    corresponding block contribution through W_O, while the projection bias is
    still applied exactly once.
    """

    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attention = attention
        self.head_scales: Optional[torch.Tensor] = None
        self.runtime_router: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self.capture_contributions = False
        self.last_contributions: Optional[torch.Tensor] = None
        self.capture_attention_map = False
        self.last_attention_map: Optional[torch.Tensor] = None
        self.capture_input = False
        self.last_input: Optional[torch.Tensor] = None
        self.capture_heads_with_grad = False
        self.last_heads_with_grad: Optional[torch.Tensor] = None

        if not isinstance(attention.norm, nn.Identity):
            raise ValueError(
                "Head decomposition currently requires attention.norm=Identity; "
                f"received {type(attention.norm).__name__}."
            )
        if attention.gate is not None:
            raise ValueError("Gated attention is not supported in Experiment 0.")

    @property
    def num_heads(self) -> int:
        return int(self.attention.num_heads)

    @property
    def head_dim(self) -> int:
        return int(self.attention.head_dim)

    def set_head_scales(self, scales: Optional[torch.Tensor]) -> None:
        """Set scales shaped [H] or [B, H]; None restores the dense model."""
        if scales is not None:
            if scales.ndim not in (1, 2):
                raise ValueError(f"Expected [H] or [B,H] scales, got {scales.shape}.")
            if scales.shape[-1] != self.num_heads:
                raise ValueError(
                    f"Expected {self.num_heads} heads, got {scales.shape[-1]}."
                )
        self.head_scales = scales

    def set_runtime_router(
        self, router: Optional[Callable[[torch.Tensor], torch.Tensor]]
    ) -> None:
        """Attach a leakage-safe router evaluated before this attention."""
        self.runtime_router = router

    def _scaled_head_tensor(
        self,
        heads: torch.Tensor,
        runtime_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scales_value = self.head_scales if runtime_scales is None else runtime_scales
        if scales_value is None:
            return heads
        scales = scales_value.to(device=heads.device, dtype=heads.dtype)
        if scales.ndim == 1:
            scales = scales.unsqueeze(0)
        if scales.shape[0] not in (1, heads.shape[0]):
            raise ValueError(
                f"Scale batch {scales.shape[0]} does not match input batch "
                f"{heads.shape[0]}."
            )
        return heads * scales[:, None, :, None]

    def _capture_post_wo_contributions(self, heads: torch.Tensor) -> None:
        # heads: [B, N, H, D]. W_O is [C_out, H*D].
        weight = self.attention.proj.weight
        contributions = []
        for head_idx in range(self.num_heads):
            start = head_idx * self.head_dim
            stop = start + self.head_dim
            contribution = F.linear(
                heads[:, :, head_idx, :],
                weight[:, start:stop],
                bias=None,
            )
            contributions.append(contribution)
        self.last_contributions = torch.stack(contributions, dim=1).detach()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn = self.attention
        runtime_scales = (
            self.runtime_router(x) if self.runtime_router is not None else None
        )
        if self.capture_input:
            self.last_input = x.detach()
        batch, tokens, _ = x.shape
        qkv = (
            attn.qkv(x)
            .reshape(batch, tokens, 3, attn.num_heads, attn.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        query, key = attn.q_norm(query), attn.k_norm(key)

        if attn.fused_attn and not self.capture_attention_map:
            heads = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=attn.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            query = query * attn.scale
            attention_map = query @ key.transpose(-2, -1)
            if attn_mask is not None:
                attention_map = attention_map + attn_mask
            attention_map = attention_map.softmax(dim=-1)
            attention_map = attn.attn_drop(attention_map)
            if self.capture_attention_map:
                self.last_attention_map = attention_map.detach()
            heads = attention_map @ value

        heads = heads.transpose(1, 2)  # [B, N, H, D]
        if self.capture_heads_with_grad:
            # Keep the exact tensor used by the attention output so a caller
            # can measure task sensitivity with respect to individual heads.
            self.last_heads_with_grad = heads
        if self.capture_contributions:
            self._capture_post_wo_contributions(heads)

        heads = self._scaled_head_tensor(heads, runtime_scales)
        merged = heads.reshape(batch, tokens, attn.attn_dim)
        merged = attn.norm(merged)
        output = attn.proj(merged)
        return attn.proj_drop(output)


def make_intervenable(model: nn.Module) -> list[IntervenableAttention]:
    """Replace every ``block.attn`` in a timm ViT and return the wrappers."""
    wrappers: list[IntervenableAttention] = []
    if not hasattr(model, "blocks"):
        raise ValueError("Expected a timm Vision Transformer with model.blocks.")
    for block in model.blocks:
        wrapper = IntervenableAttention(block.attn)
        block.attn = wrapper
        wrappers.append(wrapper)
    return wrappers


def clear_interventions(wrappers: list[IntervenableAttention]) -> None:
    for wrapper in wrappers:
        wrapper.set_head_scales(None)
        wrapper.capture_contributions = False
        wrapper.last_contributions = None
        wrapper.capture_attention_map = False
        wrapper.last_attention_map = None
        wrapper.capture_input = False
        wrapper.last_input = None
        wrapper.capture_heads_with_grad = False
        wrapper.last_heads_with_grad = None
