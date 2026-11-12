"""Real selected-head execution for timm Vision Transformer attention.

Unlike post-hoc head masking, this module slices the QKV output rows and the
output-projection input columns before the matrix multiplications. Therefore
unselected query heads are not computed. Dynamic per-sample masks are grouped
by identical subsets so every executed GEMM contains only retained heads.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectedHeadAttention(nn.Module):
    """Execute only heads selected by a static mask or runtime router."""

    def __init__(
        self,
        attention: nn.Module,
        selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(attention.norm, nn.Identity):
            raise ValueError("Selected-head execution currently requires attention.norm=Identity")
        if attention.gate is not None:
            raise ValueError("Gated timm attention is not supported")
        self.attention = attention
        self.selector = selector
        self.last_mask: Optional[torch.Tensor] = None
        self.last_unique_subsets = 0

    @property
    def num_heads(self) -> int:
        return int(self.attention.num_heads)

    @property
    def head_dim(self) -> int:
        return int(self.attention.head_dim)

    def set_selector(
        self, selector: Optional[Callable[[torch.Tensor], torch.Tensor]]
    ) -> None:
        self.selector = selector

    def _selected_forward(
        self,
        x: torch.Tensor,
        selected: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        attn = self.attention
        batch, tokens, _ = x.shape
        selected = selected.to(device=x.device, dtype=torch.long)
        kept = int(selected.numel())
        if kept == 0:
            raise ValueError("At least one attention head must be retained")

        offsets = torch.arange(self.head_dim, device=x.device)
        head_columns = (selected[:, None] * self.head_dim + offsets[None]).reshape(-1)
        qkv_rows = torch.cat(
            [head_columns + part * attn.attn_dim for part in range(3)], dim=0
        )
        qkv_weight = attn.qkv.weight.index_select(0, qkv_rows)
        qkv_bias = (
            attn.qkv.bias.index_select(0, qkv_rows)
            if attn.qkv.bias is not None
            else None
        )
        qkv = F.linear(x, qkv_weight, qkv_bias)
        qkv = qkv.reshape(batch, tokens, 3, kept, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = attn.q_norm(query), attn.k_norm(key)

        if attn.fused_attn:
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
            weights = query @ key.transpose(-2, -1)
            if attn_mask is not None:
                weights = weights + attn_mask
            weights = attn.attn_drop(weights.softmax(dim=-1))
            heads = weights @ value

        merged = heads.transpose(1, 2).reshape(batch, tokens, kept * self.head_dim)
        proj_weight = attn.proj.weight.index_select(1, head_columns)
        output = F.linear(merged, proj_weight, attn.proj.bias)
        return attn.proj_drop(output)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.selector is None:
            self.last_mask = None
            self.last_unique_subsets = 1
            return self.attention(x, attn_mask=attn_mask, is_causal=is_causal)

        mask = self.selector(x).to(device=x.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(x.shape[0], -1)
        if mask.shape != (x.shape[0], self.num_heads):
            raise ValueError(
                f"Expected selector mask {(x.shape[0], self.num_heads)}, got {tuple(mask.shape)}"
            )
        self.last_mask = mask.detach()
        if x.shape[0] == 1:
            # Exact-budget selectors expose ``keep``. topk returns a
            # fixed-shape index tensor and avoids CUDA-synchronizing nonzero,
            # unique/group discovery, and scatter for online latency.
            keep = getattr(self.selector, "keep", None)
            if keep is not None:
                selected = torch.topk(mask[0].to(torch.uint8), int(keep)).indices
                selected = selected.sort().values
                self.last_unique_subsets = 1
                return self._selected_forward(
                    x, selected, attn_mask=attn_mask, is_causal=is_causal
                )
        unique_masks, inverse = torch.unique(mask, dim=0, return_inverse=True)
        self.last_unique_subsets = int(unique_masks.shape[0])
        output = torch.empty(
            x.shape[0], x.shape[1], self.attention.proj.out_features,
            device=x.device, dtype=x.dtype,
        )
        for group, group_mask in enumerate(unique_masks):
            batch_indices = torch.nonzero(inverse == group, as_tuple=False).flatten()
            selected = torch.nonzero(group_mask, as_tuple=False).flatten()
            group_mask_value = None
            if attn_mask is not None and attn_mask.shape[0] == x.shape[0]:
                group_mask_value = attn_mask.index_select(0, batch_indices)
            else:
                group_mask_value = attn_mask
            group_output = self._selected_forward(
                x.index_select(0, batch_indices), selected, group_mask_value, is_causal
            )
            output.index_copy_(0, batch_indices, group_output)
        return output


class StaticHeadSelector(nn.Module):
    """Return one fixed exact-budget mask for every sample."""

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        if mask.ndim != 1:
            raise ValueError("Static mask must have shape [H]")
        self.register_buffer("mask", mask.bool())
        self.keep = int(mask.bool().sum())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mask.unsqueeze(0).expand(hidden.shape[0], -1)


class PackedStaticHeadAttention(nn.Module):
    """Prepack one fixed subset to measure the selected-head kernel ceiling.

    This removes runtime weight gathering and mask dispatch. It is not a
    dynamic router, but establishes whether the reduced GEMMs/SDPA are faster
    on the target device before dynamic-system overhead is considered.
    """

    def __init__(self, attention: nn.Module, mask: torch.Tensor) -> None:
        super().__init__()
        if not isinstance(attention.norm, nn.Identity):
            raise ValueError("Packed execution currently requires attention.norm=Identity")
        if attention.gate is not None:
            raise ValueError("Gated timm attention is not supported")
        selected = torch.nonzero(mask.bool(), as_tuple=False).flatten().to(
            attention.qkv.weight.device
        )
        if selected.numel() == 0:
            raise ValueError("At least one attention head must be retained")
        self.head_dim = int(attention.head_dim)
        self.keep = int(selected.numel())
        self.scale = float(attention.scale)
        self.fused_attn = bool(attention.fused_attn)
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.attn_drop = attention.attn_drop
        self.proj_drop = attention.proj_drop

        offsets = torch.arange(self.head_dim, device=selected.device)
        columns = (selected[:, None] * self.head_dim + offsets[None]).reshape(-1)
        rows = torch.cat(
            [columns + part * int(attention.attn_dim) for part in range(3)]
        )
        self.register_buffer(
            "qkv_weight", attention.qkv.weight.index_select(0, rows).detach().contiguous()
        )
        if attention.qkv.bias is None:
            self.qkv_bias = None
        else:
            self.register_buffer(
                "qkv_bias", attention.qkv.bias.index_select(0, rows).detach().contiguous()
            )
        self.register_buffer(
            "proj_weight",
            attention.proj.weight.index_select(1, columns).detach().contiguous(),
        )
        if attention.proj.bias is None:
            self.proj_bias = None
        else:
            self.register_buffer("proj_bias", attention.proj.bias.detach().clone())

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = F.linear(x, self.qkv_weight, self.qkv_bias)
        qkv = qkv.reshape(batch, tokens, 3, self.keep, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        if self.fused_attn:
            heads = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            weights = (query * self.scale) @ key.transpose(-2, -1)
            if attn_mask is not None:
                weights = weights + attn_mask
            heads = self.attn_drop(weights.softmax(dim=-1)) @ value
        merged = heads.transpose(1, 2).reshape(batch, tokens, self.keep * self.head_dim)
        return self.proj_drop(F.linear(merged, self.proj_weight, self.proj_bias))
