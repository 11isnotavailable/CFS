"""Per-head gating at the input of a Hugging Face GPT-2 attention c_proj."""

from __future__ import annotations

import torch


class GPT2HeadGate:
    """Scale additive attention-head branches immediately before c_proj."""

    def __init__(self, model: torch.nn.Module, layer_index: int):
        self.attention = model.transformer.h[int(layer_index)].attn
        self.num_heads = int(model.config.n_head)
        self.width = int(model.config.n_embd)
        if self.width % self.num_heads:
            raise ValueError("GPT-2 hidden width must be divisible by head count")
        self.head_dim = self.width // self.num_heads
        self.scales: torch.Tensor | None = None
        self.handle = self.attention.c_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        if self.scales is None:
            return None
        hidden = inputs[0]
        batch, tokens, width = hidden.shape
        if width != self.width:
            raise RuntimeError(f"Unexpected attention width {width}, expected {self.width}")
        scales = self.scales
        if scales.ndim == 1:
            scales = scales.unsqueeze(0)
        if scales.shape not in ((1, self.num_heads), (batch, self.num_heads)):
            raise ValueError(f"Invalid scales shape {tuple(scales.shape)}")
        gated = hidden.reshape(batch, tokens, self.num_heads, self.head_dim)
        gated = gated * scales.to(hidden.device, hidden.dtype)[:, None, :, None]
        return (gated.reshape(batch, tokens, width),) + tuple(inputs[1:])

    def set_scales(self, scales: torch.Tensor | None) -> None:
        self.scales = scales

    def close(self) -> None:
        self.handle.remove()
