"""Query-head gating at the input of a Qwen attention output projection."""

from __future__ import annotations

import torch


class QwenHeadGate:
    """Gate additive query-head contributions before ``o_proj``.

    Qwen attention concatenates ``H`` query-head outputs and applies one linear
    output projection.  Scaling slices of that concatenation is algebraically
    equivalent to scaling each additive post-projection contribution.
    """

    def __init__(
        self, model: torch.nn.Module, layer_index: int, num_units: int | None = None
    ):
        self.layer_index = int(layer_index)
        self.attention = model.model.layers[layer_index].self_attn
        self.num_heads = int(model.config.num_attention_heads)
        self.num_units = self.num_heads if num_units is None else int(num_units)
        if int(model.config.hidden_size) % self.num_units:
            raise ValueError(
                f"hidden_size={model.config.hidden_size} is not divisible by "
                f"num_units={self.num_units}"
            )
        self.unit_dim = int(model.config.hidden_size // self.num_units)
        self.scales: torch.Tensor | None = None
        self.handle = self.attention.o_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        if self.scales is None:
            return None
        hidden = inputs[0]
        batch, tokens, width = hidden.shape
        if width != self.num_units * self.unit_dim:
            raise RuntimeError(f"Unexpected attention width {width}")
        scales = self.scales
        if scales.ndim == 1:
            scales = scales.unsqueeze(0)
        if scales.shape not in ((1, self.num_units), (batch, self.num_units)):
            raise ValueError(
                f"Expected scales [U] or [B,U], received {tuple(scales.shape)}"
            )
        gated = hidden.reshape(batch, tokens, self.num_units, self.unit_dim)
        gated = gated * scales.to(device=hidden.device, dtype=hidden.dtype)[:, None, :, None]
        return (gated.reshape(batch, tokens, width),) + tuple(inputs[1:])

    def set_scales(self, scales: torch.Tensor | None) -> None:
        self.scales = scales

    def close(self) -> None:
        self.handle.remove()
