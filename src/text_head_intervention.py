"""Head-scale interventions for Hugging Face BERT sequence classifiers.

The intervention is applied to the concatenated per-head context vectors
immediately before BertSelfOutput.dense.  Since that projection is linear,
scaling one context block is exactly equivalent to scaling that head's
post-W_O additive contribution while applying the shared bias only once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertAttentionCapture:
    hidden: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    aggregated: torch.Tensor
    attention: torch.Tensor | None


class IntervenableBertSelfAttention(nn.Module):
    """Transparent wrapper around ``BertSelfAttention`` with head gates."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.num_heads = int(inner.num_attention_heads)
        self.head_dim = int(inner.attention_head_size)
        self.head_scales: torch.Tensor | None = None
        self.capture_enabled = False
        self.capture: BertAttentionCapture | None = None

    def set_head_scales(self, scales: torch.Tensor | None) -> None:
        if scales is not None and scales.ndim != 2:
            raise ValueError("Head scales must have shape [batch, heads].")
        if scales is not None and scales.shape[1] != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} heads, got {scales.shape[1]}."
            )
        self.head_scales = scales

    def set_capture(self, enabled: bool) -> None:
        self.capture_enabled = enabled
        if not enabled:
            self.capture = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, weights = self.inner(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        batch, tokens, width = output.shape
        heads = output.view(batch, tokens, self.num_heads, self.head_dim)

        if self.capture_enabled:
            shape = (batch, tokens, self.num_heads, self.head_dim)
            query = self.inner.query(hidden_states).view(shape).transpose(1, 2)
            key = self.inner.key(hidden_states).view(shape).transpose(1, 2)
            value = self.inner.value(hidden_states).view(shape).transpose(1, 2)
            self.capture = BertAttentionCapture(
                hidden=hidden_states,
                query=query,
                key=key,
                value=value,
                aggregated=heads.transpose(1, 2),
                attention=weights,
            )

        if self.head_scales is not None:
            if self.head_scales.shape[0] != batch:
                raise ValueError(
                    "Head-scale batch does not match the model-input batch."
                )
            heads = heads * self.head_scales[:, None, :, None].to(
                device=heads.device,
                dtype=heads.dtype,
            )
            output = heads.reshape(batch, tokens, width).contiguous()
        return output, weights


def make_bert_intervenable(model: nn.Module) -> list[IntervenableBertSelfAttention]:
    wrappers: list[IntervenableBertSelfAttention] = []
    for layer in model.bert.encoder.layer:
        original = layer.attention.self
        if isinstance(original, IntervenableBertSelfAttention):
            wrapper = original
        else:
            wrapper = IntervenableBertSelfAttention(original)
            layer.attention.self = wrapper
        wrappers.append(wrapper)
    return wrappers


def clear_bert_interventions(
    wrappers: list[IntervenableBertSelfAttention],
) -> None:
    for wrapper in wrappers:
        wrapper.set_head_scales(None)
        wrapper.set_capture(False)


def post_wo_contributions(
    model: nn.Module,
    layer_index: int,
    aggregated: torch.Tensor,
) -> torch.Tensor:
    """Return independent post-W_O head contributions [B,H,T,D]."""
    dense = model.bert.encoder.layer[layer_index].attention.output.dense
    contributions = []
    head_dim = aggregated.shape[-1]
    for head in range(aggregated.shape[1]):
        begin = head * head_dim
        end = begin + head_dim
        contributions.append(
            F.linear(aggregated[:, head], dense.weight[:, begin:end], bias=None)
        )
    return torch.stack(contributions, dim=1)
