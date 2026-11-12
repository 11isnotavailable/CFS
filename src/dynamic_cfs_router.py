"""Forward-only structured router for conditional head substitutability."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn


def subset_masks(heads: int, keep: int, device=None) -> torch.Tensor:
    """Return every size-``keep`` head subset as a boolean mask."""
    if not 1 <= keep <= heads:
        raise ValueError(f"keep must be in [1, {heads}], got {keep}")
    masks = torch.zeros(
        len(list(itertools.combinations(range(heads), keep))),
        heads,
        dtype=torch.bool,
        device=device,
    )
    for row, indices in enumerate(itertools.combinations(range(heads), keep)):
        masks[row, list(indices)] = True
    return masks


def subset_utilities(
    scores: torch.Tensor,
    importance: torch.Tensor,
    masks: torch.Tensor,
    mask_chunk_size: int = 128,
) -> torch.Tensor:
    """Importance-weighted recoverable utility for candidate keep sets.

    ``scores[b, i, j]`` is the recoverable fraction when source ``i`` is
    represented by kept head ``j``. Kept sources receive utility one. Scores
    are clipped to [0, 1] so the result is an interpretable retained fraction.
    """
    if scores.ndim != 3 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must have shape [B,H,H]")
    if importance.shape != scores.shape[:2]:
        raise ValueError("importance must have shape [B,H]")
    if masks.ndim != 2 or masks.shape[1] != scores.shape[-1]:
        raise ValueError("masks must have shape [M,H]")
    weights = importance.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    bounded = scores.clamp(0, 1)
    chunks = []
    # Vectorize over subsets but retain a bounded chunk dimension.  For H=12,
    # the middle budget has 924 masks; constructing [B,924,H,H] at once wastes
    # memory, while the former Python-per-mask implementation is needlessly
    # slow and creates a very large autograd graph.
    for begin in range(0, len(masks), mask_chunk_size):
        mask = masks[begin : begin + mask_chunk_size]
        candidate = bounded[:, None, :, :].masked_fill(
            ~mask[None, :, None, :], -torch.inf
        )
        recovered = candidate.max(dim=-1).values
        retained = torch.where(
            mask[None, :, :], torch.ones_like(recovered), recovered
        )
        chunks.append((weights[:, None, :] * retained).sum(dim=-1))
    return torch.cat(chunks, dim=-1).clamp(0, 1)


def selected_masks(
    scores: torch.Tensor,
    importance: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the highest-utility mask for each sample."""
    utilities = subset_utilities(scores, importance, masks)
    choice = utilities.argmax(dim=-1)
    return masks[choice], utilities.gather(1, choice[:, None]).squeeze(1)


def realized_utilities(
    scores: torch.Tensor,
    importance: torch.Tensor,
    keep_masks: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one sample-specific keep mask per score matrix."""
    if keep_masks.shape != importance.shape:
        raise ValueError("keep_masks and importance must both have shape [B,H]")
    weights = importance.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    candidate = scores.clamp(0, 1).masked_fill(
        ~keep_masks[:, None, :], -torch.inf
    )
    recovered = candidate.max(dim=-1).values
    retained = torch.where(keep_masks, torch.ones_like(recovered), recovered)
    return (weights * retained).sum(dim=-1).clamp(0, 1)


def candidate_subset_utilities(
    scores: torch.Tensor,
    importance: torch.Tensor,
    keep_masks: torch.Tensor,
) -> torch.Tensor:
    """Evaluate sample-specific candidate sets shaped ``[B,C,H]``."""
    if keep_masks.ndim != 3 or keep_masks.shape[0] != scores.shape[0]:
        raise ValueError("keep_masks must have shape [B,C,H]")
    if keep_masks.shape[-1] != scores.shape[-1]:
        raise ValueError("keep_masks head dimension does not match scores")
    weights = importance.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    candidate = scores.clamp(0, 1)[:, None, :, :].masked_fill(
        ~keep_masks[:, :, None, :], -torch.inf
    )
    recovered = candidate.max(dim=-1).values
    retained = torch.where(
        keep_masks, torch.ones_like(recovered), recovered
    )
    return (weights[:, None, :] * retained).sum(dim=-1).clamp(0, 1)


def compose_scales(
    keep_masks: torch.Tensor,
    scores: torch.Tensor,
    alpha: torch.Tensor,
    maximum_scale: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pair assignments into additive representative scales.

    For a dropped source i assigned to kept j, pairwise intervention defines
    ``c_i + c_j -> alpha_ij c_j``. Multiple assignments are composed as
    ``scale_j = 1 + sum_i(alpha_ij - 1)``. This is a testable approximation,
    not an identity under downstream nonlinearities.
    """
    batch, heads, _ = scores.shape
    if keep_masks.shape != (batch, heads):
        raise ValueError("keep_masks must have shape [B,H]")
    scales = keep_masks.to(dtype=scores.dtype)
    representatives = torch.full(
        (batch, heads), -1, dtype=torch.long, device=scores.device
    )
    for sample in range(batch):
        kept = torch.where(keep_masks[sample])[0]
        for source in torch.where(~keep_masks[sample])[0]:
            representative = kept[scores[sample, source, kept].argmax()]
            representatives[sample, source] = representative
            scales[sample, representative] += alpha[
                sample, source, representative
            ] - 1.0
    return scales.clamp(0, maximum_scale), representatives


class DynamicCFSRouter(nn.Module):
    """Pairwise CFS scorer conditioned only on a pre-head residual summary."""

    def __init__(
        self,
        global_dim: int,
        heads: int,
        hidden_dim: int = 256,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_embedding = nn.Embedding(heads, embedding_dim)
        pair_dim = hidden_dim + 4 * embedding_dim
        self.pair_network = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.score_output = nn.Linear(hidden_dim // 2, 1)
        self.alpha_output = nn.Linear(hidden_dim // 2, 1)
        self.importance_output = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, global_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(global_feature)
        encoded = self.global_encoder(global_feature)
        identity = self.head_embedding.weight
        source = identity[None, :, None, :].expand(batch, -1, self.heads, -1)
        candidate = identity[None, None, :, :].expand(batch, self.heads, -1, -1)
        condition = encoded[:, None, None, :].expand(
            -1, self.heads, self.heads, -1
        )
        relation = torch.cat(
            [source, candidate, (source - candidate).abs(), source * candidate],
            dim=-1,
        )
        pair = self.pair_network(torch.cat([condition, relation], dim=-1))
        scores = self.score_output(pair).squeeze(-1).tanh()
        alpha = 3.0 * self.alpha_output(pair).squeeze(-1).sigmoid()
        head_condition = torch.cat(
            [
                encoded[:, None, :].expand(-1, self.heads, -1),
                identity[None, :, :].expand(batch, -1, -1),
            ],
            dim=-1,
        )
        importance_logits = self.importance_output(head_condition).squeeze(-1)
        diagonal = torch.eye(
            self.heads, dtype=torch.bool, device=scores.device
        )[None]
        scores = scores.masked_fill(diagonal, -1.0)
        alpha = torch.where(diagonal, torch.ones_like(alpha), alpha)
        return scores, importance_logits, alpha
