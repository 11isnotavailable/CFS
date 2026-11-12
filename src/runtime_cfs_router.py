"""Online adapter that turns a trained CFS scorer into attention head masks."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.dynamic_cfs_router import DynamicCFSRouter, subset_masks, subset_utilities


def prehead_summary(hidden: torch.Tensor) -> torch.Tensor:
    """Vectorized CLS + non-CLS mean + variance feature used by the teacher."""
    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError("hidden must have shape [B,N,C] with at least two tokens")
    value = hidden.float()
    patches = value[:, 1:]
    return torch.cat(
        [
            value[:, 0],
            patches.mean(dim=1),
            patches.var(dim=1, unbiased=False),
        ],
        dim=-1,
    )


class RuntimeCFSRouter(nn.Module):
    """Select an exact-budget head subset from the current pre-head state."""

    def __init__(
        self,
        scorer: DynamicCFSRouter,
        global_mean: torch.Tensor,
        global_std: torch.Tensor,
        keep: int,
    ) -> None:
        super().__init__()
        self.scorer = scorer.eval()
        self.keep = int(keep)
        self.register_buffer("global_mean", global_mean.float())
        self.register_buffer("global_std", global_std.float().clamp_min(1e-5))
        self.register_buffer("masks", subset_masks(scorer.heads, self.keep))
        self.last_mask: torch.Tensor | None = None
        self.last_utility: torch.Tensor | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        keep: int,
        map_location: str | torch.device = "cpu",
    ) -> "RuntimeCFSRouter":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        args = payload.get("args", {})
        scorer = DynamicCFSRouter(
            global_dim=int(payload["global_dim"]),
            heads=int(payload["heads"]),
            hidden_dim=int(payload["hidden_dim"]),
            embedding_dim=int(payload["embedding_dim"]),
            dropout=float(args.get("dropout", 0.1)),
        )
        scorer.load_state_dict(payload["state_dict"], strict=True)
        return cls(
            scorer,
            torch.as_tensor(payload["global_mean"]),
            torch.as_tensor(payload["global_std"]),
            keep,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Oracle shards store pre-head summaries as FP16. Reproduce that
        # quantization boundary online so frozen/offline and deployed decisions
        # have identical semantics, including near-tied subset scores.
        summary = prehead_summary(hidden).to(torch.float16).float()
        feature = (summary - self.global_mean) / self.global_std
        scores, importance_logits, _ = self.scorer(feature)
        # At inference H=12, so even the largest candidate set has only 924
        # subsets. Evaluate it in one tensor operation rather than eight small
        # chunks; this preserves exact selection while reducing launch overhead.
        utilities = subset_utilities(
            scores,
            importance_logits.softmax(dim=-1),
            self.masks,
            mask_chunk_size=len(self.masks),
        )
        choice = utilities.argmax(dim=-1)
        mask = self.masks[choice]
        utility = utilities.gather(1, choice[:, None]).squeeze(1)
        self.last_mask = mask.detach()
        self.last_utility = utility.detach()
        return mask.to(dtype=hidden.dtype)
