"""Additive head utility baseline; no pair labels or CFS features are accepted."""
import torch
from torch import nn


class LearnedHeadGating(nn.Module):
    def __init__(self, global_dim=2304, heads=12):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(global_dim, 304), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(304, 384),
                                     nn.GELU(), nn.Linear(384, heads))

    def forward(self, feature):
        return self.network(feature)


def topk_mask(scores, keep):
    if not 1 <= keep <= scores.shape[-1]:
        raise ValueError("Invalid keep count")
    # Stable index tie-breaking also makes all-zero smoke fixtures reproducible.
    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :keep]
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)


def additive_utilities(scores, masks):
    return (scores[:, None, :] * masks.to(scores.dtype)).sum(-1)


def joint_ranking_loss(utilities, raw_kl):
    """Original R2 target normalization/temperatures, within each input and K."""
    standardized = (raw_kl - raw_kl.mean(-1, keepdim=True)) / raw_kl.std(
        -1, keepdim=True, unbiased=False).clamp_min(1e-6)
    target = (-standardized / 0.5).softmax(-1)
    return -(target * (utilities / 0.05).log_softmax(-1)).sum(-1).mean()


def assert_disjoint(*path_sets):
    normalized = [{str(p).replace('\\', '/').split('imagenette2-160/')[-1]
                   for p in paths} for paths in path_sets]
    for i, left in enumerate(normalized):
        for right in normalized[i + 1:]:
            if left & right:
                raise ValueError("Overlapping input splits")
