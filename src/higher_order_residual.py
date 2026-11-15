"""Context-conditioned latent pair residuals; not identified causal interactions."""
import numpy as np
import torch
from torch import nn


def split_masks(masks, seed=42):
    rng = np.random.default_rng(seed)
    fit = np.empty((*masks.shape[:2], 36), dtype=np.int64)
    held = np.empty((*masks.shape[:2], 12), dtype=np.int64)
    for n in range(len(masks)):
        for k in range(3):
            if len(np.unique(masks[n, k], axis=0)) != 48:
                raise ValueError('Duplicate masks must not cross fit/held-out boundaries')
            order = rng.permutation(48)
            fit[n, k], held[n, k] = order[:36], order[36:]
    return fit, held


def monotone_calibration(utility, kl):
    x = -np.asarray(utility, dtype=np.float64).reshape(-1)
    y = np.asarray(kl, dtype=np.float64).reshape(-1)
    b = max(float(np.mean((x-x.mean())*(y-y.mean())) / max(x.var(), 1e-16)), 1e-8)
    return float(y.mean()-b*x.mean()), b, max(float(y.std()), 1e-6)


def permute_graph(scores, permutations):
    b = torch.arange(len(scores), device=scores.device)[:, None, None]
    return scores[b, permutations[:, :, None], permutations[:, None, :]]


class SetResidual(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ('plain', 'cfs', 'shuffled'):
            raise ValueError(kind)
        self.kind = kind
        self.context = nn.Sequential(nn.Linear(256, 32), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(103, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer('pairs', torch.combinations(torch.arange(12), r=2))

    def forward(self, encoded, scores, weights, masks, budget):
        # masks [B,C,12]; score remains directed, with original clipping for coverage.
        B, C, H = masks.shape
        m = masks.float()
        c = self.context(encoded)[:, None].expand(-1, C, -1)
        w = weights[:, None].expand(-1, C, -1)
        k = torch.nn.functional.one_hot(torch.full((B, C), budget, device=m.device), 3).float()
        common = torch.cat((c, m, w, k), -1)  # 59
        if self.kind == 'plain':
            identity = torch.cat((m, m*w), -1)  # 24
            kept = m*w
            deleted = (1-m)*w
            stats = torch.stack((kept.sum(-1), deleted.sum(-1), kept.max(-1).values,
                deleted.max(-1).values, kept.square().sum(-1), deleted.square().sum(-1),
                w.max(-1).values, w.square().sum(-1)), -1)
            # Twelve common importance slots match the pair feature interface.
            extra = torch.cat((identity, stats, w), -1)
            return self.mlp(torch.cat((common, extra), -1)).squeeze(-1)
        i, j = self.pairs.unbind(-1)
        bounded = scores.clamp(0, 1)
        candidates = bounded[:, None].expand(-1, C, -1, -1).masked_fill(~masks[:, :, None], -torch.inf)
        best, representative = candidates.max(-1)
        si = bounded[:, None, i, :].expand(-1, C, -1, -1)
        sj = bounded[:, None, j, :].expand(-1, C, -1, -1)
        both = (si*sj*m[:, :, None]).sum(-1)/m.sum(-1, keepdim=True)
        shared = torch.minimum(si, sj).masked_fill(~masks[:, :, None], -torch.inf).max(-1).values
        cross_i = si.gather(-1, representative[:, :, j, None]).squeeze(-1)
        cross_j = sj.gather(-1, representative[:, :, i, None]).squeeze(-1)
        edges = torch.stack((scores[:, i, j][:, None].expand(-1, C, -1),
            scores[:, j, i][:, None].expand(-1, C, -1), best[:, :, i], best[:, :, j],
            both, shared, cross_i, cross_j), -1)
        identity = torch.cat((torch.eye(H, device=m.device)[i], torch.eye(H, device=m.device)[j]), -1)
        identity = identity[None, None].expand(B, C, -1, -1)
        # Preserve full importance context in both parameter-matched variants.
        extra = torch.cat((identity, edges, w[:, :, None].expand(-1, -1, len(i), -1)), -1)
        raw = self.mlp(torch.cat((common[:, :, None].expand(-1, -1, len(i), -1), extra), -1)).squeeze(-1)
        dropped_pair = (~masks[:, :, i]) & (~masks[:, :, j])
        count = dropped_pair.sum(-1)
        return (raw*dropped_pair).sum(-1)/count.clamp_min(1)
