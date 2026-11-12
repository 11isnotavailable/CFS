import torch

from src.dynamic_cfs_router import DynamicCFSRouter
from src.runtime_cfs_router import RuntimeCFSRouter, prehead_summary


def test_prehead_summary_shape_and_values():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [5.0, 9.0]]])
    result = prehead_summary(hidden)
    expected = torch.tensor([[1.0, 2.0, 4.0, 7.0, 1.0, 4.0]])
    assert torch.allclose(result, expected)


def test_runtime_router_obeys_exact_budget():
    scorer = DynamicCFSRouter(global_dim=12, heads=4, hidden_dim=16, embedding_dim=4)
    router = RuntimeCFSRouter(scorer, torch.zeros(12), torch.ones(12), keep=2)
    hidden = torch.randn(3, 5, 4)
    mask = router(hidden)
    assert mask.shape == (3, 4)
    assert torch.equal(mask.sum(dim=-1), torch.full((3,), 2.0))
    assert router.last_mask is not None
