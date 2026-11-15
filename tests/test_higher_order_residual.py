import numpy as np
import torch
from src.dynamic_cfs_router import subset_masks
from src.higher_order_residual import SetResidual, monotone_calibration, permute_graph, split_masks


def test_positive_calibration_preserves_ranking():
    u = np.linspace(.1, .9, 80)
    a, b, scale = monotone_calibration(u, 2-1.5*u)
    assert b > 0 and scale > 0
    np.testing.assert_allclose(a-b*u, 2-1.5*u, atol=1e-12)
    assert np.argmin(a-b*u) == np.argmax(u)


def test_split_and_duplicate_guard():
    masks = np.stack([subset_masks(12, k)[:48].numpy() for k in (3, 6, 9)])[None]
    a, b = split_masks(masks)
    for k in range(3):
        assert not set(a[0,k]) & set(b[0,k])
        assert len(set(a[0,k]) | set(b[0,k])) == 48
    masks[0,0,1] = masks[0,0,0]
    import pytest
    with pytest.raises(ValueError): split_masks(masks)


def test_parameter_matching_zero_initialization_and_context():
    models = [SetResidual(k) for k in ('plain','cfs','shuffled')]
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
    encoded = torch.randn(2,256); scores = torch.rand(2,12,12); weights = torch.rand(2,12).softmax(-1)
    masks = subset_masks(12,9)[:4][None].expand(2,-1,-1)
    for model in models:
        torch.testing.assert_close(model(encoded,scores,weights,masks,2), torch.zeros(2,4))
    model=models[1]
    torch.nn.init.normal_(model.mlp[-1].weight)
    out=model(encoded,scores,weights,masks,2)
    assert out.shape==(2,4) and torch.isfinite(out).all()
    assert not torch.equal(out[:,0],out[:,1])
    # With fewer than two deleted heads, the pair correction is exactly zero.
    full=torch.ones(2,1,12,dtype=torch.bool)
    torch.testing.assert_close(model(encoded,scores,weights,full,2),torch.zeros(2,1))


def test_shuffle_preserves_values_but_breaks_head_alignment():
    scores=torch.arange(288).reshape(2,12,12)
    perm=torch.stack((torch.arange(11,-1,-1),torch.arange(12).roll(3)))
    shuffled=permute_graph(scores,perm)
    torch.testing.assert_close(scores.flatten(1).sort().values,shuffled.flatten(1).sort().values)
    assert not torch.equal(scores,shuffled)
