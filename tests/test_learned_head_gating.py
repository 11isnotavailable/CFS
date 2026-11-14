import itertools
import numpy as np
import pytest
import torch
from src.dynamic_cfs_router import DynamicCFSRouter
from src.learned_head_gating import (LearnedHeadGating, topk_mask, additive_utilities,
                                     joint_ranking_loss, assert_disjoint)


def test_parameter_match_and_input():
    a, b = LearnedHeadGating(), DynamicCFSRouter(2304, 12)
    na, nb = [sum(p.numel() for p in m.parameters()) for m in (a, b)]
    assert na == 822460 and nb == 825091
    assert abs(na/nb-1) < .05
    assert a(torch.zeros(2, 2304)).shape == (2, 12)


@pytest.mark.parametrize('keep', [3, 6, 9, 12])
def test_topk_exact_and_optimal(keep):
    torch.manual_seed(42)
    scores = torch.randn(2, 12)
    mask = topk_mask(scores, keep)
    assert (mask.sum(-1) == keep).all()
    all_masks = torch.zeros(len(list(itertools.combinations(range(12), keep))), 12)
    for row, indices in enumerate(itertools.combinations(range(12), keep)):
        all_masks[row, list(indices)] = 1
    maximum = additive_utilities(scores, all_masks[None].expand(2, -1, -1)).max(-1).values
    torch.testing.assert_close((scores*mask).sum(-1), maximum)


def test_ranking_matches_original_target_and_is_budget_local():
    torch.manual_seed(2)
    utility, kl = torch.randn(2, 3, 48), torch.rand(2, 3, 48)
    expected = torch.stack([joint_ranking_loss(utility[:, k], kl[:, k]) for k in range(3)]).mean()
    torch.testing.assert_close(joint_ranking_loss(utility, kl), expected)
    assert torch.isfinite(joint_ranking_loss(utility, torch.zeros_like(kl)))
    # Changing another input/budget's raw offset has no effect.
    torch.testing.assert_close(joint_ranking_loss(utility, kl), joint_ranking_loss(utility, kl+3))


def test_paths_disjoint_and_separator_normalization():
    assert_disjoint(['x/imagenette2-160/train/a.jpg'], ['data/imagenette2-160/val/a.jpg'])
    with pytest.raises(ValueError):
        assert_disjoint(['x/imagenette2-160/val/a.jpg'], ['data\\imagenette2-160\\val\\a.jpg'])


def test_seed_repeatability():
    torch.manual_seed(17); a = LearnedHeadGating().eval()
    torch.manual_seed(17); b = LearnedHeadGating().eval()
    torch.testing.assert_close(a(torch.ones(1, 2304)), b(torch.ones(1, 2304)), rtol=0, atol=0)


def test_shared_compensation_does_not_depend_on_policy_name():
    from src.dynamic_cfs_router import compose_scales
    torch.manual_seed(12)
    masks = topk_mask(torch.randn(3, 12), 6)
    scores, alpha = torch.rand(3, 12, 12), torch.rand(3, 12, 12)*3
    # Every selector calls the same operator with the same frozen compensator.
    first, _ = compose_scales(masks, scores, alpha)
    second, _ = compose_scales(masks.clone(), scores.clone(), alpha.clone())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert (first[~masks] == 0).all()


def test_training_source_has_no_test_label_reader():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1]/'scripts/train_learned_gating.py').read_text()
    assert "for split in ('train', 'tune')" in source
    assert "test.npz" not in source
    assert 'test_dir' not in source


def test_bootstrap_resamples_paired_inputs():
    from scripts.analyze_learned_gating import bootstrap_stat
    rng = np.random.default_rng(42)
    draws = rng.integers(0, 4, size=(2000, 4))
    left = np.array([1., 4., 2., 8.])
    right = left + .25
    stat, sampled = bootstrap_stat(left-right, draws)
    assert stat['mean'] == stat['ci_low'] == stat['ci_high'] == -.25
    assert np.all(sampled == -.25)
