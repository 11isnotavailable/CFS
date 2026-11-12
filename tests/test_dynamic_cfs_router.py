import torch

from src.dynamic_cfs_router import (
    candidate_subset_utilities,
    DynamicCFSRouter,
    compose_scales,
    realized_utilities,
    selected_masks,
    subset_masks,
    subset_utilities,
)


def test_subset_utility_prefers_shared_representative():
    scores = torch.zeros(1, 3, 3)
    scores[0, 0, 2] = 0.9
    scores[0, 1, 2] = 0.8
    importance = torch.ones(1, 3)
    masks = subset_masks(3, 1)
    selected, utility = selected_masks(scores, importance, masks)
    assert selected.tolist() == [[False, False, True]]
    assert torch.allclose(utility, torch.tensor([(0.9 + 0.8 + 1.0) / 3]))


def test_compose_scales_adds_pairwise_compensation():
    keep = torch.tensor([[False, False, True]])
    scores = torch.zeros(1, 3, 3)
    scores[0, 0, 2] = 0.9
    scores[0, 1, 2] = 0.8
    alpha = torch.ones(1, 3, 3)
    alpha[0, 0, 2] = 1.4
    alpha[0, 1, 2] = 1.2
    scales, representatives = compose_scales(keep, scores, alpha)
    assert torch.allclose(scales, torch.tensor([[0.0, 0.0, 1.6]]))
    assert representatives.tolist() == [[2, 2, -1]]


def test_router_shapes_and_diagonal_mask():
    router = DynamicCFSRouter(global_dim=12, heads=4, hidden_dim=32)
    scores, importance, alpha = router(torch.randn(3, 12))
    assert scores.shape == (3, 4, 4)
    assert importance.shape == (3, 4)
    assert alpha.shape == (3, 4, 4)
    diagonal = torch.arange(4)
    assert torch.all(scores[:, diagonal, diagonal] == -1)
    assert torch.all(alpha[:, diagonal, diagonal] == 1)


def test_subset_utility_is_bounded():
    scores = torch.full((2, 4, 4), 4.0)
    importance = torch.rand(2, 4)
    utility = subset_utilities(scores, importance, subset_masks(4, 2))
    assert torch.all((utility >= 0) & (utility <= 1))


def test_realized_utility_supports_sample_specific_masks():
    scores = torch.zeros(2, 3, 3)
    scores[0, 0, 2] = 0.8
    scores[1, 2, 0] = 0.6
    importance = torch.ones(2, 3)
    keep = torch.tensor([[False, True, True], [True, True, False]])
    utility = realized_utilities(scores, importance, keep)
    assert torch.allclose(utility, torch.tensor([(0.8 + 2) / 3, (0.6 + 2) / 3]))


def test_candidate_subset_utilities_matches_single_mask_evaluation():
    scores = torch.rand(2, 4, 4)
    importance = torch.rand(2, 4)
    base = subset_masks(4, 2)
    masks = torch.stack([base[:3], base[2:5]], dim=0)
    result = candidate_subset_utilities(scores, importance, masks)
    expected = torch.stack(
        [
            realized_utilities(scores, importance, masks[:, index])
            for index in range(masks.shape[1])
        ],
        dim=1,
    )
    assert torch.allclose(result, expected)
