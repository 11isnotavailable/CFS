import torch

from src.sparse_head_attention import (
    PackedStaticHeadAttention,
    SelectedHeadAttention,
    StaticHeadSelector,
)


def _attention():
    from timm.models.vision_transformer import Attention

    module = Attention(dim=64, num_heads=4, qkv_bias=True).eval()
    module.fused_attn = False
    return module


def test_all_heads_matches_dense():
    torch.manual_seed(7)
    attention = _attention()
    x = torch.randn(3, 9, 64)
    expected = attention(x)
    selector = StaticHeadSelector(torch.ones(4, dtype=torch.bool))
    actual = SelectedHeadAttention(attention, selector)(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_selected_heads_match_dense_zero_mask():
    torch.manual_seed(11)
    attention = _attention()
    x = torch.randn(2, 7, 64)
    mask = torch.tensor([True, False, True, False])
    sparse = SelectedHeadAttention(attention, StaticHeadSelector(mask))(x)

    batch, tokens, _ = x.shape
    qkv = attention.qkv(x).reshape(batch, tokens, 3, 4, 16).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * attention.scale
    weights = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    heads = (weights @ v).transpose(1, 2)
    heads[:, :, ~mask] = 0
    dense_masked = attention.proj(heads.reshape(batch, tokens, 64))
    torch.testing.assert_close(sparse, dense_masked, rtol=1e-5, atol=1e-6)


def test_per_sample_masks_are_grouped_correctly():
    torch.manual_seed(13)
    attention = _attention()
    x = torch.randn(3, 5, 64)
    masks = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.bool
    )

    class Selector(torch.nn.Module):
        def forward(self, hidden):
            return masks.to(hidden.device)

    module = SelectedHeadAttention(attention, Selector())
    together = module(x)
    assert module.last_unique_subsets == 2
    separate = []
    for index in range(3):
        one = SelectedHeadAttention(attention, StaticHeadSelector(masks[index]))
        separate.append(one(x[index:index + 1]))
    torch.testing.assert_close(together, torch.cat(separate), rtol=1e-5, atol=1e-6)


def test_packed_static_matches_selected_execution():
    torch.manual_seed(17)
    attention = _attention()
    x = torch.randn(2, 7, 64)
    mask = torch.tensor([False, True, True, False])
    selected = SelectedHeadAttention(attention, StaticHeadSelector(mask))(x)
    packed = PackedStaticHeadAttention(attention, mask)(x)
    torch.testing.assert_close(packed, selected, rtol=1e-5, atol=1e-6)
