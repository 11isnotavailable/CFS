import pytest
import torch

from src.model_loading import repartition_timm_attention_heads


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_dim = 24
        self.num_heads = 3
        self.head_dim = 8
        self.scale = 8 ** -0.5
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])


def test_repartition_updates_attention_geometry():
    model = Model()
    repartition_timm_attention_heads(model, 6)
    for block in model.blocks:
        assert block.attn.num_heads == 6
        assert block.attn.head_dim == 4
        assert block.attn.scale == pytest.approx(0.5)


def test_repartition_requires_divisibility():
    with pytest.raises(ValueError, match="not divisible"):
        repartition_timm_attention_heads(Model(), 5)
