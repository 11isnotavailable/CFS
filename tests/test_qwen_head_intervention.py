from types import SimpleNamespace

import torch

from src.qwen_head_intervention import QwenHeadGate


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = torch.nn.Linear(8, 8, bias=False)
        self.o_proj.weight.data.copy_(torch.eye(8))

    def forward(self, hidden):
        return self.o_proj(hidden)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=4, hidden_size=8)
        layer = SimpleNamespace(self_attn=Attention())
        self.model = SimpleNamespace(layers=[layer])


def test_qwen_gate_scales_query_head_slices_per_batch():
    model = FakeModel()
    gate = QwenHeadGate(model, 0)
    hidden = torch.ones(2, 3, 8)
    gate.set_scales(torch.tensor([[1.0, 0.0, 2.0, 3.0], [0.0, 1.0, 1.0, 1.0]]))
    output = model.model.layers[0].self_attn(hidden)
    assert torch.equal(output[0, 0], torch.tensor([1, 1, 0, 0, 2, 2, 3, 3.0]))
    assert torch.equal(output[1, 0], torch.tensor([0, 0, 1, 1, 1, 1, 1, 1.0]))
    gate.close()


def test_qwen_gate_supports_coarser_equal_width_units():
    model = FakeModel()
    gate = QwenHeadGate(model, 0, num_units=2)
    hidden = torch.ones(1, 2, 8)
    gate.set_scales(torch.tensor([0.0, 2.0]))
    output = model.model.layers[0].self_attn(hidden)
    assert torch.equal(output[0, 0], torch.tensor([0, 0, 0, 0, 2, 2, 2, 2.0]))
    gate.close()
