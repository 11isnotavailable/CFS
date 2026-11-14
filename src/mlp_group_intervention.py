"""Frozen neuron-group gates and exact prefix caching for standard timm ViTs."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def neuron_groups(width: int, count: int, seed: int | None = None) -> np.ndarray:
    if count < 2 or width % count:
        raise ValueError('Groups must partition the hidden width equally')
    indices=np.arange(width) if seed is None else np.random.default_rng(seed).permutation(width)
    return np.sort(indices.reshape(count,-1),axis=1)


def neuron_scales(groups, scales, width):
    owner=torch.empty(width,device=scales.device,dtype=torch.long)
    idx=torch.as_tensor(groups,device=scales.device)
    owner[idx]=torch.arange(len(groups),device=scales.device)[:,None]
    return scales[...,owner]


def capture_mlp_inputs(model, image, layers):
    """One unchanged dense pass captures residual and post-activation fc2 input."""
    from timm.models.vision_transformer import Block
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Use eval mode and frozen parameters')
    caches={layer:{} for layer in layers}; handles=[]
    def capture(layer,key):
        def hook(module,inputs): caches[layer][key]=inputs[0].detach().clone()
        return hook
    for layer in layers:
        block=model.blocks[layer]
        if type(block) is not Block or not isinstance(block.mlp.norm,torch.nn.Identity):
            raise ValueError('Only standard timm Block with unnormalized MLP supported')
        handles += [block.norm2.register_forward_pre_hook(capture(layer,'residual')),
                    block.mlp.fc2.register_forward_pre_hook(capture(layer,'activation'))]
    try:
        with torch.no_grad(): reference=model(image)
    finally:
        for handle in handles: handle.remove()
    return reference,caches


class CachedMLP:
    def __init__(self, model, layer, cache, groups):
        self.model=model; self.layer=layer; self.cache=cache; self.groups=groups
        self.mlp=model.blocks[layer].mlp

    def logits(self, scales):
        if scales.ndim==1: scales=scales[None]
        activation=self.cache['activation']
        expanded=neuron_scales(self.groups,scales,activation.shape[-1]).to(activation.dtype)
        # Exactly the original projection with only the chosen neuron scales
        # changed. fc2 bias is applied once, without scaling.
        out=self.mlp.drop2(self.mlp.fc2(activation*expanded[:,None,:]))
        block=self.model.blocks[self.layer]
        hidden=self.cache['residual']+block.drop_path2(block.ls2(out))
        for subsequent in self.model.blocks[self.layer+1:]: hidden=subsequent(hidden)
        return self.model.forward_head(self.model.norm(hidden))

    def contributions(self):
        activation=self.cache['activation'].float()
        weight=self.mlp.fc2.weight.float()
        return torch.stack([F.linear(activation[...,idx],weight[:,idx],None) for idx in self.groups],dim=1)

    def direct_logits(self,image,scales):
        """Independent full-backbone check of the cached suffix path."""
        def gate(module,inputs):
            value=inputs[0]
            gains=neuron_scales(self.groups,scales,value.shape[-1]).to(value.dtype)
            return (value*gains[:,None,:],)+inputs[1:]
        handle=self.mlp.fc2.register_forward_pre_hook(gate)
        try:
            with torch.no_grad(): return self.model(image.expand(len(scales),-1,-1,-1))
        finally: handle.remove()
