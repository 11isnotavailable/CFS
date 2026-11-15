import pytest
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from src.gpt2_routing import GPT2Suffix,prefix_summary,output_kl
from src.gpt2_routing import GPT2RuntimeRouter
from src.dynamic_cfs_router import DynamicCFSRouter
from src.learned_head_gating import LearnedHeadGating
import numpy as np


@pytest.mark.parametrize('layer',[1,2])
def test_gpt2_suffix_projection_bias_and_causal_feature(layer):
    torch.manual_seed(42)
    config=GPT2Config(n_layer=3,n_head=2,n_embd=16,vocab_size=37,n_positions=24,attn_pdrop=0,resid_pdrop=0,embd_pdrop=0,bos_token_id=0,eos_token_id=0)
    config._attn_implementation='eager'
    model=GPT2LMHeadModel(config).eval().requires_grad_(False)
    with torch.no_grad():model.transformer.h[layer].attn.c_proj.bias.fill_(.23)
    ids=torch.randint(0,37,(1,8));engine=GPT2Suffix(model,layer)
    reference,features=engine.capture(ids)
    with torch.no_grad():
        torch.testing.assert_close(engine.suffix(torch.ones(1,2)),reference,atol=2e-6,rtol=1e-5)
        contributions=engine.contributions().sum(0)+engine.block.attn.c_proj.bias
        torch.testing.assert_close(contributions,engine.block.attn.c_proj(engine.cache['heads'])[0],atol=1e-6,rtol=1e-5)
        for scales in (torch.tensor([[0.,1.]]),torch.tensor([[0.,0.]]),torch.tensor([[0.,2.]])):
            fast=engine.suffix(scales);engine.gate.set_scales(scales)
            direct=model(ids,use_cache=False).logits[:,-1]
            torch.testing.assert_close(fast,direct,atol=2e-6,rtol=1e-5)
        engine.gate.set_scales(None)
        full=torch.cat((ids,torch.randint(0,37,(1,3))),1)
        # Dense prefix logits cannot depend on subsequently appended tokens.
        torch.testing.assert_close(model(full,use_cache=False).logits[:,7],reference,atol=2e-6,rtol=1e-5)
        h=engine.cache['prehead'];torch.testing.assert_close(features,prefix_summary(h))
    assert torch.isfinite(engine.taylor(int(ids[0,-1]))).all()
    with pytest.raises(ValueError):engine.capture(ids,torch.zeros_like(ids))
    engine.close();assert all(p.grad is None for p in model.parameters())


def test_lm_kl_direction():
    p=torch.tensor([[3.,1.,0.]]);q=torch.tensor([[0.,2.,1.]])
    expected=(p.softmax(-1)*(p.log_softmax(-1)-q.log_softmax(-1))).sum(-1)
    torch.testing.assert_close(output_kl(p,q),expected)
    assert not torch.allclose(output_kl(p,q),output_kl(q,p))


@pytest.mark.parametrize('kind',['R2','Gating'])
def test_online_prefix_routing_matches_cached_intervention(kind):
    torch.manual_seed(7)
    config=GPT2Config(n_layer=1,n_head=12,n_embd=768,vocab_size=37,n_positions=16,
        attn_pdrop=0,resid_pdrop=0,embd_pdrop=0,bos_token_id=0,eos_token_id=0)
    config._attn_implementation='eager'
    model=GPT2LMHeadModel(config).eval().requires_grad_(False)
    scorer=LearnedHeadGating() if kind=='Gating' else DynamicCFSRouter(2304,12)
    payload=dict(state_dict=scorer.state_dict(),global_mean=np.zeros(2304,np.float32),global_std=np.ones(2304,np.float32))
    ids=torch.randint(0,37,(1,8));engine=GPT2Suffix(model,0);engine.capture(ids)
    live=GPT2RuntimeRouter(model,0,payload,kind,3)
    with torch.inference_mode():
        online=live.next_logits(ids);mask=live.last_mask.clone()
        assert int(mask.sum())==3
        live.close();cached=engine.suffix(mask.float())
        torch.testing.assert_close(online,cached,atol=2e-5,rtol=1e-5)
    engine.close()
