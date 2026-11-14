import numpy as np
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from src.mlp_group_intervention import neuron_groups,neuron_scales,capture_mlp_inputs,CachedMLP
from scripts.run_experiment0 import build_substitution_masks,output_kl


def test_partition_reproducibility_and_coverage():
    for seed in [None,42,123,2027]:
        groups=neuron_groups(32,8,seed)
        np.testing.assert_array_equal(np.sort(groups.flatten()),np.arange(32))
        np.testing.assert_array_equal(groups,neuron_groups(32,8,seed))
    with pytest.raises(ValueError): neuron_groups(30,8)


@pytest.mark.parametrize('seed',[None,42])
def test_cached_mlp_bias_gates_and_alpha(seed):
    torch.manual_seed(19)
    model=VisionTransformer(img_size=16,patch_size=8,embed_dim=16,depth=3,num_heads=2,mlp_ratio=2,num_classes=5).eval().requires_grad_(False)
    with torch.no_grad():
        for block in model.blocks:
            block.mlp.fc2.bias.fill_(.17)
            block.mlp.fc1.bias.copy_(torch.linspace(-.3,.3,32))
    image=torch.randn(1,3,16,16)
    reference,caches=capture_mlp_inputs(model,image,[1,2])
    groups=neuron_groups(32,8,seed)
    for layer in [1,2]:
        cache=CachedMLP(model,layer,caches[layer],groups)
        c=cache.contributions(); fc=cache.mlp.fc2
        torch.testing.assert_close(c.sum(1)+fc.bias,fc(caches[layer]['activation']))
        scales=torch.ones(3,8); scales[1:,0]=0; scales[2,1]=0
        neuron=neuron_scales(groups,scales,32)
        assert (neuron[1,groups[1:]]==1).all()
        assert (neuron[1,groups[0]]==0).all()
        activation=caches[layer]['activation']
        torch.testing.assert_close(fc(activation*neuron[1,None,:]),fc(activation)-c[:,0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(fc(activation*neuron[2,None,:]),fc(activation)-c[:,0]-c[:,1],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(cache.logits(scales),cache.direct_logits(image,scales),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(cache.logits(scales[:1]),reference)
        masks,meta=build_substitution_masks(8,torch.tensor([0.,1.,2.]))
        assert torch.equal(masks[1],scales[1])  # alpha=1 is source-only drop
        assert torch.equal(masks[0],scales[2])  # alpha=0 drops source AND target
        assert torch.all(output_kl(reference,reference,1)==0)
        gates=torch.ones(1,8,requires_grad=True)
        grad=torch.autograd.grad(cache.logits(gates).sum(),gates)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_source_conditioned_importance_mapping():
    from scripts.analyze_mlp_group_cfs import input_metrics
    scores=np.array([[np.nan,.8,.2,.3],[.1,np.nan,.1,.2],[.4,.9,np.nan,.1],[.2,.7,.6,np.nan]])
    data=dict(cfs=scores,single_drop=np.array([1.,0.,.2,.3]),valid=np.array([True,False,True,True]),
              substitution=(1-scores)*np.array([1.,0.,.2,.3])[:,None],
              taylor=np.array([1.,9.,2.,3.]),contribution_cosine=np.eye(4),no_op_kl=0.)
    metrics,sources=input_metrics(data)
    assert metrics['raw_oracle']==pytest.approx((2.4,3))
    assert metrics['taylor_recovery']==pytest.approx((2.4,3))
    # Invalid sources remain eligible representatives, as in attention analysis.
    assert sources[0]['taylor_rep']==1
    assert metrics['random_recovery'][1]==9
    assert metrics['directed_asymmetry_abs']==pytest.approx((.8,3))
    assert metrics['valid_source_fraction']==pytest.approx((3.,4))


def test_input_conditioning_statistics_preserve_group_identity():
    from scripts.analyze_mlp_group_cfs import conditional_statistics
    base=np.array([[np.nan,.8,.2],[.4,np.nan,.1],[.2,.7,np.nan]])
    scores=np.repeat(base[None],4,axis=0); valid=np.ones((4,3),bool)
    result=conditional_statistics(scores,valid,np.arange(4)[None])[0]
    np.testing.assert_allclose(result,[0,0],atol=2e-8)
    scores[2:,0,2]=.9
    result=conditional_statistics(scores,valid,np.arange(4)[None])[0]
    assert result[0]>0 and result[1]>0
    # Resampling a whole input retains its entire matrix, not independent pairs.
    repeated=conditional_statistics(scores,valid,np.array([[2,2,3,3]]))[0]
    np.testing.assert_allclose(repeated,[0,0],atol=2e-8)
    # Enumerate distinct ORIGINAL image pairs with bootstrap multiplicities.
    selected=np.array([0,0,1,2]); best=np.nanargmax(scores,axis=-1)
    expected=np.mean([np.mean([best[i,g]!=best[j,g] for i in selected for j in selected if i!=j]) for g in range(3)])
    actual=conditional_statistics(scores,valid,selected[None])[0,1]
    assert actual==pytest.approx(expected)
