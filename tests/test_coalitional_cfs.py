import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer
from src.coalitional_cfs import candidate_pool,relations,scales_for,LastViTSuffix,grid_oracle,gradient_oracle,RelationPredictor
from src.coalitional_metrics import input_metrics,bootstrap_ratio


def test_pool_and_joint_boundary():
    pool=candidate_pool();assert pool.shape==(12,4)
    assert all(len(set(row))==4 and i not in row for i,row in enumerate(pool))
    assert relations(pool).shape==(48,2) and relations(pool,True).shape==(72,3)
    rel=torch.tensor([[0,1,2],[3,5,6]])
    s=scales_for(rel,torch.ones(2,2));assert s[0,0]==0 and s[1,3]==0 and s.sum()==22
    pair=scales_for(rel[:,:2],torch.tensor([[2.],[.5]]))
    joint=scales_for(rel,torch.tensor([[2.,1.],[.5,1.]]));torch.testing.assert_close(pair,joint,atol=0,rtol=0)


def test_symmetric_predictor():
    torch.manual_seed(5);model=RelationPredictor().eval();x=torch.randn(2,2304)
    rel=torch.tensor([[[0,1,2],[3,4,7]],[[1,2,3],[5,4,6]]])
    torch.testing.assert_close(model(x,rel),model(x,rel[:,:,[0,2,1]]),atol=0,rtol=0)


def test_vit_suffix_matches_direct_bias_and_gate():
    torch.manual_seed(3)
    model=VisionTransformer(img_size=16,patch_size=8,embed_dim=768,depth=1,num_heads=12,mlp_ratio=.25,num_classes=7).eval().requires_grad_(False)
    with torch.no_grad():model.blocks[0].attn.proj.bias.fill_(.2)
    engine=LastViTSuffix(model);image=torch.randn(1,3,16,16);ref,_=engine.capture(image)
    with torch.no_grad():
        torch.testing.assert_close(engine.suffix(torch.ones(1,12)),ref,atol=3e-6,rtol=1e-5)
        rel=torch.tensor([[0,1,2]])
        for alpha in (torch.ones(1,2),torch.zeros(1,2),torch.tensor([[2.3,.7]])):
            scales=scales_for(rel,alpha);fast=engine.suffix(scales);engine.wrapper.set_head_scales(scales)
            torch.testing.assert_close(model(image),fast,atol=3e-6,rtol=1e-5)
        engine.wrapper.set_head_scales(None)
        pair=grid_oracle(engine,np.array([[0,1],[0,2]]),steps=7)[0]
        joint=grid_oracle(engine,np.array([[0,1,2]]),steps=7)[0]
        assert joint.item()<=pair.min().item()+1e-6
    engine.close()


def test_ranking_metrics_and_input_clusters():
    truth=np.array([[[0.,.2,.6,1.],[.2,.4,.6,.8]],[[0.,.1,.2,.3],[0.,0.,0.,0.]]])
    valid=np.array([[True,False],[True,False]])
    m=input_metrics(truth,truth,valid,.5)
    np.testing.assert_array_equal(m['regret'],0);np.testing.assert_array_equal(m['best_hit'],1)
    np.testing.assert_array_equal(m['top2_recall'],1);np.testing.assert_allclose(m['spearman'],1)
    np.testing.assert_allclose(m['random_top2_recall'],.5);np.testing.assert_allclose(m['oracle'],[1,.3])
    flat=input_metrics(truth,np.zeros_like(truth),valid,.5)
    np.testing.assert_array_equal(flat['undefined_spearman'],1)
    draws=np.array([[0,0],[0,1],[1,0],[1,1]])
    value,ci=bootstrap_ratio(np.array([1.,3.]),np.array([2.,4.]),draws)
    assert value==2/3;np.testing.assert_allclose(ci,np.quantile([.5,2/3,2/3,.75],[.025,.975]))
