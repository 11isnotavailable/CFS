import torch
from src.cfs_regularization import auxiliary_losses,gradient_diagnostics
from scripts.train_dynamic_cfs_router import loss_terms
from src.dynamic_cfs_router import DynamicCFSRouter,subset_masks


def test_auxiliary_terms_match_original():
    torch.manual_seed(42);model=DynamicCFSRouter(12,4,hidden_dim=16,embedding_dim=4)
    s,imp,a=model(torch.randn(2,12));target=torch.rand_like(s)*2-1
    drops=torch.rand(2,4);ta=torch.rand_like(a)*3
    old=loss_terms(s,imp,a,target,drops,ta,{2:subset_masks(4,2)},.1,.05)
    new=auxiliary_losses(s,imp,a,target,drops,ta)
    for x,y in [('listwise','listwise'),('score','score'),('deletion','importance'),('alpha','alpha')]:
        torch.testing.assert_close(new[x],old[y])
    torch.testing.assert_close(new['cfs'],old['listwise']+.5*old['score'])
    new['cfs'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_gradient_cosine_known_opposition():
    x=torch.tensor([1.,2.],requires_grad=True)
    norms,rows=gradient_diagnostics({'a':x.sum(),'b':-x.sum()},[x])
    assert abs(rows[0]['cosine']+1)<1e-6 and norms['a']>0
