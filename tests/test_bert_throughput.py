import torch
from src.bert_throughput import make_model,mask_tokens,sparse_mlm_loss

def test_sparse_mlm_matches_dense_loss_and_gradients():
    torch.set_num_threads(2); torch.manual_seed(42)
    model=make_model('B1',vocab=256,dropout=0).eval()
    ids=torch.randint(104,256,(2,8)); ids[:,0]=101; ids[:,-1]=102
    labels=ids.clone(); labels[:,::2]=-100
    full=model(input_ids=ids,labels=labels).loss; full.backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True); sparse,count=sparse_mlm_loss(model,ids,labels); sparse.backward()
    assert count==8
    torch.testing.assert_close(full,sparse,rtol=1e-6,atol=1e-6)
    for n,p in model.named_parameters():
        if p.grad is not None: torch.testing.assert_close(grads[n],p.grad,rtol=1e-4,atol=2e-6)

def test_dynamic_masking_excludes_special_tokens():
    torch.manual_seed(42); ids=torch.randint(104,30522,(1000,128)); ids[:,0]=101; ids[:,-1]=102
    changed,labels=mask_tokens(ids); chosen=labels!=-100
    assert not chosen[:,[0,127]].any()
    assert .145<float(chosen[:,1:-1].float().mean())<.155
    assert torch.equal(changed[~chosen],ids[~chosen])
    assert torch.equal(labels[chosen],ids[chosen])
    assert .78<float((changed[chosen]==103).float().mean())<.82
