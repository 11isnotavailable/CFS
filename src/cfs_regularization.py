"""Original pairwise auxiliary definitions, factored without subset auxiliary."""
import torch
import torch.nn.functional as F


def auxiliary_losses(scores,importance,alpha,target_scores,drops,target_alpha):
    heads=scores.shape[-1]
    valid=(~torch.eye(heads,dtype=torch.bool,device=scores.device))[None].expand_as(scores)
    target=(target_scores/.1).masked_fill(~valid,-torch.inf).softmax(-1)
    logq=scores.masked_fill(~valid,-torch.inf).log_softmax(-1)
    listwise=-torch.where(valid,target*logq,torch.zeros_like(target)).sum()/(len(scores)*heads)
    regression=F.smooth_l1_loss(scores[valid],target_scores[valid])
    deletion_target=drops.clamp_min(0)
    deletion_target=deletion_target/deletion_target.sum(-1,keepdim=True).clamp_min(1e-12)
    deletion=-(deletion_target*importance.log_softmax(-1)).sum(-1).mean()
    alpha_loss=(target.masked_fill(~valid,0)*F.smooth_l1_loss(alpha,target_alpha,reduction='none')).sum()/(len(scores)*heads)
    return dict(cfs=listwise+.5*regression,listwise=listwise,score=regression,deletion=deletion,alpha=alpha_loss)


def gradient_diagnostics(losses,parameters):
    vectors={}
    for name,loss in losses.items():
        grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        vectors[name]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,grads)]).detach()
    norms={name:float(v.norm()) for name,v in vectors.items()}
    rows=[]
    for a,va in vectors.items():
        for b,vb in vectors.items():
            if a>=b:continue
            denom=va.norm()*vb.norm()
            cosine=float((va@vb)/denom) if float(denom)>0 else float('nan')
            rows.append(dict(left=a,right=b,cosine=cosine,left_norm=norms[a],right_norm=norms[b]))
    return norms,rows
