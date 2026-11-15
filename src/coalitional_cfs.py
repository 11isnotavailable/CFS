"""Bounded one-source pair/coalition substitution; original CFS is unchanged."""
import itertools,time
import numpy as np
import torch
from torch import nn
from scripts.evaluate_dynamic_cfs_execution import dense_kl
from src.head_intervention import make_intervenable,clear_interventions
from src.runtime_cfs_router import prehead_summary


def candidate_pool(seed=42,heads=12,width=4):
    rng=np.random.default_rng(seed)
    return np.stack([np.sort(rng.choice([j for j in range(heads) if j!=i],width,replace=False)) for i in range(heads)])


def relations(pool,joint=False):
    return np.array([(i,*js) for i,row in enumerate(pool) for js in (itertools.combinations(row,2) if joint else [(j,) for j in row])],dtype=np.int64)


def scales_for(rel,alpha,heads=12):
    scales=torch.ones(len(rel),heads,device=alpha.device,dtype=alpha.dtype)
    scales.scatter_(1,rel[:,:1],0.)
    return scales.scatter(1,rel[:,1:],alpha)


class LastViTSuffix:
    def __init__(self,model):
        self.model=model;self.wrappers=make_intervenable(model);self.block=model.blocks[-1];self.wrapper=self.wrappers[-1]
        self.cache={};self.capturing=False
        self.handles=[self.block.register_forward_pre_hook(self._residual),self.wrapper.attention.proj.register_forward_pre_hook(self._heads)]

    def _residual(self,module,args):
        if self.capturing:self.cache['residual']=args[0][:,:1].detach()

    def _heads(self,module,args):
        if self.capturing:self.cache['heads']=args[0][:,:1].detach()

    def capture(self,image):
        clear_interventions(self.wrappers);self.wrapper.capture_input=True;self.capturing=True
        with torch.no_grad():reference=self.model(image)
        self.capturing=False;self.cache['reference']=reference.detach()
        feature=prehead_summary(self.wrapper.last_input);self.wrapper.capture_input=False
        return reference,feature

    def suffix(self,scales):
        self.wrapper.set_head_scales(None)
        b=len(scales);heads=self.cache['heads'].expand(b,-1,-1).reshape(b,1,12,64)
        z=(heads*scales[:,None,:,None]).reshape(b,1,768)
        a=self.wrapper.attention.proj_drop(self.wrapper.attention.proj(z))
        h=self.cache['residual'].expand(b,-1,-1)+self.block.drop_path1(self.block.ls1(a))
        h=h+self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(h))))
        return self.model.forward_head(self.model.norm(h))

    def losses(self,scales,batch=1024):
        with torch.no_grad():return torch.cat([dense_kl(self.cache['reference'],self.suffix(s)) for s in scales.split(batch)])

    def close(self):
        for handle in self.handles:handle.remove()


def grid_oracle(engine,rel,steps=31,batch=1024):
    device=engine.cache['heads'].device;rel=torch.as_tensor(rel,device=device)
    grid=torch.linspace(0,3,steps,device=device)
    alpha=grid[:,None] if rel.shape[1]==2 else torch.cartesian_prod(grid,grid)
    count=len(alpha);local=rel[:,None].expand(-1,count,-1).reshape(-1,rel.shape[1])
    values=engine.losses(scales_for(local,alpha[None].expand(len(rel),-1,-1).reshape(-1,rel.shape[1]-1)),batch).reshape(len(rel),count)
    loss,index=values.min(-1)
    return loss,alpha[index],dict(variants=len(rel)*count,forward_calls=(len(rel)*count+batch-1)//batch,backward_calls=0)


def gradient_oracle(engine,rel,initial,steps=60,lr=.08):
    rel=torch.as_tensor(rel,device=initial.device);starts=initial.shape[1];dims=rel.shape[1]-1;local=rel[:,None].expand(-1,starts,-1).reshape(-1,dims+1)
    alpha=initial.reshape(-1,dims).clone().requires_grad_(True);optimizer=torch.optim.Adam([alpha],lr=lr)
    best=torch.full((len(alpha),),float('inf'),device=alpha.device);best_alpha=alpha.detach().clone()
    for _ in range(steps+1):
        values=dense_kl(engine.cache['reference'],engine.suffix(scales_for(local,alpha)))
        better=values.detach()<best;best=torch.minimum(best,values.detach());best_alpha[better]=alpha.detach()[better]
        if _==steps:break
        optimizer.zero_grad(set_to_none=True);values.sum().backward();optimizer.step()
        with torch.no_grad():alpha.clamp_(0,3)
    best=best.reshape(len(rel),starts);value,index=best.min(-1)
    return value,best_alpha.reshape(len(rel),starts,dims)[torch.arange(len(rel),device=alpha.device),index],dict(variants=len(local)*(steps+1),forward_calls=steps+1,backward_calls=steps,starts=starts)


class PairInformationPredictor(nn.Module):
    """Privileged information-content control, NOT a forward-only predictor."""
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,x):return self.network(torch.stack((x.mean(-1),(x[...,0]-x[...,1]).abs()),-1)).squeeze(-1)


class RelationPredictor(nn.Module):
    """PairScorePredictor level0 adapted to symmetric sets, equal parameter count.

    Candidate mean/absolute difference preserve singleton identity and unordered
    pair diversity. Both tasks use the same embedding/scorer interface.
    """
    def __init__(self,hidden=256,embedding=16):
        super().__init__();self.global_encoder=nn.Sequential(nn.Linear(2304,hidden),nn.GELU())
        self.source_embedding=nn.Embedding(12,embedding);self.candidate_embedding=nn.Embedding(12,embedding)
        self.scorer=nn.Sequential(nn.Linear(hidden+3*embedding,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,1))

    def forward(self,features,rel):
        # [B,C,3], singleton represented by repeated candidate identity.
        encoded=self.global_encoder(features)[:,None].expand(-1,rel.shape[1],-1)
        a=self.candidate_embedding(rel[...,1]);b=self.candidate_embedding(rel[...,2])
        return self.scorer(torch.cat((encoded,self.source_embedding(rel[...,0]),(a+b)/2,(a-b).abs()),-1)).squeeze(-1)
