"""Version-compatible adapter for the existing BERT pre-WO head intervention."""
from types import SimpleNamespace
import torch
from src.text_head_intervention import IntervenableBertSelfAttention

class CompatibleBertGate(IntervenableBertSelfAttention):
    def forward(self,*args,**kwargs):
        output=self.inner(*args,**kwargs)
        value=output[0]
        if self.head_scales is not None:
            b,t,w=value.shape
            assert self.head_scales.shape==(b,self.num_heads)
            value=(value.reshape(b,t,self.num_heads,self.head_dim)*self.head_scales[:,None,:,None].to(value)).reshape(b,t,w)
        return (value,)+tuple(output[1:])

def wrap_last(model):
    layer=model.bert.encoder.layer[-1]
    if not isinstance(layer.attention.self,CompatibleBertGate):
        layer.attention.self=CompatibleBertGate(layer.attention.self)
    return layer.attention.self

class LastBlockOnly(torch.nn.Module):
    def __init__(self,layer,hidden):super().__init__();self.layer=layer;self.hidden=hidden
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(last_hidden_state=self.layer(self.hidden.expand(len(input_ids),-1,-1),attention_mask=None)[0])

class CachedSuffix(torch.nn.Module):
    def __init__(self,model,hidden):
        super().__init__();self.bert=LastBlockOnly(model.bert.encoder.layer[-1],hidden);self.cls=model.cls

def stream_oracle(model,wrapper,inputs,positions,reference,alphas,chunk=32):
    # Same original absolute candidate alpha: source=0, candidate=alpha;
    # alpha=1 equals deletion-only, alpha=0 deletes both heads.
    from scripts.collect_bert_mlm_scaling import variant_logits,outcome_kl
    h=wrapper.num_heads;device=next(model.parameters()).device
    assert len(reference)==1
    def errors(scales):
        result=[]
        for begin in range(0,len(scales),chunk):
            part=scales[begin:begin+chunk];idx=torch.zeros(len(part),device=device,dtype=torch.long)
            logits=variant_logits(model,wrapper,inputs,positions,idx,part,chunk)
            result.append(outcome_kl(reference.expand(len(part),-1,-1),logits))
        return torch.cat(result)
    scale=torch.ones(h,h,device=device);scale[torch.arange(h),torch.arange(h)]=0
    drops=errors(scale)
    coords=[(i,j,k) for i in range(h) for j in range(h) if i!=j for k in range(len(alphas))]
    coords=torch.tensor(coords,device=device);scale=torch.ones(len(coords),h,device=device)
    row=torch.arange(len(coords),device=device);scale[row,coords[:,0]]=0;scale[row,coords[:,1]]=alphas[coords[:,2]]
    cube=torch.full((h,h,len(alphas)),torch.nan)
    values=errors(scale).reshape(h,h-1,len(alphas))
    for i in range(h):cube[i,[j for j in range(h) if j!=i]]=values[i]
    best,index=cube.min(-1);alpha=alphas.cpu()[index];alpha[torch.arange(h),torch.arange(h)]=torch.nan
    return drops[None],best[None],alpha[None],cube[None]
