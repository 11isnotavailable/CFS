"""Cached GPT-2 head interventions, preserving the original c_proj bias."""
import torch
import torch.nn.functional as F
from scripts.controlled_gpt2.gpt2_head_intervention import GPT2HeadGate
from src.dynamic_cfs_router import DynamicCFSRouter,subset_masks,selected_masks
from src.learned_head_gating import LearnedHeadGating,topk_mask


def prefix_summary(hidden):
    value=hidden.float()
    return torch.cat((value[:,-1],value.mean(1),value.var(1,unbiased=False)),dim=-1)


def output_kl(reference,variants):
    logp=reference.float().log_softmax(-1)
    return (logp.exp()*(logp-variants.float().log_softmax(-1))).sum(-1).clamp_min(0)


class GPT2Suffix:
    def __init__(self,model,layer):
        self.model=model;self.layer=layer;self.block=model.transformer.h[layer]
        self.gate=GPT2HeadGate(model,layer);self.cache={}
        self.handles=[self.block.register_forward_pre_hook(self._input),
            self.block.ln_1.register_forward_hook(self._norm),
            self.block.attn.c_proj.register_forward_pre_hook(self._heads)]
        self.capturing=False

    def _input(self,module,args):
        if self.capturing:self.cache['residual']=args[0].detach()

    def _norm(self,module,args,out):
        if self.capturing:self.cache['prehead']=out.detach()

    def _heads(self,module,args):
        if self.capturing:self.cache['heads']=args[0].detach()

    def capture(self,ids,attention_mask=None):
        if ids.ndim!=2 or len(ids)!=1:raise ValueError('One unpadded document prefix required')
        if attention_mask is not None and not bool(attention_mask.all()):raise ValueError('Padded prefixes must be trimmed before capture')
        self.gate.set_scales(None);self.capturing=True
        with torch.no_grad():
            h=self.model.transformer(ids,use_cache=False,return_dict=True).last_hidden_state
            logits=self.model.lm_head(h[:,-1])
        self.capturing=False;self.cache['reference']=logits.detach()
        return logits,prefix_summary(self.cache['prehead'])

    def suffix(self,scales,heads=None):
        B=len(scales);base=self.cache['heads'] if heads is None else heads
        residual=self.cache['residual']
        if self.layer==len(self.model.transformer.h)-1:
            base=base[:,-1:];residual=residual[:,-1:]
        H=self.gate.num_heads;D=self.gate.head_dim
        gated=(base.expand(B,-1,-1).reshape(B,-1,H,D)*scales[:,None,:,None].to(base.dtype)).reshape(B,-1,H*D)
        # Gate hook is inactive here: projection bias is applied exactly once.
        self.gate.set_scales(None)
        h=residual.expand(B,-1,-1)+self.block.attn.resid_dropout(self.block.attn.c_proj(gated))
        h=h+self.block.mlp(self.block.ln_2(h))
        for block in self.model.transformer.h[self.layer+1:]:
            value=block(h,attention_mask=None,use_cache=False)
            h=value[0] if isinstance(value,tuple) else value
        h=self.model.transformer.ln_f(h[:,-1])
        return self.model.lm_head(h)

    def evaluate(self,scales,batch=128):
        values=[]
        with torch.inference_mode():
            for local in scales.split(batch):values.append(output_kl(self.cache['reference'],self.suffix(local)).cpu())
        return torch.cat(values)

    def taylor(self,target):
        # Original ViT baseline: absolute derivative of supervised CE w.r.t.
        # one shared scalar gate per head (absolute value AFTER summation).
        device=self.cache['heads'].device
        with torch.enable_grad():
            scales=torch.ones(1,self.gate.num_heads,device=device,requires_grad=True)
            logits=self.suffix(scales)
            loss=F.cross_entropy(logits.float(),torch.tensor([target],device=device))
            grad=torch.autograd.grad(loss,scales)[0]
        return (scales*grad).abs()[0].detach()

    def contributions(self):
        heads=self.cache['heads'][0].float().reshape(-1,self.gate.num_heads,self.gate.head_dim)
        weight=self.block.attn.c_proj.weight.detach().float().reshape(self.gate.num_heads,self.gate.head_dim,-1)
        return torch.einsum('thd,hdc->htc',heads,weight)

    def close(self):
        self.gate.close()
        for h in self.handles:h.remove()


class GPT2RuntimeRouter:
    """Forward-only prefix-to-next-token routing; no labels, teacher or KV cache.

    The returned result is ONLY the last prefix position. A shared prefix gate
    is not a claim of causal per-token routing for earlier logits in that pass.
    """
    def __init__(self,model,layer,payload,kind,keep):
        device=next(model.parameters()).device
        self.model=model;self.kind=kind;self.keep=keep
        self.router=(LearnedHeadGating() if kind=='Gating' else DynamicCFSRouter(2304,12)).to(device).eval().requires_grad_(False)
        self.router.load_state_dict(payload['state_dict'])
        self.mean=torch.as_tensor(payload['global_mean'],device=device);self.std=torch.as_tensor(payload['global_std'],device=device)
        self.catalog=subset_masks(12,keep,device=device);self.gate=GPT2HeadGate(model,layer);self.last_mask=None
        self.handle=model.transformer.h[layer].ln_1.register_forward_hook(self._route)

    def _route(self,module,args,hidden):
        features=(prefix_summary(hidden)-self.mean)/self.std
        result=self.router(features)
        if self.kind=='Gating':mask=topk_mask(result,self.keep)
        else:
            scores,weights,_=result;mask=selected_masks(scores,weights.softmax(-1),self.catalog)[0]
        self.last_mask=mask;self.gate.set_scales(mask.float())

    def next_logits(self,ids):
        hidden=self.model.transformer(ids,use_cache=False,return_dict=True).last_hidden_state
        return self.model.lm_head(hidden[:,-1])

    def close(self):
        self.handle.remove();self.gate.close()
