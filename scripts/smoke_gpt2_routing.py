"""32 real pretrained prefixes: numerical correctness and measured cost."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from src.gpt2_routing import GPT2Suffix,output_kl


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,default=11);args=p.parse_args()
    root=args.root;out=root/f'smoke_layer{args.layer}.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(root/'model',local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    assert (model.config.n_layer,model.config.n_head,model.config.n_embd)==(12,12,768)
    engine=GPT2Suffix(model,args.layer);data=np.load(root/'data/train.npz')['input_ids'][:32];rows=[]
    for n,tokens in enumerate(data):
        ids=torch.tensor(tokens[:-1],device='cuda')[None];reference,feature=engine.capture(ids)
        before={k:v.clone() for k,v in engine.cache.items()};scales=torch.ones(4,12,device='cuda');scales[1,0]=0;scales[2,0]=0;scales[2,1]=2;scales[3]=0
        with torch.no_grad():
            logits=engine.suffix(scales);ones_error=float(abs(logits[0]-reference[0]).max());direct_error=0.
            for index in range(4):
                engine.gate.set_scales(scales[index:index+1]);h=model.transformer(ids,use_cache=False).last_hidden_state
                direct=model.lm_head(h[:,-1]);direct_error=max(direct_error,float(abs(direct[0]-logits[index]).max()))
            engine.gate.set_scales(None)
            full=torch.tensor(tokens,device='cuda')[None];h=model.transformer(full,use_cache=False).last_hidden_state
            future_error=float(abs(model.lm_head(h[:,127])-reference).max())
            c=engine.contributions().sum(0)+engine.block.attn.c_proj.bias
            projected=engine.block.attn.c_proj(engine.cache['heads'])[0]
            projection_error=float(abs(c-projected).max())
            torch.testing.assert_close(c,projected,atol=1e-4,rtol=1e-5)
        if max(ones_error,direct_error,future_error)>=3e-4:
            raise RuntimeError(dict(sample=n,ones=ones_error,direct=direct_error,future=future_error,projection=projection_error))
        assert feature.shape==(1,2304)
        torch.cuda.synchronize();start=time.perf_counter()
        engine.evaluate(torch.ones(128,12,device='cuda'),128);torch.cuda.synchronize();seconds=time.perf_counter()-start
        score=engine.taylor(int(tokens[-1]));assert torch.isfinite(score).all()
        rows.append(dict(sample=n,ones_max_error=ones_error,direct_max_error=direct_error,future_max_error=future_error,
            projection_max_error=projection_error,variants128_seconds=seconds,dense_nll=float(torch.nn.functional.cross_entropy(reference.float(),torch.tensor([tokens[-1]],device='cuda')))))
    record=dict(layer=args.layer,parameters=sum(p.numel() for p in model.parameters()),config=model.config.to_dict(),dtype='float32',
        rows=rows,median_seconds_128=float(np.median([r['variants128_seconds'] for r in rows])),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    out.write_text(json.dumps(record,indent=2));engine.close();print(json.dumps({k:v for k,v in record.items() if k not in ('rows','config')},indent=2))


if __name__=='__main__':main()
