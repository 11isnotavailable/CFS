"""Untrained correctness checks only; original full oracle versus cached/streamed."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from src.bert_medium import make_model
from src.bert_medium_intervention import wrap_last,CachedSuffix,stream_oracle
from scripts.collect_bert_mlm_scaling import selected_mlm_logits,intervention_oracle

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'smoke';out.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42);model=make_model('B6').cuda().eval().requires_grad_(False)
    cache=np.load('/data/cfs_bert_benchmark_20261116/data/train_tokens.npy',mmap_mode='r')
    record=[]
    for i in range(4):
        ids=torch.tensor(cache[i:i+1].astype(np.int64),device='cuda');positions=torch.tensor([[1,8,16,32,48,64,80,96]],device='cuda');ids[0,positions[0]]=103;inputs=dict(input_ids=ids)
        if i==0:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):untouched=selected_mlm_logits(model,inputs,positions).cpu()
        gate=wrap_last(model);gate.set_head_scales(None);captured=[]
        hook=model.bert.encoder.layer[-1].register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):reference=selected_mlm_logits(model,inputs,positions).cpu()
        hook.remove();suffix=CachedSuffix(model,captured[0]);alphas=torch.linspace(0,3,16,device='cuda')
        if i==0:torch.testing.assert_close(untouched,reference,rtol=0,atol=0)
        started=time.perf_counter();drop,sub,alpha,cube=stream_oracle(suffix,gate,inputs,positions,reference,alphas)
        streamed_seconds=time.perf_counter()-started
        original_drop,original_sub,original_alpha=intervention_oracle(model,gate,inputs,positions,reference,alphas,32)
        finite=torch.isfinite(original_sub)
        torch.testing.assert_close(drop,original_drop,rtol=.02,atol=2e-6)
        torch.testing.assert_close(sub[finite],original_sub[finite],rtol=.02,atol=2e-6)
        # The unchanged candidate multiplier1 in the original protocol equals source deletion.
        target=drop[0,:,None].expand(4,4);off=~torch.eye(4,dtype=torch.bool)
        torch.testing.assert_close(cube[0,:,:,5][off],target[off],rtol=.02,atol=2e-6)
        assert torch.isnan(sub[0].diagonal()).all() and torch.isfinite(sub[finite]).all()
        record.append(dict(input=i,max_drop_difference=float((drop-original_drop).abs().max()),
            max_substitution_difference=float((sub[finite]-original_sub[finite]).abs().max()),
            alpha_agreement=float((alpha[finite]==original_alpha[finite]).float().mean()),streamed_seconds=streamed_seconds))
    (out/'CFS_CHECKS_PASSED.json').write_text(json.dumps(dict(purpose='Untrained correctness only',checks=record),indent=2));print(json.dumps(record),flush=True)
if __name__=='__main__':main()
