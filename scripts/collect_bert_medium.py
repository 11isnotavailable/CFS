"""Same BERT MLM CFS math, cached final-block suffix and streaming KL chunks."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from src.bert_medium import SPECS,P,make_model,fingerprint
from src.bert_medium_intervention import wrap_last,CachedSuffix,stream_oracle
from scripts.collect_bert_mlm_scaling import selected_mlm_logits

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=SPECS,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,choices=[4,64,256],required=True)
    a=p.parse_args();torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert payload['protocol_hash']==fingerprint() and payload['model_id']==a.model
    model=make_model(a.model).cuda().eval();model.load_state_dict(payload['model']);model.requires_grad_(False)
    gate=wrap_last(model);frozen=np.load(a.data);indices=np.linspace(0,255,a.count,dtype=int)
    metadata=dict(model='medium_'+a.model,heads=SPECS[a.model][2],layers=[SPECS[a.model][1]-1],
        parameters=sum(p.numel() for p in model.parameters()),sample_indices=indices.tolist(),samples=a.count,
        training_tokens=payload['tokens'],alpha_grid=np.linspace(0,3,16).tolist(),mask_positions=8,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        input_identity_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        input_document_hashes=frozen['document_sha256'][indices].tolist(),
        protocol='Original BERT MLM FP16 autocast,8 fixedMASK positions,absolute candidate alpha; final-block suffix cache')
    meta=a.output/'metadata.json'
    if meta.exists():assert json.loads(meta.read_text())==metadata
    else:meta.write_text(json.dumps(metadata,indent=2))
    started=time.time();alphas=torch.linspace(0,3,16,device='cuda')
    for chunk_start in range(0,len(indices),8):
        dest=a.output/f'shard_{chunk_start//8:05d}.npz'
        if dest.exists():continue
        records=[]
        for index in indices[chunk_start:chunk_start+8]:
            ids=torch.tensor(frozen['input_ids'][index:index+1],device='cuda');positions=torch.tensor(frozen['positions'][index:index+1],device='cuda')
            inputs=dict(input_ids=ids);gate.set_head_scales(None);captured=[]
            handle=model.bert.encoder.layer[-1].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):reference=selected_mlm_logits(model,inputs,positions).cpu()
            handle.remove();suffix=CachedSuffix(model,captured[0])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):cached=selected_mlm_logits(suffix,inputs,positions).cpu()
            torch.testing.assert_close(reference,cached,rtol=1e-4,atol=1e-4)
            drop,sub,alpha,cube=stream_oracle(suffix,gate,inputs,positions,reference,alphas)
            if index==indices[0]:
                fd,fs,fa,_=stream_oracle(model,gate,inputs,positions,reference,alphas)
                off=torch.isfinite(fs)
                torch.testing.assert_close(drop,fd,rtol=.02,atol=2e-6)
                torch.testing.assert_close(sub[off],fs[off],rtol=.02,atol=2e-6)
                (a.output/'prefix_equivalence.json').write_text(json.dumps(dict(input_id=int(index),
                    max_drop_error=float((drop-fd).abs().max()),max_substitution_error=float((sub[off]-fs[off]).abs().max()),
                    valid_source_flips=int(((drop>1e-5)!=(fd>1e-5)).sum()),
                    alpha_agreement=float((alpha[off]==fa[off]).float().mean())),indent=2))
            records.append(dict(dataset_index=int(index),single_drop=drop.numpy(),substitution=sub.numpy(),best_alpha=alpha.numpy(),alpha_error_cube=cube.numpy()))
            del suffix,captured;print('CFS',a.model,int(index),flush=True)
        arrays={k:np.stack([r[k] for r in records]) for k in ['single_drop','substitution','best_alpha','alpha_error_cube']}
        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,dataset_index=np.array([r['dataset_index'] for r in records]),**arrays);tmp.replace(dest)
    (a.output/'COMPLETE.json').write_text(json.dumps(dict(seconds=time.time()-started,**metadata),indent=2))
if __name__=='__main__':main()
