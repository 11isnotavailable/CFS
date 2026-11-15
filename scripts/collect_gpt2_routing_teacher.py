"""Resumable pairwise GPT-2 teacher and cached candidate KL without full reruns."""
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent/'controlled_gpt2'))
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from src.gpt2_routing import GPT2Suffix
from scripts.controlled_gpt2.collect_gpt2_cfs import make_masks
from scripts.run_experiment0 import current_distance_matrices


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,default=11)
    p.add_argument('--split',choices=['train','tune'],required=True);p.add_argument('--limit',type=int);p.add_argument('--batch',type=int,default=128)
    args=p.parse_args();root=args.root;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert (root/f'smoke_layer{args.layer}.json').exists()
    data=np.load(root/f'data/{args.split}.npz');ids=data['input_ids'];count=len(ids) if args.limit is None else args.limit
    directory=root/f'layer{args.layer}/pair/{args.split}';directory.mkdir(parents=True,exist_ok=True)
    config=dict(layer=args.layer,split=args.split,alpha_steps=16 if args.split=='train' else 31,alpha_min=0,alpha_max=3,
        dtype='float32',output_kl='reference_to_intervention, temp1, position127, clamp_min0',position=127,
        denominator_floor=1e-12,valid_drop_diagnostic=1e-5,input_hash=hashlib.sha256((root/f'data/{args.split}.npz').read_bytes()).hexdigest())
    path=directory/'config.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:path.write_text(json.dumps(config,indent=2))
    model=AutoModelForCausalLM.from_pretrained(root/'model',local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    engine=GPT2Suffix(model,args.layer);alphas=torch.linspace(0,3,config['alpha_steps']);pair,_=make_masks(12,alphas)
    deletion=torch.ones(12,12);deletion[torch.arange(12),torch.arange(12)]=0
    all_masks=torch.cat((deletion,pair)).cuda();start=time.perf_counter();batch=args.batch
    for sample in range(count):
        path=directory/f'{sample:05d}.npz'
        if path.exists():
            old=np.load(path);assert str(old['document_sha256'])==data['document_sha256'][sample];continue
        reference,features=engine.capture(torch.tensor(ids[sample,:-1],device='cuda')[None])
        while True:
            try:values=engine.evaluate(all_masks,batch).numpy();break
            except torch.cuda.OutOfMemoryError:
                if batch<=1:raise
                batch//=2;torch.cuda.empty_cache()
        drop=values[:12];sub=np.full((12,12),np.nan,np.float32);alpha=sub.copy();cursor=12
        for i in range(12):
            for j in range(12):
                if i==j:continue
                v=values[cursor:cursor+len(alphas)];best=v.argmin();sub[i,j]=v[best];alpha[i,j]=float(alphas[best]);cursor+=len(alphas)
        taylor=engine.taylor(int(ids[sample,-1])).cpu().numpy();c=engine.contributions();flat=c.flatten(1)
        l2,_=current_distance_matrices(c[None]);norm=flat.norm(dim=-1).cpu().numpy()
        cache=engine.cache;residual=cache['residual'];heads=cache['heads']
        if args.layer==11:residual=residual[:,-1:];heads=heads[:,-1:]
        record=dict(sample=sample,document_id=data['document_id'][sample],document_sha256=data['document_sha256'][sample],
            feature=features[0].cpu().numpy(),drops=drop,substitution=sub,alpha=alpha,taylor=taylor,l2=l2,norm=norm,
            residual=residual.cpu().numpy(),heads=heads.cpu().numpy(),reference=reference.cpu().numpy(),target=ids[sample,-1],
            batch=batch,dense_nll=float(torch.nn.functional.cross_entropy(reference,torch.tensor([ids[sample,-1]],device='cuda'))))
        tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**record);tmp.replace(path)
        if (sample+1)%16==0:print('pair',args.layer,args.split,sample+1,count,'seconds',time.perf_counter()-start,'batch',batch,flush=True)
    (directory/f'complete_{count}.json').write_text(json.dumps(dict(count=count,seconds=time.perf_counter()-start,batch=batch)))
    engine.close()


if __name__=='__main__':main()
