"""Full-prefix next-token wall time; logical gates do not skip dense kernels."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.gpt2_routing import GPT2RuntimeRouter
from scripts.train_gpt2_routing import load_backbone


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,required=True);a=p.parse_args()
    out=a.root/f'layer{a.layer}/runtime';out.mkdir(exist_ok=False);torch.set_num_threads(4)
    model=load_backbone(a.root);ids=torch.tensor(np.load(a.root/'data/train.npz')['input_ids'][:32,:-1],device='cuda');rows=[]
    with torch.inference_mode():
        for kind,k in [('Dense',12)]+[(kind,k) for kind in ('R2','Gating','A0') for k in (3,6,9)]:
            live=None
            if kind!='Dense':
                payload=torch.load(a.root/f'layer{a.layer}/full/runs/{kind}_seed17/model.pt',map_location='cpu',weights_only=False)
                live=GPT2RuntimeRouter(model,a.layer,payload,kind,k);forward=live.next_logits
            else:
                def forward(x):return model.lm_head(model.transformer(x,use_cache=False).last_hidden_state[:,-1])
            for _ in range(3):forward(ids[:1])
            for i in range(32):
                for repeat in range(3):
                    torch.cuda.synchronize();start=time.perf_counter();forward(ids[i:i+1]);torch.cuda.synchronize()
                    rows.append(dict(layer=a.layer,method=kind,keep=k,sample=i,repeat=repeat,seconds=time.perf_counter()-start))
            if live:live.close()
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_input.csv',index=False)
    summary=frame.groupby(['layer','method','keep']).seconds.agg(['mean','median','std']).reset_index()
    summary['median_relative_to_dense']=summary['median']/float(summary.loc[summary.method=='Dense','median'].iloc[0])
    summary.to_csv(out/'summary.csv',index=False)
    (out/'config.json').write_text(json.dumps(dict(split='train',samples=32,repeats=3,seed=17,batch=1,prefix=128,dtype='float32',
        includes='whole backbone prefix forward, features, normalization, router, subset search, gates, next-token LM head',
        excludes='model construction, tokenization, data transfer',kv_cache=False,optimized_sparse_kernels=False),indent=2))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
