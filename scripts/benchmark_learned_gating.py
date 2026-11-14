"""Isolated router costs; excludes backbone and is not an end-to-end speedup."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.dynamic_cfs_router import DynamicCFSRouter, subset_masks, subset_utilities
from src.learned_head_gating import LearnedHeadGating, topk_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    rows=[];feature=torch.randn(1,2304,device='cuda')
    for name,model in [('Gating',LearnedHeadGating()),('R2',DynamicCFSRouter(2304,12))]:
        model=model.cuda().eval();macs=[0]
        def hook(layer, inputs, output):
            macs[0]+=output.numel()*layer.in_features
        handles=[layer.register_forward_hook(hook) for layer in model.modules() if isinstance(layer,torch.nn.Linear)]
        with torch.inference_mode():model(feature)
        for handle in handles:handle.remove()
        for keep in (3,6,9):
            masks=subset_masks(12,keep,device='cuda')
            def route():
                result=model(feature)
                if name=='Gating':return topk_mask(result,keep)
                utility=subset_utilities(result[0],result[1].softmax(-1),masks,mask_chunk_size=len(masks))
                return masks[utility.argmax(-1)]
            with torch.inference_mode():
                for _ in range(50):route()
                torch.cuda.synchronize();times=[]
                for _ in range(200):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();route();end.record();end.synchronize();times.append(start.elapsed_time(end))
            rows.append(dict(method=name,keep=keep,parameters=sum(v.numel() for v in model.parameters()),
                             linear_macs=macs[0],linear_flops=2*macs[0],median_gpu_ms=np.median(times),
                             p10_gpu_ms=np.quantile(times,.1),p90_gpu_ms=np.quantile(times,.9),
                             mask_candidates=1 if name=='Gating' else len(masks)))
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/'router_costs.csv',index=False)
    (args.output/'router_costs_protocol.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
        batch=1,dtype='float32',warmup=50,repeats=200,random_weights=True,
        latency_scope='router forward and one K selection, GPU events; no backbone, summary or compensation',
        flop_scope='linear multiplies/adds only; excludes GELU, softmax, TopK and subset reductions'),indent=2))


if __name__=='__main__':main()
