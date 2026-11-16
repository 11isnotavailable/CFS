"""Instantiate all seven models and verify gate equivalence before training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pandas as pd
from src.controlled_vit_imagenet import SPECS, make_model, config, flop_counts, write_configs
from src.head_intervention import make_intervenable


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); write_configs(a.output/'configs')
    torch.set_num_threads(4); torch.manual_seed(42); rows=[]
    for name,(w,d,h) in SPECS.items():
        model=make_model(name).eval()
        assert len(model.blocks)==d and all(b.attn.num_heads==h and b.attn.head_dim==w//h for b in model.blocks)
        total=sum(p.numel() for p in model.parameters())
        x=torch.randn(1,3,224,224)
        with torch.no_grad():
            dense=model(x); wrappers=make_intervenable(model)
            for wrap in wrappers: wrap.set_head_scales(torch.ones(h))
            gated=model(x)
        error=(dense-gated).abs().max().item()
        torch.testing.assert_close(dense,gated,atol=2e-6,rtol=2e-5)
        rows.append(dict(model=name,width=w,depth=d,heads=h,head_dim=w//h,parameters=total,
                         trainable_parameters=total,gate_max_error=error,**flop_counts(name)))
    table=pd.DataFrame(rows)
    assert table.set_index('model').loc[['V2','V6','V7'],'parameters'].nunique()==1
    table.to_csv(a.output/'architectures.csv',index=False)
    table.to_json(a.output/'architectures.json',orient='records',indent=2)
    table.to_latex(a.output/'architectures.tex',index=False)
    print(table.to_string(index=False))


if __name__=='__main__': main()
