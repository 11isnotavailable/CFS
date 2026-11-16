"""Numerical integration checks only: UNTRAINED models, never scientific results."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from timm.data import ImageNetInfo, resolve_model_data_config
from src.controlled_vit_imagenet import make_model, config, MEAN, STD
from scripts.collect_controlled_vit_imagenet import load_with_frozen_transform
from scripts.analyze_matched_scaling import analyze


def main():
    root=Path(__file__).resolve().parents[1]; output=root/'smoke_only'; output.mkdir(exist_ok=True)
    classes={n:i for i,n in enumerate(ImageNetInfo().label_names())}
    for name in ['V6','V7']:
        torch.manual_seed(42); model=make_model(name); checkpoint=output/f'{name}_UNTRAINED.pt'
        torch.save(dict(model_config=config(name),state_dict=model.state_dict(),class_to_idx=classes),checkpoint)
        restored,_=load_with_frozen_transform(f'controlled_{name}',checkpoint)
        transform=resolve_model_data_config(restored)
        assert tuple(transform['mean'])==MEAN and tuple(transform['std'])==STD and transform['crop_pct']==.875
        for original,loaded in zip(model.parameters(),restored.parameters()): torch.testing.assert_close(original,loaded,rtol=0,atol=0)
        destination=output/name
        subprocess.run([sys.executable,'scripts/collect_controlled_vit_imagenet.py','--model',f'controlled_{name}',
            '--checkpoint',str(checkpoint),'--data-root','/data/cfs_rebuttal_20261114/data/imagenet1k_val',
            '--output-dir',str(destination),'--num-samples','4','--shard-size','4','--alpha-steps','31',
            '--layers','11','--variant-batch-size','64','--save-alpha-error-cube'],cwd=root,check=True)
        h=config(name)['num_heads']; out=output/(name+'_analysis'); out.mkdir(exist_ok=True)
        entry=dict(name=name,family='ViT',parameters=sum(p.numel() for p in model.parameters()),
            model_id=f'controlled_{name}',heads=h,samples=4,layers=[11],N=[1] if h==2 else [1,2],K=[2],path=str(destination))
        analyze(entry,root,out,10,20,42)
    (output/'CHECKS_PASSED.json').write_text(json.dumps(dict(purpose='Correctness tests only; no trained experiment results',
        models=['V6','V7'],checks=['checkpoint roundtrip','preprocessing','FP16 dense/gate equivalence',
        'full alpha cube','original collector and offline graph analysis','H2 no matched2'])))


if __name__=='__main__': main()
