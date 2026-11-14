"""Finite measured quartets, using an unchanged frozen last-block suffix."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from timm.data import create_transform,resolve_model_data_config
from scripts.evaluate_dynamic_cfs_execution import FrozenPathDataset,dense_kl
from src.model_loading import create_timm_model
from src.head_intervention import make_intervenable,clear_interventions
from src.subset_interactions import interaction_masks,quartet_interactions


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args();root=args.root
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    model,_=create_timm_model('vit_base_patch16_224.augreg_in21k_ft_in1k',root/'baseline/base.pth')
    model.eval().requires_grad_(False).cuda().half();wrappers=make_intervenable(model);wrapper=wrappers[11]
    transform=create_transform(**resolve_model_data_config(model),is_training=False)
    checks=[];started=time.perf_counter();processed=0
    for split in ('tune','test'):
        if args.smoke and split=='test':continue
        data=np.load(root/f'baseline/learned_gating_inputs_20261115/{split}.npz')
        selected=np.arange(4 if args.smoke else 64) if split=='tune' else np.load(root/'inputs/test_selected.npy')
        image_root=root/'inputs/images' if split=='tune' else root/'baseline/learned_gating_inputs_20261115/images'
        paths=[str(image_root/str(v).replace('\\','/').split('imagenette2-160/')[-1]) for v in data['paths']]
        dataset=FrozenPathDataset(np.asarray(paths),data['target'],transform)
        for sample in selected:
            image,target,_=dataset[int(sample)];image=image[None].cuda().half()
            clear_interventions(wrappers);wrapper.capture_contributions=True;cache={}
            handle=model.blocks[11].register_forward_pre_hook(lambda module,args:cache.update(x=args[0].detach()))
            with torch.inference_mode():reference=model(image)
            handle.remove();flat=wrapper.last_contributions[0].float().flatten(1)
            cosine=(F.normalize(flat,dim=1)@F.normalize(flat,dim=1).T).cpu().numpy()
            wrapper.capture_contributions=False
            def suffix(scales):
                wrapper.set_head_scales(scales)
                hidden=model.blocks[11](cache['x'].expand(len(scales),-1,-1))
                return model.forward_head(model.norm(hidden))
            masks,pairs,quartets,random_ids=interaction_masks(42+int(sample)+(10000 if split=='test' else 0))
            with torch.inference_mode():
                dense=suffix(torch.ones(1,12,device='cuda',dtype=torch.float16))
                error=float((dense-reference).abs().max())
                if error>.002:raise ValueError(f'Suffix dense reconstruction {error}')
                direct_error=0.
                if processed<4:
                    for mask in masks[:4]:
                        scale=torch.tensor(mask[None],device='cuda',dtype=torch.float16)
                        fast=suffix(scale);direct=model(image)
                        direct_error=max(direct_error,float((fast-direct).abs().max()))
                    if direct_error>.002:raise ValueError(f'Intervention reconstruction {direct_error}')
                values=[];batch=128;offset=0
                while offset<len(masks):
                    scales=torch.tensor(masks[offset:offset+batch],device='cuda',dtype=torch.float16)
                    n=len(scales)
                    if n<batch:scales=torch.cat([scales,scales[-1:].expand(batch-n,-1)])
                    try:
                        values.append(dense_kl(reference,suffix(scales))[:n].cpu().numpy());offset+=n
                    except torch.cuda.OutOfMemoryError:
                        if batch==1:raise
                        batch//=2;torch.cuda.empty_cache()
                kl=np.concatenate(values)
                noop=float(dense_kl(reference,suffix(torch.ones(batch,12,device='cuda',dtype=torch.float16))).max())
            assert np.isfinite(kl).all()
            np.savez_compressed(args.output/f'{split}_{sample:04d}.npz',sample=sample,split=split,dataset_index=data['dataset_index'][sample],
                path=data['paths'][sample],target=target,dense_logits=reference.float().cpu().numpy(),masks=masks,kl=kl,pairs=pairs,
                quartets=quartets,random_ids=random_ids,interaction=quartet_interactions(kl,quartets),cosine=cosine)
            processed+=1;checks.append(dict(split=split,sample=int(sample),dense_max_error=error,direct_max_error=direct_error,
                noop_kl=noop,unique_masks=len(masks),batch=batch))
            if processed%8==0 or args.smoke:print(f'collected {processed} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    (args.output/'checks.json').write_text(json.dumps(checks,indent=2))
    (args.output/'metadata.json').write_text(json.dumps(dict(inputs=processed,seed=42,background_sizes=[1,4,7],pairs=66,
        random_masks_per_budget=48,backbone_frozen=True,new_pairwise_teacher=False,seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':main()
