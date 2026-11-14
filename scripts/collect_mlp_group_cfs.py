"""Analytical MLP Group CFS; fixed backbone, original KL and alpha protocol."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from timm.data import create_transform, resolve_model_data_config, ImageNetInfo
from src.model_loading import create_timm_model
from src.mlp_group_intervention import neuron_groups, CachedMLP, capture_mlp_inputs
from scripts.collect_predictor_dataset import evenly_subsample_indices, target_for_path
from scripts.run_experiment0 import (make_dataset, output_kl, build_substitution_masks,
    summarize_substitutions, substitution_error_cube, current_distance_matrices)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_masks(cached,masks,reference,batch):
    results=[]; start=0
    while start<len(masks):
        try:
            chunk=masks[start:start+batch].to(reference.device)
            count=len(chunk)
            # Identical GEMM shapes for deletions, replacements, and alpha=1.
            # Padding is discarded and never treated as an observation.
            if count<batch: chunk=torch.cat([chunk,chunk[-1:].expand(batch-count,-1)])
            with torch.no_grad(): values=output_kl(reference,cached.logits(chunk),1.)
            if not torch.isfinite(values).all(): raise ValueError('Nonfinite intervention KL')
            results.append(values[:count].cpu()); start+=count
        except torch.cuda.OutOfMemoryError:
            if batch==1: raise
            batch=max(1,batch//2); torch.cuda.empty_cache()
    return torch.cat(results),batch


def check(cached,image,reference,contributions,full):
    a=cached.cache['activation']; fc=cached.mlp.fc2; g=len(cached.groups)
    reconstruction=contributions.sum(1)+fc.bias.float()
    dense32=F.linear(a.float(),fc.weight.float(),fc.bias.float())
    recon_error=float((reconstruction-dense32).abs().max())
    torch.testing.assert_close(reconstruction,dense32,atol=2e-4,rtol=2e-5)
    with torch.no_grad(): dense=cached.logits(torch.ones(1,g,device=image.device))
    dense_error=float((dense-reference).abs().max())
    if dense_error>2e-3: raise ValueError(f'Dense cached path failed: {dense_error}')
    result=dict(reconstruction_fp32_max_abs=recon_error,reconstruction_vs_fp16_max_abs=float((reconstruction-fc(a).float()).abs().max()),dense_logits_max_abs=dense_error)
    if full:
        masks=torch.ones(4,g,device=image.device); masks[1:,0]=0; masks[2,1]=1.7; masks[3,1]=0
        with torch.no_grad():
            fast=torch.cat([cached.logits(row[None]) for row in masks])
            direct=torch.cat([cached.direct_logits(image,row[None]) for row in masks])
            full_batch=cached.direct_logits(image,masks)
        logit_error=float((fast-direct).abs().max())
        kl_error=float((output_kl(reference,fast,1)-output_kl(reference,direct,1)).abs().max())
        if logit_error>2e-3 or kl_error>1e-5:
            raise ValueError(f'Cached vs full intervention mismatch: {logit_error}, {kl_error}')
        result.update(full_logits_max_abs=logit_error,full_kl_max_abs=kl_error,
            fp16_full_batch_shape_logits_max_abs=float((full_batch-direct).abs().max()),
            fp16_full_batch_shape_kl_max_abs=float((output_kl(reference,full_batch,1)-output_kl(reference,direct,1)).abs().max()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--indices-metadata',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=64); p.add_argument('--layers',default='6,11')
    p.add_argument('--groups',default='8,16'); p.add_argument('--partitions',default='contiguous,42,123,2027')
    p.add_argument('--max-batch',type=int,default=128); p.add_argument('--verify-full-inputs',type=int,default=4)
    p.add_argument('--device',default='cuda'); args=p.parse_args()
    if not 1<=args.samples<=64: raise ValueError('This protocol is limited to 64 inputs')
    args.output.mkdir(parents=True,exist_ok=False)
    name='vit_base_patch16_224.augreg_in21k_ft_in1k'
    model,_=create_timm_model(name,args.checkpoint)
    model.eval().requires_grad_(False).to(device=args.device,dtype=torch.float16)
    transform_config=resolve_model_data_config(model)
    dataset=make_dataset(args.data_root,create_transform(**transform_config,is_training=False))
    assert len(dataset)==50000
    source=json.loads(args.indices_metadata.read_text())
    # Smoke uses first four of the FINAL 64, so its inputs are not a new cohort.
    indices=evenly_subsample_indices(np.asarray(source['sample_indices']),64)[:args.samples]
    layers=list(map(int,args.layers.split(','))); group_counts=list(map(int,args.groups.split(',')))
    configurations=[]
    for g in group_counts:
        for part in args.partitions.split(','):
            seed=None if part=='contiguous' else int(part)
            configurations.append((g,part,seed))
    labels={v:i for i,v in enumerate(ImageNetInfo().label_names())}
    alphas=torch.linspace(0,3,31)
    metadata=dict(model=name,checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        preprocessing=transform_config,layers=layers,groups=group_counts,partitions=args.partitions.split(','),
        sample_indices=indices.tolist(),samples=len(indices),alpha_grid=alphas.tolist(),
        alpha_parameterization='total representative gain; alpha=1 is source deletion; alpha=0 deletes both',
        discrepancy='KL(p_dense || p_intervened), temperature=1, FP32 logits, clamp tiny negative KL to zero',
        minimum_drop_kl=1e-5,normalization_floor=1e-12,dtype='float16',
        fixed_seed=42,indices_metadata_sha256=sha(args.indices_metadata),
        scripts_sha256={str(path.name):sha(path) for path in [Path(__file__),Path(__file__).parents[1]/'src/mlp_group_intervention.py']})
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    started=time.monotonic(); input_records=[]; checks=[]
    for sample,index in enumerate(indices):
        image,_,path=dataset[int(index)]; image=image[None].to(args.device,dtype=torch.float16)
        target=target_for_path(path,labels); assert target>=0
        reference,caches=capture_mlp_inputs(model,image,layers)
        input_records.append(dict(input_id=int(index),path=str(path),image_sha256=sha(path),target=target,prediction=int(reference.argmax(-1))))
        for layer in layers:
            for g,part,seed in configurations:
                grouping=neuron_groups(model.blocks[layer].mlp.fc1.out_features,g,seed)
                cached=CachedMLP(model,layer,caches[layer],grouping)
                with torch.no_grad(): contributions=cached.contributions()
                diagnostics=check(cached,image,reference,contributions,sample<args.verify_full_inputs)
                scales=torch.ones(1,g,device=args.device,requires_grad=True)
                loss=F.cross_entropy(cached.logits(scales).float(),torch.tensor([target],device=args.device))
                gradient=torch.autograd.grad(loss,scales)[0]
                taylor=(gradient*scales).abs().detach().float().cpu().numpy()[0]
                _,cosine_distance=current_distance_matrices(contributions)
                del contributions,loss,gradient,scales
                free,_=torch.cuda.mem_get_info() if str(args.device).startswith('cuda') else (2**30,2**30)
                bytes_per_variant=caches[layer]['activation'].numel()*64
                batch=min(args.max_batch,max(1,int(free*.15/bytes_per_variant)))
                deletion=torch.ones(g,g); deletion[torch.arange(g),torch.arange(g)]=0
                drop,batch=run_masks(cached,deletion,reference,batch)
                no_op,_=run_masks(cached,torch.ones(1,g),reference,batch)
                masks,mask_metadata=build_substitution_masks(g,alphas)
                errors,batch=run_masks(cached,masks,reference,batch)
                sub,best_alpha=summarize_substitutions(g,mask_metadata,errors)
                cube=substitution_error_cube(g,len(alphas),mask_metadata,errors)
                score=1-sub/np.maximum(drop.numpy()[:,None],1e-12); np.fill_diagonal(score,np.nan)
                if not np.isfinite(score[~np.eye(g,dtype=bool)]).all(): raise ValueError('Nonfinite CFS')
                # Grid alpha=1 replays the deletion mask exactly; differences
                # only reflect batched numerical kernels. Record, never hide.
                one=np.flatnonzero(np.isclose(alphas.numpy(),1))[0]
                identity_errors=cube[:,:,one]-drop.numpy()[:,None]
                alpha1_error=float(np.nanmax(np.abs(identity_errors)))
                if alpha1_error>1e-4: raise ValueError(f'Alpha=1 deletion mismatch: {alpha1_error}')
                diagnostics.update(input_id=int(index),layer=layer,G=g,partition=part,
                    alpha1_drop_kl_max_abs=alpha1_error,actual_batch=batch,nonfinite_pair_fraction=0.,
                    no_op_batch_kl=float(no_op[0]),
                    valid_sources_within_10x_no_op_fraction=float(((drop>1e-5)&(drop<=10*no_op[0])).float().mean()))
                checks.append(diagnostics)
                folder=args.output/f'G{g}_{part}_L{layer}'; folder.mkdir(exist_ok=True)
                output=folder/f'input_{int(index):05d}.npz'
                np.savez_compressed(output,input_id=index,layer=layer,G=g,group_indices=grouping,
                    single_drop=drop.numpy(),substitution=sub,best_alpha=best_alpha,cfs=score,
                    alpha_grid=alphas.numpy(),alpha_error_cube=cube,taylor=taylor,
                    contribution_cosine=1-cosine_distance,valid=drop.numpy()>1e-5,no_op_kl=no_op.numpy()[0])
        (args.output/'input_provenance.json').write_text(json.dumps(input_records,indent=2))
        (args.output/'numerical_checks.json').write_text(json.dumps(checks,indent=2))
        print(f'input={sample+1}/{len(indices)} elapsed={(time.monotonic()-started)/60:.2f}m',flush=True)
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(inputs=len(indices),configurations=len(configurations)*len(layers))))
    print('COLLECTION_COMPLETE',flush=True)


if __name__=='__main__': main()
