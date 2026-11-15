"""Unchanged R2 objectives with language features and real routing validation."""
import argparse,json,random,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from src.dynamic_cfs_router import DynamicCFSRouter,subset_masks,subset_utilities,candidate_subset_utilities
from src.learned_head_gating import LearnedHeadGating,additive_utilities,joint_ranking_loss,topk_mask
from src.gpt2_routing import GPT2Suffix,output_kl
from scripts.train_dynamic_cfs_router import loss_terms


def load_pair(root,layer,split,count):
    records=[dict(np.load(root/f'layer{layer}/pair/{split}/{i:05d}.npz')) for i in range(count)]
    keys=('feature','drops','substitution','alpha','taylor','l2','norm','target')
    data={k:np.stack([r[k] for r in records]) for k in keys}
    data['scores']=1-data['substitution']/np.maximum(data['drops'][:,:,None],1e-12)
    return data,records


def load_backbone(root):
    torch.backends.cuda.matmul.allow_tf32=False
    return AutoModelForCausalLM.from_pretrained(root/'model',local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval().requires_grad_(False)


def selected(model,features,kind,catalogs):
    result=model(features)
    if kind=='Gating':return {k:topk_mask(result,k) for k in catalogs}
    score,importance,_=result
    return {k:m[subset_utilities(score,importance.softmax(-1),m).argmax(-1)] for k,m in catalogs.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,default=11)
    p.add_argument('--phase',choices=['pilot','full'],required=True);p.add_argument('--kind',choices=['bootstrap','R2','Gating','A0'],required=True)
    p.add_argument('--seed',type=int,default=17);args=p.parse_args();root=args.root
    N,V,epochs=(256,64,20) if args.phase=='pilot' else (2048,128,60)
    out=root/f'layer{args.layer}/{args.phase}/runs/{args.kind}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(f'Nonempty training output: {out}')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    train,_=load_pair(root,args.layer,'train',N);tune,records=load_pair(root,args.layer,'tune',V)
    mean=train['feature'].mean(0);std=train['feature'].std(0).clip(1e-5)
    def data_tensors(data,split):
        a=dict(feature=(data['feature']-mean)/std,scores=np.nan_to_num(data['scores'],nan=-1.).clip(-1,1),
            drops=data['drops'],alpha=np.nan_to_num(data['alpha'],nan=1.).clip(0,3))
        if args.kind!='bootstrap':
            joint=[]
            for i in range(len(data['feature'])):
                with np.load(root/f'layer{args.layer}/{args.phase}/joint/{split}/{i:05d}.npz') as file:
                    joint.append({k:file[k] for k in ('masks','kl')})
            a.update(masks=np.stack([j['masks'] for j in joint]).astype(bool),kl=np.stack([j['kl'] for j in joint]))
        return {k:torch.tensor(v,device='cuda') for k,v in a.items()}
    a=data_tensors(train,'train');v=data_tensors(tune,'tune')
    catalogs={k:subset_masks(12,k,device='cuda') for k in (3,6,9)}
    model=(LearnedHeadGating() if args.kind=='Gating' else DynamicCFSRouter(2304,12)).cuda()
    if args.kind=='A0':model.alpha_output.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    config=dict(layer=args.layer,phase=args.phase,kind=args.kind,seed=args.seed,epochs=epochs,train_inputs=N,tune_inputs=V,
        parameters=sum(p.numel() for p in model.parameters()),optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        batch=64,lr=3e-4,weight_decay=1e-4,validation='true full-search hard KL every10epochs' if args.kind!='bootstrap' else 'true pairwise utility every10epochs',test_loaded=False)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    engine=GPT2Suffix(load_backbone(root),args.layer) if args.kind!='bootstrap' else None
    history=[];best=float('inf');generator=torch.Generator().manual_seed(args.seed);started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for ids in torch.randperm(N,generator=generator).split(64):
            ids=ids.cuda();optimizer.zero_grad(set_to_none=True);feature=a['feature'][ids]
            if args.kind in ('bootstrap','R2'):
                s,w,alpha=model(feature)
                terms=loss_terms(s,w,alpha,a['scores'][ids],a['drops'][ids],a['alpha'][ids],catalogs,.1,.05,
                    None if args.kind=='bootstrap' else a['masks'][ids],None if args.kind=='bootstrap' else a['kl'][ids])
                loss=terms['listwise']+.5*terms['score']+.5*terms['importance']+.1*terms['alpha']+terms['structured']+.5*terms.get('joint',0.)
            else:
                result=model(feature)
                if args.kind=='Gating':u=torch.stack([additive_utilities(result,a['masks'][ids,k]) for k in range(3)],1)
                else:
                    s,w,_=result;u=torch.stack([candidate_subset_utilities(s,w.softmax(-1),a['masks'][ids,k]) for k in range(3)],1)
                loss=.5*joint_ranking_loss(u,a['kl'][ids])
            assert torch.isfinite(loss);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.perf_counter()-started)
        if epoch%10==0:
            model.eval()
            with torch.inference_mode():
                choices=selected(model,v['feature'],args.kind,catalogs)
                if args.kind=='bootstrap':
                    values=[]
                    for k,masks in choices.items():
                        u=candidate_subset_utilities(v['scores'],v['drops'],masks[:,None]).squeeze(-1);values.append(-u.mean().item())
                    criterion=float(np.mean(values));row['val_negative_utility']=criterion
                else:
                    values=[]
                    for i,r in enumerate(records):
                        engine.cache={key:torch.tensor(r[key],device='cuda') for key in ('residual','heads','reference')}
                        scales=torch.stack([choices[k][i] for k in catalogs]).float()
                        values.append(output_kl(engine.cache['reference'],engine.suffix(scales)).cpu().numpy())
                    values=np.stack(values);criterion=float(values.mean())
                    row.update({f'val_K{k}':float(values[:,ki].mean()) for ki,k in enumerate(catalogs)})
                    row['val_hard_kl']=criterion
                    np.savez_compressed(out/f'validation_epoch{epoch}.npz',kl=values,**{f'mask{k}':m.cpu().numpy() for k,m in choices.items()})
                payload=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},global_mean=mean,global_std=std,config=config,
                    best_epoch=epoch,validation_metric=criterion)
                torch.save(payload,out/f'epoch{epoch}.pt')
                if criterion<best:
                    best=criterion;shutil.copyfile(out/f'epoch{epoch}.pt',out/'model.pt')
            print(args.layer,args.phase,args.kind,args.seed,row,flush=True)
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    if engine:engine.close()
    (out/'COMPLETE.json').write_text(json.dumps(dict(best=best,seconds=time.perf_counter()-started)))


if __name__=='__main__':main()
