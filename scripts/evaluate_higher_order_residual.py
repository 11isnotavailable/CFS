"""Locked exhaustive residual routing and real hard-gating evaluations."""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader
from scripts.evaluate_dynamic_cfs_execution import FrozenPathDataset, dense_kl
from scripts.evaluate_learned_gating import predict
from src.dynamic_cfs_router import DynamicCFSRouter, subset_masks, subset_utilities
from src.higher_order_residual import SetResidual, permute_graph
from src.head_intervention import make_intervenable, clear_interventions
from src.model_loading import create_timm_model
from src.runtime_cfs_router import prehead_summary


def dataset(root,cohort,data,transform):
    if cohort=='confirmation':directory=root/'confirmation/images';paths=[directory/str(x) for x in data['paths']]
    else:
        directory=root/'baseline/learned_gating_inputs_20261115/images'
        paths=[directory/str(x).replace('\\','/').split('imagenette2-160/')[-1] for x in data['paths']]
    return FrozenPathDataset(np.array([str(p) for p in paths]),data['target'],transform)


def backbone(root):
    path=root/'baseline/base.pth'
    manifest=json.loads((root/'baseline/learned_gating_inputs_20261115/manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['backbone_sha256']
    model,_=create_timm_model('vit_base_patch16_224.augreg_in21k_ft_in1k',path)
    model.eval().requires_grad_(False).cuda().half();wrappers=make_intervenable(model)
    transform=create_transform(**resolve_model_data_config(model),is_training=False)
    return model,wrappers,transform


def extract_confirmation(root):
    assert (root/'selection_lock.json').exists()
    output=root/'confirmation/features.npz'
    if output.exists():raise FileExistsError(output)
    data=dict(np.load(root/'confirmation/confirmation.npz'));model,wrappers,transform=backbone(root)
    wrappers[11].capture_input=True;features=[];logits=[]
    loader=DataLoader(dataset(root,'confirmation',data,transform),batch_size=16,shuffle=False)
    with torch.inference_mode():
        for images,_,_ in loader:
            out=model(images.cuda().half());features.append(prehead_summary(wrappers[11].last_input).half().float().cpu().numpy())
            logits.append(out.float().cpu().numpy())
    np.savez_compressed(output,**data,feature=np.concatenate(features),dense_logits=np.concatenate(logits))


def masks(root,cohort):
    lock=json.loads((root/'selection_lock.json').read_text());assert (root/'TRAINING_COMPLETE').exists()
    data=dict(np.load(root/('confirmation/features.npz' if cohort=='confirmation' else 'baseline/learned_gating_inputs_20261115/test.npz')))
    features=data['feature'];all_masks={};times=[];predictions={}
    out=root/f'eval_{cohort}';out.mkdir(exist_ok=False)
    for seed in (17,23,41):
        payload=torch.load(root/f'source/runs/A3_lambda0.01_seed{seed}/model.pt',map_location='cpu',weights_only=False)
        router=DynamicCFSRouter(2304,12).cuda().eval().requires_grad_(False);router.load_state_dict(payload['state_dict'])
        cache=np.load(root/f'cache/train_seed{seed}.npz');cal=cache['calibration']
        parts={k:[] for k in ('encoded','scores','weights')};torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            for begin in range(0,len(features),64):
                x=torch.tensor((features[begin:begin+64]-payload['global_mean'])/payload['global_std'],device='cuda')
                score,importance,_=router(x);h=router.global_encoder(x)
                for key,v in zip(parts,(h,score,importance.softmax(-1))):parts[key].append(v)
        torch.cuda.synchronize();times.append(dict(cohort=cohort,seed=seed,method='frozen_A3_features',seconds=time.perf_counter()-start,inputs=len(features)))
        h,score,weights=[torch.cat(v) for v in parts.values()];h=(h-torch.tensor(cache['h_mean'],device='cuda'))/torch.tensor(cache['h_std'],device='cuda')
        permutation=torch.tensor(np.stack([np.random.default_rng(seed*100000+i+(20000 if cohort=='test' else 30000)).permutation(12) for i in range(len(features))]),device='cuda')
        shuffled=permute_graph(score,permutation)
        residuals={}
        for kind in ('plain','cfs','shuffled'):
            p=torch.load(root/f'selected/{kind}_seed{seed}/model.pt',map_location='cpu',weights_only=False)
            m=SetResidual(kind).cuda().eval();m.load_state_dict(p['state_dict']);residuals[kind]=m
        with torch.inference_mode():
            for ki,k in enumerate((3,6,9)):
                candidates=subset_masks(12,k,device='cuda');utility=subset_utilities(score,weights,candidates)
                selected=candidates[utility.argmax(-1)].cpu().numpy();all_masks[f'{seed}_A3_{k}']=selected
                base=cal[ki,0]-cal[ki,1]*utility;predictions[f'{seed}_base_{k}']=base.cpu().numpy()
                for kind,model in residuals.items():
                    torch.cuda.synchronize();start=time.perf_counter();raw=[]
                    for begin in range(0,len(features),8):
                        stop=min(begin+8,len(features));blocks=[]
                        for c in candidates.split(64):
                            blocks.append(model(h[begin:stop],(shuffled if kind=='shuffled' else score)[begin:stop],
                                weights[begin:stop],c[None].expand(stop-begin,-1,-1),ki))
                        raw.append(torch.cat(blocks,1))
                    raw=torch.cat(raw);gamma=lock['variants'][kind]['gamma'][ki]
                    d=base+gamma*cal[ki,2]*raw
                    chosen=selected if gamma==0 else candidates[d.argmin(-1)].cpu().numpy()
                    all_masks[f'{seed}_{kind}_{k}']=chosen;predictions[f'{seed}_{kind}_{k}']=d.cpu().numpy()
                    # Pre-test diagnostic amendment: fixed unit strength for every
                    # variant, never a selectable alternative to the locked gamma.
                    unit=base+cal[ki,2]*raw
                    all_masks[f'{seed}_{kind}_raw_{k}']=candidates[unit.argmin(-1)].cpu().numpy()
                    predictions[f'{seed}_{kind}_raw_{k}']=unit.cpu().numpy()
                    torch.cuda.synchronize();times.append(dict(cohort=cohort,seed=seed,method=kind,keep=k,seconds=time.perf_counter()-start,inputs=len(features),candidate_count=len(candidates),gamma=gamma))
        basepath=root/'baseline/learned_gating_inputs_20261115'
        for kind,name,path in [('r2','R2',basepath/f'r2_seed{seed}.pt'),('gating','Gating',root/f'baseline/runs/gating_seed{seed}/model.pt')]:
            p=torch.load(path,map_location='cpu',weights_only=False);result,_=predict(p,features,kind,'cuda')
            chosen=result if kind=='gating' else result[0]
            if name=='R2' and cohort=='test':
                old=np.load(basepath/f'r2_predictions_seed{seed}.npz')
                chosen={k:torch.tensor(old[f'keep{k}_dynamic_cfs_router']) for k in (3,6,9)}
            for k,m in chosen.items():all_masks[f'{seed}_{name}_{k}']=m.numpy()
        print('ranked',cohort,seed,flush=True)
    for key,value in all_masks.items():assert np.all(value.sum(-1)==int(key.split('_')[-1]))
    np.savez_compressed(out/'masks.npz',**all_masks)
    np.savez_compressed(out/'candidate_predictions.npz',**predictions)
    pd.DataFrame(times).to_csv(out/'timings.csv',index=False)


def evaluate(root,cohort):
    out=root/f'eval_{cohort}';data=dict(np.load(root/('confirmation/features.npz' if cohort=='confirmation' else 'baseline/learned_gating_inputs_20261115/test.npz')))
    choices=dict(np.load(out/'masks.npz'));model,wrappers,transform=backbone(root);wrapper=wrappers[11]
    loader=DataLoader(dataset(root,cohort,data,transform),batch_size=16,shuffle=False,num_workers=0)
    rows=[];dense=[];start=time.perf_counter()
    with torch.inference_mode():
        for batch,(images,targets,indices) in enumerate(loader):
            images=images.cuda().half();clear_interventions(wrappers);reference=model(images)
            dense.append(reference.float().cpu().numpy());dense_prediction=reference.argmax(-1).cpu().numpy()
            for key,m in choices.items():
                seed,rest=key.split('_',1);method,k=rest.rsplit('_',1)
                wrapper.set_head_scales(torch.tensor(m[indices.numpy()],device='cuda',dtype=torch.float16))
                logits=model(images);kl=dense_kl(reference,logits).cpu().numpy();pred=logits.argmax(-1).cpu().numpy()
                assert np.isfinite(kl).all()
                for offset,idx in enumerate(indices.tolist()):
                    rows.append(dict(cohort=cohort,sample=idx,dataset_index=int(data['dataset_index'][idx]),path=str(data['paths'][idx]),
                        seed=int(seed),method=method,keep=int(k),kl=float(kl[offset]),correct=int(pred[offset]==int(targets[offset])),
                        prediction=int(pred[offset]),dense_prediction=int(dense_prediction[offset]),dense_correct=int(dense_prediction[offset]==int(targets[offset]))))
            if batch%4==0:print('evaluated',cohort,min((batch+1)*16,len(data['paths'])),'seconds',time.perf_counter()-start,flush=True)
    dense=np.concatenate(dense)
    expected=np.load(root/('confirmation/features.npz' if cohort=='confirmation' else 'source/eval_A3/dense_logits.npz'))
    np.testing.assert_array_equal(dense,expected['dense_logits' if cohort=='confirmation' else 'logits'])
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_input.csv',index=False);frame.to_json(out/'per_input.json',orient='records')
    np.savez_compressed(out/'dense_logits.npz',logits=dense)
    check=dict(dense_reference_exact=True,inputs=len(data['paths']),rows=len(rows),seconds=time.perf_counter()-start)
    if cohort=='test':
        old=pd.read_csv(root/'source/routing_per_input.csv');old=old[(old.compensation=='hard') & old.method.isin(['A3','R2','Gating'])]
        now=frame[frame.method.isin(['A3','R2','Gating'])]
        merged=now.merge(old,on=['sample','seed','method','keep'],suffixes=('_now','_old'),validate='one_to_one')
        assert len(merged)==len(now)
        error=float(abs(merged.kl_now-merged.kl_old).max());assert error<1e-7
        check['old_baseline_max_kl_difference']=error
    (out/'checks.json').write_text(json.dumps(check,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['extract','rank','evaluate','all'],default='all')
    p.add_argument('--cohort',choices=['test','confirmation']);args=p.parse_args();root=args.root.resolve();torch.set_num_threads(4)
    assert (root/'selection_lock.json').exists()
    if args.stage in ('extract','all'):extract_confirmation(root)
    for cohort in ((args.cohort,) if args.cohort else ('test','confirmation')):
        if args.stage in ('rank','all'):masks(root,cohort)
        if args.stage in ('evaluate','all'):evaluate(root,cohort)


if __name__=='__main__':main()
