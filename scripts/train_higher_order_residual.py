"""Nine bounded residual fits on immutable training teachers; no test loading."""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.dynamic_cfs_router import DynamicCFSRouter, candidate_subset_utilities
from src.higher_order_residual import SetResidual, split_masks, monotone_calibration, permute_graph

SEEDS=(17,23,41)
KINDS=('plain','cfs','shuffled')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_cache(payload, features, masks, device='cuda'):
    model=DynamicCFSRouter(2304,12).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload['state_dict']);parts={k:[] for k in ('encoded','scores','weights','utility')}
    with torch.inference_mode():
        for begin in range(0,len(features),64):
            f=torch.tensor((features[begin:begin+64]-payload['global_mean'])/payload['global_std'],device=device)
            score,importance,_=model(f);encoded=model.global_encoder(f);weights=importance.softmax(-1)
            utility=torch.stack([candidate_subset_utilities(score,weights,torch.tensor(masks[begin:begin+64,k],device=device).bool()) for k in range(3)],1)
            for name,value in zip(parts,(encoded,score,weights,utility)):parts[name].append(value.cpu().numpy())
    return {k:np.concatenate(v) for k,v in parts.items()}


def prepare(root):
    output=root/'cache';output.mkdir(exist_ok=False)
    data=root/'baseline/learned_gating_inputs_20261115'
    train,tune=[dict(np.load(data/f'{s}.npz')) for s in ('train','tune')]
    def canon(paths):return {str(v).replace('\\','/').split('imagenette2-160/')[-1] for v in paths}
    assert not canon(train['paths']) & canon(tune['paths'])
    fit,held=split_masks(train['masks']);np.savez_compressed(output/'mask_split.npz',fit=fit,held=held)
    hashes={str(data/f'{s}.npz'):digest(data/f'{s}.npz') for s in ('train','tune')}
    for seed in SEEDS:
        path=root/f'source/runs/A3_lambda0.01_seed{seed}/model.pt';hashes[str(path)]=digest(path)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        caches={s:frozen_cache(payload,a['feature'],a['masks']) for s,a in [('train',train),('tune',tune)]}
        mean=caches['train']['encoded'].mean(0);std=caches['train']['encoded'].std(0).clip(1e-5)
        calibration=[]
        for k in range(3):
            u=np.take_along_axis(caches['train']['utility'][:,k],fit[:,k],1)
            y=np.take_along_axis(train['kl'][:,k],fit[:,k],1)
            calibration.append(monotone_calibration(u,y))
        for split,a in [('train',train),('tune',tune)]:
            c=caches[split];c['encoded']=(c['encoded']-mean)/std
            permutation=np.stack([np.random.default_rng(seed*100000+i+(0 if split=='train' else 10000)).permutation(12) for i in range(len(a['paths']))])
            np.savez_compressed(output/f'{split}_seed{seed}.npz',**c,permutation=permutation,
                h_mean=mean,h_std=std,calibration=np.array(calibration,dtype=np.float32))
        print('cached seed',seed,flush=True)
    (root/'source_hashes.json').write_text(json.dumps(hashes,indent=2))


def arrays(root,split,seed,indices=None,device='cuda'):
    a=dict(np.load(root/f'baseline/learned_gating_inputs_20261115/{split}.npz'))
    c=dict(np.load(root/f'cache/{split}_seed{seed}.npz'))
    masks=a['masks'].astype(bool);kl=a['kl'];utility=c['utility']
    if indices is not None:
        masks=np.take_along_axis(masks,indices[...,None],2)
        kl=np.take_along_axis(kl,indices,2);utility=np.take_along_axis(utility,indices,2)
    cal=c['calibration'];base=cal[None,:,0,None]-cal[None,:,1,None]*utility
    output={k:torch.tensor(v,device=device) for k,v in dict(encoded=c['encoded'],scores=c['scores'],weights=c['weights'],
        masks=masks,kl=kl,base=base,target=(kl-base)/cal[None,:,2,None],permutation=c['permutation']).items()}
    return output,cal


def predict(model,a,ids):
    score=a['scores'][ids]
    if model.kind=='shuffled':score=permute_graph(score,a['permutation'][ids])
    return torch.stack([model(a['encoded'][ids],score,a['weights'][ids],a['masks'][ids,k],k) for k in range(3)],1)


def train(root,kind,seed,epochs=20):
    output=root/f'runs/{kind}_seed{seed}';output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.use_deterministic_algorithms(True)
    fit=np.load(root/'cache/mask_split.npz')['fit'];a,cal=arrays(root,'train',seed,fit);v,_=arrays(root,'tune',seed)
    model=SetResidual(kind).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    config=dict(kind=kind,seed=seed,epochs=epochs,lr=3e-4,batch=64,weight_decay=1e-4,
        parameters=sum(p.numel() for p in model.parameters()),fit_masks=36,test_loaded=False,
        train_supervision='Existing train joint KL only; no measured tune/test quartet labels')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    history=[];start=time.perf_counter();generator=torch.Generator().manual_seed(seed)
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(a['encoded']),generator=generator);train_loss=[]
        for ids in order.split(64):
            ids=ids.cuda();optimizer.zero_grad(set_to_none=True)
            prediction=predict(model,a,ids);loss=(prediction-a['target'][ids]).square().mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite residual loss')
            loss.backward();optimizer.step();train_loss.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            pred=predict(model,v,torch.arange(len(v['encoded']),device='cuda'))
            errors=(pred-v['target']).square().mean((0,2)).cpu().numpy()
        history.append(dict(epoch=epoch,train_mse=np.mean(train_loss),val_mse=errors.mean(),
            **{f'val_K{k}':errors[i] for i,k in enumerate((3,6,9))},seconds=time.perf_counter()-start))
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=config,calibration=cal,epoch=epoch),output/f'epoch_{epoch}.pt')
        np.savez_compressed(output/f'tune_epoch_{epoch}.npz',residual=pred.cpu().numpy())
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        if epoch==1 or epoch%5==0:print(kind,seed,epoch,history[-1],flush=True)
    (output/'COMPLETE').write_text('20 fixed epochs completed\n')


def lock(root):
    import shutil
    all_history=[];selection={};rows=[]
    tune=np.load(root/'baseline/learned_gating_inputs_20261115/tune.npz')['kl']
    for kind in KINDS:
        histories=[]
        for seed in SEEDS:
            run=root/f'runs/{kind}_seed{seed}';assert (run/'COMPLETE').exists()
            h=pd.read_csv(run/'history.csv');h['seed']=seed;h['kind']=kind;histories.append(h);all_history.append(h)
        means=pd.concat(histories).groupby('epoch').val_mse.mean()
        epoch=int(means.idxmin());predictions=[];baselines=[]
        for seed in SEEDS:
            c=np.load(root/f'cache/tune_seed{seed}.npz');cal=c['calibration']
            base=cal[None,:,0,None]-cal[None,:,1,None]*c['utility']
            residual=np.load(root/f'runs/{kind}_seed{seed}/tune_epoch_{epoch}.npz')['residual']*cal[None,:,2,None]
            predictions.append(residual);baselines.append(base)
            dest=root/f'selected/{kind}_seed{seed}';dest.mkdir(parents=True,exist_ok=False)
            shutil.copyfile(root/f'runs/{kind}_seed{seed}/epoch_{epoch}.pt',dest/'model.pt')
        gammas=[];chosen_val=[];base_val=[]
        for ki,k in enumerate((3,6,9)):
            candidates=[]
            for gamma in (0.,.25,.5,1.):
                values=[]
                for base,residual in zip(baselines,predictions):
                    chosen=(base[:,ki]+gamma*residual[:,ki]).argmin(-1)
                    values.append(tune[np.arange(len(tune)),ki,chosen].mean())
                mean=float(np.mean(values));candidates.append((mean,gamma))
                rows.append(dict(kind=kind,epoch=epoch,keep=k,gamma=gamma,chosen_kl=mean))
            mean,gamma=min(candidates);gammas.append(gamma);chosen_val.append(mean);base_val.append(candidates[0][0])
        selection[kind]=dict(epoch=epoch,gamma=gammas,tune_kl=chosen_val,base_tune_kl=base_val)
    proposed=selection['cfs'];eligible=bool(proposed['tune_kl'][2]<proposed['base_tune_kl'][2]-1e-8 and
        all(proposed['tune_kl'][k]<=proposed['base_tune_kl'][k]+1e-8 for k in (0,1)))
    result=dict(variants=selection,primary='cfs' if eligible else 'A3',eligible=eligible,
        test_used=False,confirmation_used=False,selection_metric='Tune only; identical policy for every variant')
    (root/'selection_lock.json').write_text(json.dumps(result,indent=2));pd.concat(all_history).to_csv(root/'training_history.csv',index=False)
    pd.DataFrame(rows).to_csv(root/'calibration_choices.csv',index=False)
    for path,sha in json.loads((root/'source_hashes.json').read_text()).items():assert digest(path)==sha
    (root/'TRAINING_COMPLETE').write_text('All frozen source hashes unchanged\n')
    print(json.dumps(result,indent=2),flush=True)


def diagnostic(root):
    assert (root/'selection_lock.json').exists();lock_data=json.loads((root/'selection_lock.json').read_text());rows=[]
    held=np.load(root/'cache/mask_split.npz')['held']
    output=root/'prediction_diagnostics';output.mkdir(exist_ok=False)
    for split in ('train','tune'):
        for seed in SEEDS:
            a,cal=arrays(root,split,seed,held if split=='train' else None)
            for kind in KINDS:
                payload=torch.load(root/f'selected/{kind}_seed{seed}/model.pt',map_location='cpu',weights_only=False)
                model=SetResidual(kind).cuda().eval();model.load_state_dict(payload['state_dict'])
                parts=[]
                with torch.inference_mode():
                    for ids in torch.arange(len(a['encoded']),device='cuda').split(32):parts.append(predict(model,a,ids).cpu().numpy())
                raw=np.concatenate(parts);base=a['base'].cpu().numpy().astype(np.float64)
                truth=a['kl'].cpu().numpy().astype(np.float64)
                for mode,gamma in [('uncalibrated',np.ones(3)),('selected',np.array(lock_data['variants'][kind]['gamma']))]:
                    prediction=base+raw*cal[None,:,2,None]*gamma[None,:,None]
                    np.savez_compressed(output/f'{split}_{kind}_seed{seed}_{mode}.npz',prediction=prediction,truth=truth,baseline=base)
                    for ki,k in enumerate((3,6,9)):
                        for n in range(len(truth)):
                            y=truth[n,ki];pred=prediction[n,ki];b=base[n,ki]
                            rows.append(dict(split=split,seed=seed,kind=kind,mode=mode,sample=n,keep=k,
                                mse=np.mean((pred-y)**2),mae=np.mean(abs(pred-y)),base_mse=np.mean((b-y)**2),
                                regret=y[pred.argmin()]-y.min(),base_regret=y[b.argmin()]-y.min()))
    pd.DataFrame(rows).to_csv(output/'per_input.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','train','lock','diagnostic','all'],default='all');p.add_argument('--kind',choices=KINDS);p.add_argument('--seed',type=int)
    args=p.parse_args();torch.set_num_threads(4);root=args.root.resolve()
    if args.stage in ('all','prepare'):prepare(root)
    if args.stage in ('all','train'):
        for kind in ((args.kind,) if args.kind else KINDS):
            for seed in ((args.seed,) if args.seed else SEEDS):train(root,kind,seed)
    if args.stage in ('all','lock'):lock(root)
    if args.stage in ('all','diagnostic'):diagnostic(root)


if __name__=='__main__':main()
