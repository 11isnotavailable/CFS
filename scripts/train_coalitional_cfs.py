"""Matched-parameter/step label curves; fixed validation-only selection."""
import argparse,json,sys,time,random,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.coalitional_cfs import RelationPredictor,PairInformationPredictor


def arrays(root,split,pool=42):
    records=[]
    for path in sorted((root/f'teacher/pool{pool}/{split}').glob('[0-9][0-9][0-9][0-9][0-9].npz')):
        with np.load(path) as f:records.append({k:f[k] for k in ('feature','drop','pair_kl','joint_kl','pair_rel','joint_rel','sha256')})
    out={k:np.stack([r[k] for r in records]) for k in records[0]}
    out['valid']=out['drop']>1e-5
    for kind in ('pair','joint'):out[kind]=1-out[f'{kind}_kl']/np.maximum(out['drop'][...,None],1e-12)
    indices=np.array(list(__import__('itertools').combinations(range(4),2)))
    out['pair_info']=out['pair'][:,:,indices].reshape(len(records),72,2)
    return out


def plan(root,evaluation_budget=False):
    path=root/('training_plan_evaluations.json' if evaluation_budget else 'training_plan.json')
    if path.exists():return json.loads(path.read_text())
    c=pd.read_csv(root/'benchmark/costs.csv').query('sample>0')
    cost={k:float(c.loc[c.method==m,'seconds_per_label'].median()) for k,m in [('pair','pair_grid31'),('joint','joint_grid31x31')]}
    budgets=[2500,5000,10000,20000];jobs=[]
    for seed in (17,23,41):
        for budget in budgets:
            for kind,count,curve in [('pair',budget,'labels'),('joint',budget,'labels'),('joint',max(1,int(budget*cost['pair']/cost['joint'])),'cost')]:
                jobs.append(dict(kind=kind,labels=count,nominal_budget=budget,curve=curve,seed=seed,steps=2000))
        jobs.append(dict(kind='pair_info',labels=20000,nominal_budget=20000,curve='information',seed=seed,steps=2000))
    if evaluation_budget:
        jobs=[dict(kind='joint',labels=budget//31,nominal_budget=budget,curve='evaluations',seed=seed,steps=2000) for seed in (17,23,41) for budget in budgets]
    result=dict(cost=cost,jobs=jobs,batch=128,lr=3e-4,weight_decay=1e-4,selection='minimum validation per-input MSE every250 steps',
        budget_sampling_seed=20261116,test_loaded=False)
    path.write_text(json.dumps(result,indent=2));return result


def name(job):return f'{job["kind"]}_{job["curve"]}_{job["nominal_budget"]}_seed{job["seed"]}'


def fit(root,job,train,tune,pilot=False):
    out=root/('pilot' if pilot else 'runs')/name(job)
    if (out/'COMPLETE.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(out)
    seed=job['seed'];kind=job['kind'];actual='joint' if kind=='pair_info' else kind
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.use_deterministic_algorithms(True)
    width=4 if actual=='pair' else 6;train_y=train[actual].reshape(len(train['feature']),-1);val_y=tune[actual].reshape(len(tune['feature']),-1)
    valid=np.repeat(train['valid'],width,axis=1);order=np.random.default_rng(20261116).permutation(np.flatnonzero(valid.ravel()))
    chosen=order[:job['labels']];assert len(chosen)==job['labels'];image_id,rel_id=np.unravel_index(chosen,train_y.shape)
    labels=train_y[image_id,rel_id];ym=float(labels.mean());ys=max(float(labels.std()),1e-5)
    mean=train['feature'].mean(0);std=train['feature'].std(0).clip(1e-5)
    def rel(data):
        r=data[f'{actual}_rel'][0]
        if actual=='pair':r=np.concatenate((r,r[:,-1:]),axis=1)
        return torch.tensor(r,device='cuda')
    relation=rel(train)
    model=(PairInformationPredictor() if kind=='pair_info' else RelationPredictor()).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    tx=torch.tensor(train['pair_info'] if kind=='pair_info' else (train['feature']-mean)/std,device='cuda')
    vx=torch.tensor(tune['pair_info'] if kind=='pair_info' else (tune['feature']-mean)/std,device='cuda')
    ti=torch.tensor(image_id,device='cuda');tr=torch.tensor(rel_id,device='cuda');ty=torch.tensor((labels-ym)/ys,device='cuda')
    vy=torch.tensor(val_y,device='cuda');vv=torch.tensor(tune['valid'],device='cuda');history=[];best=float('inf');begin=time.perf_counter()
    config={**job,'parameters':sum(p.numel() for p in model.parameters()),'unique_training_inputs':len(np.unique(image_id)),
        'target_mean':ym,'target_std':ys,'normalization':'all train features, no validation/test','backbone_frozen':True,'loss':'standardized raw CFS MSE',
        'steps':250 if pilot else job['steps'],'batch':128,'lr':3e-4,'weight_decay':1e-4,'test_used':False}
    (out/'config.json').write_text(json.dumps(config,indent=2));np.savez_compressed(out/'label_indices.npz',flat_indices=chosen,image=image_id,relation=rel_id)
    for step in range(1,config['steps']+1):
        model.train();index=torch.randint(len(labels),(128,),device='cuda');optimizer.zero_grad(set_to_none=True)
        pred=model(tx[ti[index],tr[index]]) if kind=='pair_info' else model(tx[ti[index]],relation[tr[index]][:,None]).squeeze(-1)
        loss=(pred-ty[index]).square().mean();assert torch.isfinite(loss);loss.backward();optimizer.step()
        if step%250==0:
            model.eval()
            with torch.inference_mode():
                values=[]
                for x in vx.split(64):values.append(model(x) if kind=='pair_info' else model(x,relation[None].expand(len(x),-1,-1)))
                prediction=torch.cat(values)*ys+ym;error=(prediction-vy).square().reshape(len(vx),12,width).mean(-1)
                criterion=float(((error*vv).sum(-1)/vv.sum(-1).clamp_min(1)).mean())
            row=dict(step=step,loss=float(loss),validation_mse=criterion,seconds=time.perf_counter()-begin);history.append(row)
            payload=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},global_mean=mean,global_std=std,
                target_mean=ym,target_std=ys,config=config,step=step,validation_mse=criterion)
            torch.save(payload,out/f'step{step}.pt')
            if criterion<best:best=criterion;torch.save(payload,out/'model.pt')
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    (out/'COMPLETE.json').write_text(json.dumps(dict(best_validation_mse=best,seconds=time.perf_counter()-begin)))
    print('trained',name(job),'best',best,'seconds',time.perf_counter()-begin,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--pilot',action='store_true');p.add_argument('--evaluation-budget',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;root=a.root
    train,tune=arrays(root,'train'),arrays(root,'tune');cfg=plan(root,a.evaluation_budget)
    jobs=[j for j in cfg['jobs'] if j['kind'] in ('pair','joint') and j['curve']=='labels' and j['nominal_budget']==2500 and j['seed']==17] if a.pilot else cfg['jobs']
    for i,job in enumerate(jobs):
        if i%a.workers==a.worker:fit(root,job,train,tune,a.pilot)
    (root/f'{"PILOT" if a.pilot else "EVALUATIONS" if a.evaluation_budget else "WORKER"}_{a.worker}_COMPLETE').write_text('complete')


if __name__=='__main__':main()
