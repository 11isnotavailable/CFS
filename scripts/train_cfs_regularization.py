"""Validation-only auxiliary supervision ablations on the unchanged R2 model."""
import argparse,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from src.dynamic_cfs_router import DynamicCFSRouter,candidate_subset_utilities
from src.learned_head_gating import joint_ranking_loss,assert_disjoint
from src.cfs_regularization import auxiliary_losses,gradient_diagnostics


def losses(model,batch):
    feature,masks,kl,target,drops,target_alpha=batch
    score,importance,alpha=model(feature)
    utilities=torch.stack([candidate_subset_utilities(score,importance.softmax(-1),masks[:,k]) for k in range(3)],1)
    terms=auxiliary_losses(score,importance,alpha,target,drops,target_alpha)
    terms['joint']=joint_ranking_loss(utilities,kl)
    for idx,k in enumerate((3,6,9)):terms[f'joint_K{k}']=joint_ranking_loss(utilities[:,idx],kl[:,idx])
    return terms,utilities


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--aux',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['A1','A2','A3'],default='A1')
    p.add_argument('--strength',type=float,default=.1);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--diagnose-checkpoint',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    train,tune=[dict(np.load(args.data/f'{s}.npz')) for s in ('train','tune')]
    assert_disjoint(train['paths'],tune['paths'])
    mean=train['feature'].mean(0);std=train['feature'].std(0).clip(min=1e-5)
    def tensors(data,split):
        aux=np.load(args.aux/f'{split}_aux.npz');np.testing.assert_array_equal(data['paths'],aux['paths'])
        return [torch.from_numpy(a) for a in [(data['feature']-mean)/std,data['masks'].astype(bool),data['kl'].astype(np.float32),
            np.nan_to_num(aux['scores'],nan=-1.).clip(-1,1),aux['drops'],np.nan_to_num(aux['alpha'],nan=1.).clip(0,3)]]
    ts=tensors(train,'train');vs=[t.cuda() for t in tensors(tune,'tune')]
    loader=DataLoader(TensorDataset(*ts),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    probe=[v[:64].cuda() for v in ts];model=DynamicCFSRouter(2304,12).cuda()
    model.alpha_output.requires_grad_(args.variant=='A3')
    common=list(model.global_encoder.parameters())+list(model.head_embedding.parameters())
    history,gradient_rows,probe_rows=[],[],[]
    def diagnose(epoch):
        model.eval();terms,_=losses(model,probe)
        measured={k:terms[k] for k in ['joint','cfs','deletion','alpha','joint_K3','joint_K6','joint_K9']}
        if args.diagnose_checkpoint:
            from scripts.train_dynamic_cfs_router import loss_terms
            from src.dynamic_cfs_router import subset_masks
            score,importance,alpha=model(probe[0])
            original=loss_terms(score,importance,alpha,probe[3],probe[4],probe[5],
                {k:subset_masks(12,k,device='cuda') for k in (3,6,9)},.1,.05)
            measured['original_structured']=original['structured']
            measured['original_auxiliary_total']=terms['cfs']+.5*terms['deletion']+.1*terms['alpha']+original['structured']
        norms,rows=gradient_diagnostics(measured,common)
        gradient_rows.extend(dict(epoch=epoch,**row) for row in rows)
        probe_rows.extend(dict(epoch=epoch,loss=k,value=float(v.detach()),grad_norm=norms.get(k)) for k,v in terms.items())
        pd.DataFrame(gradient_rows).to_csv(args.output/'gradients.csv',index=False)
        pd.DataFrame(probe_rows).to_csv(args.output/'probe_losses.csv',index=False)
    if args.diagnose_checkpoint:
        payload=torch.load(args.diagnose_checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['state_dict']);diagnose(-1);return
    config={**vars(args),'epochs':args.epochs,'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'batch':64,
            'formula':'.5*(joint+lambda*(cfs+[A2]*.5*deletion+[A3]*.1*alpha))',
            'test_loaded':False,'parameters':sum(x.numel() for x in model.parameters()),
            'optimized_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad)}
    (args.output/'config.json').write_text(json.dumps(config,default=str,indent=2))
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=3e-4,weight_decay=1e-4)
    best=float('inf');start=time.perf_counter();diagnose(0)
    for epoch in range(1,args.epochs+1):
        model.train();totals=[]
        for raw in loader:
            batch=[v.cuda() for v in raw];optimizer.zero_grad(set_to_none=True)
            terms,_=losses(model,batch)
            aux=terms['cfs']
            if args.variant=='A2':aux=aux+.5*terms['deletion']
            if args.variant=='A3':aux=aux+.1*terms['alpha']
            loss=.5*(terms['joint']+args.strength*aux)
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();optimizer.step();totals.append({k:float(v.detach()) for k,v in terms.items()})
        model.eval()
        with torch.inference_mode():
            _,u=losses(model,vs);chosen=vs[2].gather(-1,u.argmax(-1,keepdim=True)).squeeze(-1)
            regrets=(chosen-vs[2].min(-1).values).mean(0).cpu().numpy()
        row=dict(epoch=epoch,val_joint_regret=float(regrets.mean()),**{f'val_K{k}':float(regrets[i]) for i,k in enumerate((3,6,9))},
                 **{f'train_{k}':float(np.mean([t[k] for t in totals])) for k in totals[0]})
        history.append(row)
        if row['val_joint_regret']<best:
            best=row['val_joint_regret']
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},global_mean=mean,global_std=std,
                config=config,best_epoch=epoch,val_joint_regret=best),args.output/'model.pt')
        pd.DataFrame(history).to_csv(args.output/'training_history.csv',index=False)
        if epoch in (1,10,30,60):diagnose(epoch)
        if epoch==1 or epoch%10==0:print(f'{args.variant} lambda={args.strength} seed={args.seed} epoch={epoch} val={row["val_joint_regret"]:.6f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(best_validation=best,seconds=time.perf_counter()-start)))


if __name__=='__main__':main()
