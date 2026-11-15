"""Ten-epoch, low-LR continuation from immutable A3 checkpoints."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.dynamic_cfs_router import DynamicCFSRouter
from src.cfs_regularization import gradient_diagnostics
from src.budget_aware_objective import weighted_objective
from scripts.train_cfs_regularization import losses


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weight9',type=float,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=10);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    source=args.source;start_path=source/f'runs/A3_lambda0.01_seed{args.seed}/model.pt'
    payload=torch.load(start_path,map_location='cpu',weights_only=False)
    mean,std=payload['global_mean'],payload['global_std']
    def load(split):
        data=dict(np.load(source/f'baseline/learned_gating_inputs_20261115/{split}.npz'))
        aux=dict(np.load(source/f'inputs/{split}_aux.npz'));np.testing.assert_array_equal(data['paths'],aux['paths'])
        if split=='train':
            np.testing.assert_array_equal(mean,data['feature'].mean(0));np.testing.assert_array_equal(std,data['feature'].std(0).clip(min=1e-5))
        return [torch.from_numpy(a) for a in [(data['feature']-mean)/std,data['masks'].astype(bool),data['kl'].astype(np.float32),
                np.nan_to_num(aux['scores'],nan=-1.).clip(-1,1),aux['drops'],np.nan_to_num(aux['alpha'],nan=1.).clip(0,3)]]
    train=load('train');tune=[v.cuda() for v in load('tune')];probe=[v[:64].cuda() for v in train]
    model=DynamicCFSRouter(2304,12).cuda();model.load_state_dict(payload['state_dict'])
    assert sum(p.numel() for p in model.parameters())==825091
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(lr=3e-5,weight_decay=1e-4,batch=64,optimizer='fresh AdamW',parameters=825091,
                  source_sha256=hashlib.sha256(start_path.read_bytes()).hexdigest(),source_best_epoch=payload['best_epoch'],
                  normalized_weights=[1/(2+args.weight9),1/(2+args.weight9),args.weight9/(2+args.weight9)],
                  auxiliary_strength=.01,test_loaded=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(*train),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    common=list(model.global_encoder.parameters())+list(model.head_embedding.parameters())
    histories=[];gradients=[];diagnostics=[];started=time.perf_counter()
    for epoch in range(args.epochs+1):
        averages={}
        if epoch:
            model.train();records=[]
            for raw in loader:
                batch=[v.cuda() for v in raw];optimizer.zero_grad(set_to_none=True)
                terms,_=losses(model,batch);loss=weighted_objective(terms,args.weight9)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();optimizer.step();records.append({k:float(v.detach()) for k,v in terms.items()})
            averages={f'train_{k}':float(np.mean([r[k] for r in records])) for k in records[0]}
        model.eval()
        with torch.inference_mode():
            _,u=losses(model,tune);chosen=tune[2].gather(-1,u.argmax(-1,keepdim=True)).squeeze(-1)
            regrets=(chosen-tune[2].min(-1).values).mean(0).cpu().numpy()
        histories.append(dict(epoch=epoch,weight9=args.weight9,seed=args.seed,val_joint_regret=float(regrets.mean()),
                              **{f'val_K{k}':float(regrets[i]) for i,k in enumerate((3,6,9))},**averages))
        if epoch:
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},global_mean=mean,global_std=std,
                            config=config,best_epoch=epoch,val_joint_regret=float(regrets.mean())),args.output/f'epoch_{epoch:02d}.pt')
        if epoch in (0,1,5,10):
            terms,_=losses(model,probe)
            measured={k:terms[k] for k in ('joint_K3','joint_K6','joint_K9','cfs','alpha')}
            measured['weighted_joint']=(terms['joint_K3']+terms['joint_K6']+args.weight9*terms['joint_K9'])/(2+args.weight9)
            norms,rows=gradient_diagnostics(measured,common)
            gradients.extend(dict(epoch=epoch,**r) for r in rows)
            for name,value in measured.items():
                weight={'joint_K3':1/(2+args.weight9),'joint_K6':1/(2+args.weight9),'joint_K9':args.weight9/(2+args.weight9),'cfs':.01,'alpha':.001,'weighted_joint':1}[name]
                diagnostics.append(dict(epoch=epoch,loss=name,value=float(value.detach()),gradient_norm=norms[name],
                                        objective_coefficient=.5*weight,weighted_gradient_norm=.5*weight*norms[name]))
        pd.DataFrame(histories).to_csv(args.output/'training_history.csv',index=False)
        pd.DataFrame(gradients).to_csv(args.output/'gradients.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(args.output/'probe_losses.csv',index=False)
        print(f'w9={args.weight9} seed={args.seed} epoch={epoch} val={regrets.tolist()} seconds={time.perf_counter()-started:.1f}',flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(epochs=args.epochs,updates=args.epochs*64,seconds=time.perf_counter()-started)))


if __name__=='__main__':main()
