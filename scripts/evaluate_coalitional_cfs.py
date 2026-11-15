"""Locked unseen-input prediction/ranking and clearly labeled pair-info controls."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.coalitional_cfs import RelationPredictor,PairInformationPredictor
from src.coalitional_metrics import input_metrics
from scripts.train_coalitional_cfs import arrays,name


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pool-seed',type=int,default=42);p.add_argument('--evaluation-budget',action='store_true');a=p.parse_args();root=a.root
    suffix='_evaluations' if a.evaluation_budget else ''
    lock=json.loads((root/f'selection_lock{suffix}.json').read_text());out=root/f'evaluation/pool{a.pool_seed}{suffix}';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;data=arrays(root,'test',a.pool_seed);rows=[];predictions={};configs=[]
    def record(pred,kind,job,constant):
        truth=data['pair' if kind=='pair' else 'joint'];pred=pred.reshape(truth.shape)
        metrics=input_metrics(truth,pred,data['valid'],constant)
        for i in range(len(truth)):
            rows.append(dict(pool=a.pool_seed,sample=i,sha256=str(data['sha256'][i]),method=kind,curve=job['curve'],
                budget=job['nominal_budget'],labels=job['labels'],seed=job['seed'],**{k:float(v[i]) for k,v in metrics.items()}))
    with torch.inference_mode():
        for job in json.loads((root/f'training_plan{suffix}.json').read_text())['jobs']:
            run=root/'runs'/name(job);path=run/'model.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['models'][name(job)]['sha256']
            p=torch.load(path,map_location='cpu',weights_only=False);kind=job['kind'];actual='joint' if kind=='pair_info' else kind
            model=(PairInformationPredictor() if kind=='pair_info' else RelationPredictor()).cuda().eval();model.load_state_dict(p['state_dict'])
            rel=data[f'{actual}_rel'][0]
            if actual=='pair':rel=np.concatenate((rel,rel[:,-1:]),1)
            rel=torch.tensor(rel,device='cuda');values=[]
            f=data['pair_info'] if kind=='pair_info' else (data['feature']-p['global_mean'])/p['global_std']
            for x in torch.tensor(f,device='cuda').split(64):values.append(model(x) if kind=='pair_info' else model(x,rel[None].expand(len(x),-1,-1)))
            pred=(torch.cat(values)*p['target_std']+p['target_mean']).cpu().numpy();predictions[name(job)]=pred;record(pred,kind,job,p['target_mean']);configs.append(p['config'])
        # Information diagnostics: true pair scores are privileged test measurements.
        joint_constant=float(np.mean([c['target_mean'] for c in configs if c['kind']=='joint' and c['labels']==max(x['labels'] for x in configs)]))
        for kind,pred in ([] if a.evaluation_budget else [('oracle_pair_max',data['pair_info'].max(-1)),('oracle_pair_mean',data['pair_info'].mean(-1))]):
            record(pred,kind,dict(curve='information',nominal_budget=20000,labels=20000,seed=0),joint_constant);predictions[kind]=pred
        combos=np.array(list(__import__('itertools').combinations(range(4),2)))
        for seed in (() if a.evaluation_budget else (17,23,41)):
            for budget in (2500,5000,10000,20000):
                pair=predictions[f'pair_labels_{budget}_seed{seed}'].reshape(len(data['feature']),12,4)
                pair=pair[:,:,combos].reshape(len(pair),72,2)
                for operation in ('max','mean'):
                    pred=getattr(pair,operation)(-1);kind=f'predicted_pair_{operation}'
                    record(pred,kind,dict(curve='pair_transfer',nominal_budget=budget,labels=budget,seed=seed),joint_constant)
                    predictions[f'{kind}_{budget}_seed{seed}']=pred
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_input.csv',index=False);np.savez_compressed(out/'predictions.npz',**predictions)
    (out/'checks.json').write_text(json.dumps(dict(inputs=len(data['feature']),rows=len(frame),oracle_alpha_conditioned=True,
        source_lock_sha256=hashlib.sha256((root/f'selection_lock{suffix}.json').read_bytes()).hexdigest(),test_teacher_not_used_for_selection=True),indent=2))
    print('evaluated',a.pool_seed,len(frame))


if __name__=='__main__':main()
