"""Reuse the seven ViT anchors and exact local-mutation candidate generator."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.dynamic_cfs_router import DynamicCFSRouter,subset_masks,selected_masks
from src.gpt2_routing import GPT2Suffix
from scripts.train_gpt2_routing import load_pair,load_backbone
from scripts.train_dynamic_cfs_router import choose_top,similarity_scores,static_masks_for_train
from scripts.collect_joint_subset_teacher import candidate_pool,ANCHORS,select_in_chunks


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,default=11)
    p.add_argument('--phase',choices=['pilot','full'],required=True);args=p.parse_args();root=args.root;torch.set_num_threads(4)
    N,V=(256,64) if args.phase=='pilot' else (2048,128)
    directory=root/f'layer{args.layer}/{args.phase}/joint';directory.mkdir(parents=True,exist_ok=True)
    path=root/f'layer{args.layer}/{args.phase}/runs/bootstrap_seed17/model.pt'
    config=dict(layer=args.layer,phase=args.phase,anchors=ANCHORS,candidates=48,seed=20260804,bootstrap_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if (directory/'config.json').exists():assert json.loads((directory/'config.json').read_text())==json.loads(json.dumps(config))
    else:(directory/'config.json').write_text(json.dumps(config,indent=2))
    payload=torch.load(path,map_location='cpu',weights_only=False);router=DynamicCFSRouter(2304,12).cuda().eval().requires_grad_(False);router.load_state_dict(payload['state_dict'])
    train,_=load_pair(root,args.layer,'train',N);static=static_masks_for_train(train,[3,6,9]);engine=GPT2Suffix(load_backbone(root),args.layer);started=time.perf_counter()
    for split,count in [('train',N),('tune',V)]:
        out=directory/split;out.mkdir(exist_ok=True);data,records=load_pair(root,args.layer,split,count)
        score=torch.tensor(np.nan_to_num(data['scores'],nan=-1.).clip(-1,1),device='cuda');drop=torch.tensor(data['drops'],device='cuda')
        with torch.inference_mode():
            s,w,_=router(torch.tensor((data['feature']-payload['global_mean'])/payload['global_std'],device='cuda'))
            anchors={}
            for k in (3,6,9):
                catalog=subset_masks(12,k,device='cuda')
                anchors[k]={name:value.cpu() for name,value in dict(
                    pairwise_cfs_oracle=select_in_chunks(score,drop,catalog),dynamic_cfs_router=select_in_chunks(s,w.softmax(-1),catalog),
                    deletion_oracle=choose_top(drop,k),taylor=choose_top(torch.tensor(data['taylor'],device='cuda'),k),
                    magnitude=choose_top(torch.tensor(data['norm'],device='cuda'),k),
                    current_l2=select_in_chunks(similarity_scores(data['l2']).cuda(),torch.ones_like(drop),catalog),
                    static_cfs=static[k][None].expand(count,-1)).items()}
            for i,r in enumerate(records):
                path=out/f'{i:05d}.npz'
                if path.exists():continue
                masks=[];indices=[]
                for k in (3,6,9):
                    m,a=candidate_pool(anchors[k],i,k,48,20260804);masks.append(m);indices.append(a)
                masks=np.stack(masks);engine.cache={key:torch.tensor(r[key],device='cuda') for key in ('residual','heads','reference')}
                kl=engine.evaluate(torch.tensor(masks.reshape(-1,12),device='cuda'),128).numpy().reshape(3,48)
                tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,masks=masks,kl=kl,anchor_indices=np.stack(indices),document_sha256=r['document_sha256']);tmp.replace(path)
                if (i+1)%64==0:print('joint',args.layer,args.phase,split,i+1,count,'elapsed',time.perf_counter()-started,flush=True)
    engine.close();(directory/'COMPLETE.json').write_text(json.dumps(dict(train=N,tune=V,seconds=time.perf_counter()-started)))


if __name__=='__main__':main()
