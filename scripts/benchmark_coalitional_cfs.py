"""Eight train inputs: intervention boundaries and honest batched oracle costs."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from timm.data import create_transform,resolve_model_data_config
from scripts.evaluate_dynamic_cfs_execution import FrozenPathDataset,dense_kl
from src.model_loading import create_timm_model
from src.coalitional_cfs import LastViTSuffix,candidate_pool,relations,scales_for,grid_oracle,gradient_oracle


def load(root):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=Path('/data/cfs_learned_gating_20261115/base.pth')
    model,_=create_timm_model('vit_base_patch16_224.augreg_in21k_ft_in1k',checkpoint)
    model.cuda().eval().requires_grad_(False);engine=LastViTSuffix(model)
    transform=create_transform(**resolve_model_data_config(model),is_training=False)
    return model,engine,transform


def dataset(root,split,transform):
    data=np.load(root/f'data/{split}.npz')
    return data,FrozenPathDataset(np.array([str(root/'images'/str(p)) for p in data['paths']]),data['target'],transform)


def pair_initial(pool,pair_alpha):
    values=[]
    for source,row in enumerate(pool):
        for j in range(4):
            for k in range(j+1,4):values.append([[float(pair_alpha[source*4+j,0]),1.],[1.,float(pair_alpha[source*4+k,0])],[1.,1.]])
    return torch.tensor(values,device=pair_alpha.device)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    out=root/'benchmark';out.mkdir(exist_ok=False);model,engine,transform=load(root);data,ds=dataset(root,'train',transform)
    pool=candidate_pool(42);pair=relations(pool);joint=relations(pool,True);rows=[];checks=[]
    def timed(name,fn,n):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();result=fn();torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        rows.append(dict(sample=n,method=name,seconds=seconds,labels=len(result[0]),seconds_per_label=seconds/len(result[0]),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,**result[2]))
        return result
    for n in range(8):
        image=ds[n][0][None].cuda();ref,feature=engine.capture(image)
        with torch.no_grad():
            scales=torch.ones(5,12,device='cuda');scales[1:,0]=0;scales[2,1]=0;scales[3,1]=2.3;scales[4,1:3]=torch.tensor([.7,2.1],device='cuda')
            fast=engine.suffix(scales);error=0.
            for i,s in enumerate(scales):
                engine.wrapper.set_head_scales(s[None]);direct=model(image);error=max(error,float(abs(direct-fast[i:i+1]).max()))
                torch.testing.assert_close(direct,fast[i:i+1],atol=3e-5,rtol=3e-5)
            engine.wrapper.set_head_scales(None)
            drop=engine.losses(1-torch.eye(12,device='cuda'))
            rel=torch.tensor([[0,1,2]],device='cuda');no_comp=engine.losses(scales_for(rel,torch.ones(1,2,device='cuda')))
            torch.testing.assert_close(no_comp,drop[:1],atol=2e-6,rtol=1e-4)
        p1=timed('pair_grid31',lambda:grid_oracle(engine,pair),n);j2=timed('joint_grid31x31',lambda:grid_oracle(engine,joint),n)
        init=pair_initial(pool,p1[1]);warm=init[:,:2].clone()
        with torch.no_grad():
            losses=engine.losses(scales_for(torch.tensor(joint,device='cuda')[:,None].expand(-1,2,-1).reshape(-1,3),warm.reshape(-1,2))).reshape(-1,2)
            idx=losses.argmin(-1);warm=warm[torch.arange(len(joint),device='cuda'),idx][:,None]
        outputs={'pair_grid31':p1,'joint_grid31x31':j2}
        for name,start,steps in [('gradient_ones60',init[:,2:],60),('gradient_pair60',warm,60),('gradient_multi60',init,60),('gradient_multi120',init,120)]:
            outputs[name]=timed(name,lambda:gradient_oracle(engine,joint,start,steps),n)
        for name,result in outputs.items():
            if name=='pair_grid31':continue
            excess=(result[0]-j2[0])/drop[torch.tensor(joint[:,0],device='cuda')].clamp_min(1e-12)
            valid=drop[torch.tensor(joint[:,0],device='cuda')]>1e-5
            rows[-(len(outputs)-list(outputs).index(name))].update(mean_recovery_excess=float(excess[valid].mean()),max_recovery_excess=float(excess[valid].max()))
        # Time single-label execution separately; do not equate relation count with time.
        timed('pair_serial_one',lambda:grid_oracle(engine,pair[:1]),n)
        timed('joint_serial_one',lambda:grid_oracle(engine,joint[:1]),n)
        checks.append(dict(sample=n,direct_max_error=error,valid_sources=int((drop>1e-5).sum()),features=list(feature.shape)))
        np.savez_compressed(out/f'input{n}.npz',drop=drop.cpu().numpy(),pair_rel=pair,joint_rel=joint,**{f'{name}_kl':v[0].detach().cpu().numpy() for name,v in outputs.items()},
            **{f'{name}_alpha':v[1].detach().cpu().numpy() for name,v in outputs.items()})
        print('pilot',n,'done',flush=True)
    pd.DataFrame(rows).to_csv(out/'costs.csv',index=False);(out/'checks.json').write_text(json.dumps(checks,indent=2));engine.close()
    print(pd.DataFrame(rows).groupby('method')[['seconds','seconds_per_label','mean_recovery_excess','max_recovery_excess']].mean().to_string())


if __name__=='__main__':main()
