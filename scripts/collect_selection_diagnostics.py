"""Read-only frozen-router predictions; no backbone or optimizer is constructed."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.dynamic_cfs_router import DynamicCFSRouter,candidate_subset_utilities,subset_masks,subset_utilities
from src.learned_head_gating import LearnedHeadGating,additive_utilities,topk_mask,assert_disjoint
from src.selection_diagnostics import mask_codes,ranking_metrics,union_oracle


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-inputs',type=int);p.add_argument('--device',default='cuda');args=p.parse_args()
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    torch.set_num_threads(4);started=time.perf_counter();hashes={};metric_rows=[];coverage_rows=[];cross_rows=[];hard_rows=[];checks=[]
    def checkpoint(method,seed):
        if method=='A3':return root/f'source/runs/A3_lambda0.01_seed{seed}/model.pt'
        if method=='R2':return root/f'baseline/learned_gating_inputs_20261115/r2_seed{seed}.pt'
        kind='gating' if method=='Gating' else 'joint_only_r2'
        return root/f'baseline/runs/{kind}_seed{seed}/model.pt'
    data_root=root/'baseline/learned_gating_inputs_20261115'
    assert_disjoint(np.load(data_root/'train.npz')['paths'],np.load(data_root/'tune.npz')['paths'])
    catalogs={k:subset_masks(12,k,device=args.device) for k in (3,6,9)}
    codes={k:mask_codes(v.cpu().numpy()) for k,v in catalogs.items()}
    for split in ('train','tune'):
        path=data_root/f'{split}.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();data=dict(np.load(path))
        if args.max_inputs:data={key:v[:args.max_inputs] for key,v in data.items()}
        N=len(data['paths']);pool_codes=mask_codes(data['masks'])
        assert data['kl'].shape==(N,3,48) and np.isfinite(data['kl']).all()
        assert all(len(np.unique(c))==48 for c in pool_codes.reshape(-1,48))
        np.savez_compressed(out/f'labels_{split}.npz',**{k:v for k,v in data.items() if k!='feature'})
        for seed in (17,23,41):
            predictions={};fullchoices={};orders={}
            for method in ('A3','R2','Gating','A0'):
                path=checkpoint(method,seed);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                payload=torch.load(path,map_location='cpu',weights_only=False)
                model=(LearnedHeadGating() if method=='Gating' else DynamicCFSRouter(2304,12)).to(args.device).eval().requires_grad_(False)
                model.load_state_dict(payload['state_dict']);parts=[];fullparts=[];topcodes=[];topscores=[];scoregaps=[];consistency=0.
                with torch.inference_mode():
                    for begin in range(0,N,64):
                        stop=min(begin+64,N);features=torch.tensor((data['feature'][begin:stop]-payload['global_mean'])/payload['global_std'],device=args.device)
                        result=model(features);u=[];choices=[];tc=[];ts=[];gaps=[]
                        for ki,k in enumerate((3,6,9)):
                            local=torch.tensor(data['masks'][begin:stop,ki],device=args.device).bool()
                            if method=='Gating':
                                pool=additive_utilities(result,local);full=result@catalogs[k].float().T
                                chosen=mask_codes(topk_mask(result,k).cpu().numpy())
                            else:
                                scores,importance,_=result;weights=importance.softmax(-1)
                                pool=candidate_subset_utilities(scores,weights,local);full=subset_utilities(scores,weights,catalogs[k])
                                chosen=codes[k][full.argmax(-1).cpu().numpy()]
                            f=full.cpu().numpy();pv=pool.cpu().numpy();order=np.argsort(-f,axis=-1,kind='stable')[:,:20]
                            u.append(pv);choices.append(chosen);tc.append(codes[k][order]);ts.append(np.take_along_axis(f,order,-1))
                            gaps.append((f.max(-1)-pv.max(-1))/np.maximum(pv.std(-1),1e-8))
                            lookup={int(c):j for j,c in enumerate(codes[k])};indices=np.array([[lookup[int(c)] for c in row] for row in pool_codes[begin:stop,ki]])
                            consistency=max(consistency,float(abs(np.take_along_axis(f,indices,-1)-pv).max()))
                        parts.append(np.stack(u,1));fullparts.append(np.stack(choices,1));topcodes.append(np.stack(tc,1));topscores.append(np.stack(ts,1));scoregaps.append(np.stack(gaps,1))
                utility=np.concatenate(parts);fullchoice=np.concatenate(fullparts);gaps=np.concatenate(scoregaps)
                predictions[method]=utility;fullchoices[method]=fullchoice;orders[method]=[]
                np.savez_compressed(out/f'predictions/{split}_{method}_seed{seed}.npz',pool_utility=utility,
                    full_choice_code=fullchoice,full_top20_codes=np.concatenate(topcodes),full_top20_scores=np.concatenate(topscores))
                checks.append(dict(split=split,seed=seed,method=method,inputs=N,pool_vs_full_score_max_error=consistency,parameters=sum(p.numel() for p in model.parameters())))
                assert consistency<2e-5
                for ki,k in enumerate((3,6,9)):
                    metrics,order=ranking_metrics(utility[:,ki],data['kl'][:,ki]);orders[method].append(order)
                    known=pool_codes[:,ki]==fullchoice[:,ki,None];has=known.any(-1);index=known.argmax(-1)
                    for n in range(N):
                        metric_rows.append(dict(split=split,seed=seed,method=method,keep=k,sample=n,**{key:float(v[n]) for key,v in metrics.items()}))
                        coverage_rows.append(dict(split=split,seed=seed,method=method,keep=k,sample=n,total_candidates=len(codes[k]),
                            labelled_candidates=48,full_choice_code=int(fullchoice[n,ki]),full_choice_known=bool(has[n]),
                            full_choice_kl=float(data['kl'][n,ki,index[n]]) if has[n] else np.nan,
                            score_gap_standardized=float(gaps[n,ki])))
                print('scored',split,method,seed,N,'elapsed',time.perf_counter()-started,flush=True)
            for ki,k in enumerate((3,6,9)):
                truth=data['kl'][:,ki];oracle=truth.min(-1);chosen={m:orders[m][ki][:,0] for m in predictions}
                loss={m:truth[np.arange(N),v] for m,v in chosen.items()}
                union={size:union_oracle([predictions[m][:,ki] for m in ('A3','R2','Gating')],truth,size) for size in (1,3,5)}
                a3mask=data['masks'][np.arange(N),ki,chosen['A3']];r2mask=data['masks'][np.arange(N),ki,chosen['R2']]
                for n in range(N):
                    row=dict(split=split,seed=seed,keep=k,sample=n,oracle_kl=float(oracle[n]),
                        A3_R2_pool_disagree=float(chosen['A3'][n]!=chosen['R2'][n]),
                        A3_Gating_pool_disagree=float(chosen['A3'][n]!=chosen['Gating'][n]),
                        A3_R2_swaps=float(np.count_nonzero(a3mask[n]!=r2mask[n])/2),
                        A3_R2_full_disagree=float(fullchoices['A3'][n,ki]!=fullchoices['R2'][n,ki]),
                        A3_R2_full_swaps=float((int(fullchoices['A3'][n,ki])^int(fullchoices['R2'][n,ki])).bit_count()/2))
                    row.update({f'{m}_kl':float(v[n]) for m,v in loss.items()})
                    row.update({f'union_top{size}_kl':float(v[n]) for size,v in union.items()});cross_rows.append(row)
                    if split=='train':
                        for origin in ('pool','full'):
                            for method in ('A3','R2','Gating'):
                                code=int(pool_codes[n,ki,chosen[method][n]]) if origin=='pool' else int(fullchoices[method][n,ki])
                                found=np.flatnonzero(pool_codes[n,ki]==code)
                                hard_rows.append(dict(sample=n,path=str(data['paths'][n]),dataset_index=int(data['dataset_index'][n]),seed=seed,keep=k,
                                    origin=origin,method=method,mask_code=code,label_known=bool(len(found)),
                                    measured_kl=float(truth[n,found[0]]) if len(found) else np.nan,
                                    full_A3_R2_disagree=row['A3_R2_full_disagree'],full_A3_R2_swaps=row['A3_R2_full_swaps']))
    for name,rows in [('per_input',metric_rows),('coverage',coverage_rows),('cross_method',cross_rows),('training_candidate_inventory',hard_rows)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2));(out/'checks.json').write_text(json.dumps(checks,indent=2))
    (out/'metadata.json').write_text(json.dumps(dict(training_run=False,backbone_run=False,new_labels_collected=False,
        test_loaded=False,confirmation_loaded=False,source_hashes_unchanged=True,seeds=[17,23,41],budgets=[3,6,9],
        masks_per_budget=48,seconds=time.perf_counter()-started,max_inputs=args.max_inputs),indent=2))


if __name__=='__main__':main()
