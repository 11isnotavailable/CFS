"""Locked three-seed test; uniform hard gating, NLL and online route checks."""
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.gpt2_routing import GPT2Suffix,GPT2RuntimeRouter,output_kl
from src.dynamic_cfs_router import DynamicCFSRouter,subset_masks
from src.learned_head_gating import LearnedHeadGating,topk_mask
from scripts.train_gpt2_routing import load_backbone,selected


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,required=True);args=p.parse_args();root=args.root;layer=args.layer
    lock=root/f'layer{layer}/selection_lock.json';assert lock.exists() and (root/f'layer{layer}/TRAINING_COMPLETE').exists()
    out=root/f'layer{layer}/evaluation';out.mkdir(exist_ok=False);torch.set_num_threads(4)
    data=np.load(root/'data/test.npz');model=load_backbone(root);engine=GPT2Suffix(model,layer)
    routers={};catalogs={k:subset_masks(12,k,device='cuda') for k in (3,6,9)};payloads={};timings=[];hashes={}
    for kind in ('R2','Gating','A0'):
        for seed in (17,23,41):
            path=root/f'layer{layer}/full/runs/{kind}_seed{seed}/model.pt';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            payload=torch.load(path,map_location='cpu',weights_only=False);payloads[kind,seed]=payload
            router=(LearnedHeadGating() if kind=='Gating' else DynamicCFSRouter(2304,12)).cuda().eval().requires_grad_(False);router.load_state_dict(payload['state_dict']);routers[kind,seed]=router
    rows=[];dense_rows=[];mask_rows=[];checks=[];features=[];dense_logits=[];started=time.perf_counter()
    for sample,tokens in enumerate(data['input_ids']):
        ids=torch.tensor(tokens[:-1],device='cuda')[None];target=int(tokens[-1]);reference,feature=engine.capture(ids)
        features.append(feature.cpu().numpy());dense_logits.append(reference.cpu().numpy());target_t=torch.tensor([target],device='cuda')
        dense_nll=float(torch.nn.functional.cross_entropy(reference,target_t));dense_rows.append(dict(sample=sample,document_id=str(data['document_id'][sample]),document_sha256=str(data['document_sha256'][sample]),target=target,nll=dense_nll))
        torch.cuda.synchronize();start=time.perf_counter();taylor=engine.taylor(target);torch.cuda.synchronize()
        timings.append(dict(sample=sample,method='Taylor',seed=0,seconds=time.perf_counter()-start,scope='supervised suffix backward; dense capture excluded'))
        scales=[];keys=[]
        with torch.inference_mode():
            for seed in (17,23,41):
                for k in catalogs:scales.append(topk_mask(taylor[None],k)[0]);keys.append(('Taylor',seed,k))
            for (kind,seed),router in routers.items():
                payload=payloads[kind,seed];f=(feature-torch.tensor(payload['global_mean'],device='cuda'))/torch.tensor(payload['global_std'],device='cuda')
                torch.cuda.synchronize();start=time.perf_counter();choice=selected(router,f,kind,catalogs);torch.cuda.synchronize()
                timings.append(dict(sample=sample,method=kind,seed=seed,seconds=time.perf_counter()-start,scope='feature normalization excluded; all three budgets, exact search'))
                for k,mask in choice.items():scales.append(mask[0]);keys.append((kind,seed,k))
            scales=torch.stack(scales).float();logits=engine.suffix(scales);kl=output_kl(reference,logits);nll=-logits.float().log_softmax(-1)[:,target]
            assert torch.isfinite(kl).all() and torch.isfinite(nll).all()
            for index,(kind,seed,k) in enumerate(keys):
                rows.append(dict(layer=layer,sample=sample,document_id=str(data['document_id'][sample]),position=127,target=target,method=kind,seed=seed,keep=k,
                    kl=float(kl[index]),nll=float(nll[index]),dense_nll=dense_nll,nll_difference=float(nll[index])-dense_nll))
                mask_rows.append(scales[index].cpu().numpy())
            # Verify deployment route uses only ln_1 prefix features, not dense or teacher.
            if sample<4:
                for index,(kind,seed,k) in enumerate(keys):
                    if kind=='Taylor' or seed!=17:continue
                    live=GPT2RuntimeRouter(model,layer,payloads[kind,seed],kind,k);online=live.next_logits(ids)
                    torch.testing.assert_close(live.last_mask.float(),scales[index:index+1],atol=0,rtol=0)
                    error=float(abs(online[0]-logits[index]).max())
                    # Logits have large offsets (up to hundreds); use the same
                    # scale-aware FP32 tolerance independently checked on train.
                    torch.testing.assert_close(online[0],logits[index],atol=1e-4,rtol=1e-5)
                    kl_error=float(abs(output_kl(reference,online)[0]-kl[index]));assert kl_error<2e-5
                    checks.append(dict(sample=sample,kind=kind,keep=k,online_max_error=error,online_kl_error=kl_error,
                        atol=1e-4,rtol=1e-5));live.close()
        if (sample+1)%32==0:print('test',layer,sample+1,'elapsed',time.perf_counter()-started,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_input.csv',index=False);frame.to_json(out/'per_input.json',orient='records')
    pd.DataFrame(dense_rows).to_csv(out/'dense_per_input.csv',index=False);pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    np.savez_compressed(out/'predictions.npz',masks=np.stack(mask_rows),feature=np.concatenate(features),dense_logits=np.concatenate(dense_logits),document_sha256=data['document_sha256'])
    for path,sha in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    check=dict(samples=256,rows=len(frame),selection_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        online_checks=checks,checkpoint_hashes=hashes,backbone_trained=False,shared_hard_gating=True,taylor_uses_target=True,seconds=time.perf_counter()-started)
    (out/'checks.json').write_text(json.dumps(check,indent=2));engine.close()


if __name__=='__main__':main()
