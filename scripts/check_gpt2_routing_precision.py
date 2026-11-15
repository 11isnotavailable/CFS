"""Audit batched versus live routing on TRAIN prefixes, without model selection."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.train_gpt2_routing import load_backbone,selected
from src.gpt2_routing import GPT2Suffix,GPT2RuntimeRouter,output_kl
from src.dynamic_cfs_router import DynamicCFSRouter,subset_masks
from src.learned_head_gating import LearnedHeadGating


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(4);model=load_backbone(a.root);engine=GPT2Suffix(model,a.layer);data=np.load(a.root/'data/train.npz');rows=[]
    catalogs={k:subset_masks(12,k,device='cuda') for k in (3,6,9)}
    with torch.inference_mode():
        for sample in range(32):
            ids=torch.tensor(data['input_ids'][sample,:-1],device='cuda')[None];ref,feature=engine.capture(ids)
            for kind in ('R2','Gating','A0'):
                payload=torch.load(a.root/f'layer{a.layer}/full/runs/{kind}_seed17/model.pt',map_location='cpu',weights_only=False)
                router=(LearnedHeadGating() if kind=='Gating' else DynamicCFSRouter(2304,12)).cuda().eval();router.load_state_dict(payload['state_dict'])
                f=(feature-torch.tensor(payload['global_mean'],device='cuda'))/torch.tensor(payload['global_std'],device='cuda')
                choices=selected(router,f,kind,catalogs)
                for k,mask in choices.items():
                    scales=mask.float();batched=engine.suffix(scales.expand(36,-1));single=engine.suffix(scales)
                    live=GPT2RuntimeRouter(model,a.layer,payload,kind,k);online=live.next_logits(ids);assert torch.equal(live.last_mask,mask);live.close()
                    row=dict(sample=sample,kind=kind,keep=k,batch_single=float(abs(batched[0]-single[0]).max()),single_online=float(abs(single-online).max()),batch_online=float(abs(batched[0]-online[0]).max()),
                        kl_difference=float(abs(output_kl(ref,batched[:1])-output_kl(ref,online)).max()),logit_scale=float(online.abs().max()))
                    torch.testing.assert_close(batched[:1],online,atol=1e-4,rtol=1e-5)
                    rows.append(row)
    engine.close();(a.root/f'precision_layer{a.layer}.json').write_text(json.dumps(rows,indent=2))
    print({key:max(r[key] for r in rows) for key in ('batch_single','single_online','batch_online','kl_difference','logit_scale')})


if __name__=='__main__':main()
