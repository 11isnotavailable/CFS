"""Audit historical feature drift, without training or selecting checkpoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from timm.data import create_transform,resolve_model_data_config
from torch.utils.data import DataLoader
from scripts.evaluate_dynamic_cfs_execution import FrozenPathDataset
from src.model_loading import create_timm_model
from src.head_intervention import make_intervenable
from src.runtime_cfs_router import prehead_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    data=np.load(args.data/'test.npz')
    model,_=create_timm_model('vit_base_patch16_224.augreg_in21k_ft_in1k',args.checkpoint)
    model.eval().requires_grad_(False).cuda().half()
    wrapper=make_intervenable(model)[11];wrapper.capture_input=True
    paths=[str(args.data/'images'/str(raw).replace('\\','/').split('imagenette2-160/')[-1]) for raw in data['paths']]
    dataset=FrozenPathDataset(np.array(paths),data['target'],create_transform(**resolve_model_data_config(model),is_training=False))
    features=[];single=[]
    with torch.inference_mode():
        for images,_,_ in DataLoader(dataset,batch_size=16):
            model(images.cuda().half());features.append(prehead_summary(wrapper.last_input).half().float().cpu().numpy())
        for idx in range(16):
            model(dataset[idx][0][None].cuda().half());single.append(prehead_summary(wrapper.last_input).half().float().cpu().numpy())
    features=np.concatenate(features);single=np.concatenate(single)
    original=torch.load(args.data/'r2_seed17.pt',map_location='cpu',weights_only=False)
    error=features-data['feature'];scaled=error/original['global_std']
    diagnostics={'feature_max_abs_error':float(abs(error).max()),'feature_mean_abs_error':float(abs(error).mean()),
                 'normalized_feature_rmse':float(np.sqrt(np.mean(scaled**2))),
                 'normalized_feature_max_error':float(abs(scaled).max()),
                 'same_backend_batch1_vs_batch16_max_error_first16':float(abs(single-features[:16]).max()),
                 'cached_feature_max_abs':float(abs(data['feature']).max()),
                 'interpretation':'Cross-environment FP16 drift is measured; no claim of bitwise historical feature reproduction.'}
    np.savez_compressed(args.output/'live_features.npz',feature=features,dataset_index=data['dataset_index'])
    (args.output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));print(json.dumps(diagnostics))


if __name__=='__main__':main()
