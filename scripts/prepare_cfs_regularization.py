"""Export original auxiliary labels and a preselected small interaction cohort."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.train_dynamic_cfs_router import load_router_data
from scripts.prepare_learned_gating import load_fields


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    out=args.workspace/'outputs';root=out/'learned_gating_inputs_20261115'
    dirs={'train':out/'vit_base_layer11_train4096_alpha16','tune':out/'vit_base_layer11_router_final_split/tune','test':out/'vit_base_layer11_router_final_split/test'}
    for split,path in dirs.items():
        values=load_router_data(path,0);old=np.load(root/f'{split}.npz')
        np.testing.assert_array_equal(old['paths'],values['paths'])
        np.testing.assert_array_equal(old['feature'],values['global'])
        np.savez_compressed(args.output/f'{split}_aux.npz',**{k:v for k,v in values.items() if k!='global'})
    selected=np.sort(np.random.default_rng(42).choice(448,128,replace=False))
    np.save(args.output/'test_selected.npy',selected)
    for raw in np.load(root/'tune.npz')['paths']:
        rel=str(raw).replace('\\','/').split('imagenette2-160/')[-1]
        dest=args.output/'images'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.workspace/'.cache/data/imagenette2-160'/rel,dest)
    (args.output/'manifest.json').write_text(json.dumps(dict(seed=42,tune_count=64,test_count=128,test_positions=selected.tolist(),
        sources={k:str(v) for k,v in dirs.items()},original_training_alpha_grid_points=16),indent=2))


if __name__=='__main__':main()
