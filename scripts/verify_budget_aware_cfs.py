"""Check the frozen finite search, source integrity and validation-only selection."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.budget_aware_objective import select_validation


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    frame=pd.read_csv(root/'validation_trajectories.csv');assert len(frame)==12*11
    assert set(frame.seed)=={17,23,41} and set(frame.weight9)=={1.,1.5,2.,3.}
    saved=json.loads((root/'selection_lock.json').read_text());rebuilt,_=select_validation(frame)
    assert rebuilt==saved
    for seed in (17,23,41):
        rows=frame[(frame.seed==seed)&(frame.epoch==0)]
        for col in ('val_K3','val_K6','val_K9'):assert rows[col].max()-rows[col].min()<1e-8
    for run in (root/'runs').glob('w*_seed*'):
        if not run.is_dir():continue
        config=json.loads((run/'config.json').read_text());complete=json.loads((run/'complete.json').read_text())
        assert complete['epochs']==10 and complete['updates']==640 and config['test_loaded'] is False
        original=root/f'source/runs/A3_lambda0.01_seed{config["seed"]}/model.pt'
        assert hashlib.sha256(original.read_bytes()).hexdigest()==config['source_sha256']
        chosen=saved['representatives'][f'{config["weight9"]:g}'];selected=root/f'selected/w{config["weight9"]:g}_seed{config["seed"]}/model.pt'
        assert hashlib.sha256(selected.read_bytes()).hexdigest()==hashlib.sha256((run/f'epoch_{int(chosen["epoch"]):02d}.pt').read_bytes()).hexdigest()
        payload=torch.load(selected,map_location='cpu',weights_only=False);assert payload['config']['parameters']==825091
    test=pd.read_csv(root/'routing_per_input.csv');assert len(test)==448*3*9*3*2
    assert test.groupby('sample').dense_prediction.nunique().max()==1
    assert np.isfinite(test.kl).all()
    result=dict(runs=12,seeds=[17,23,41],epochs=10,updates_per_run=640,total_updates=7680,parameters=825091,
        weights=[1,1.5,2,3],validation_only_selection=True,source_weights_unchanged=True,routing_rows=len(test),
        primary='A3' if saved['fallback_to_A3'] else f'W{saved["primary"]["weight9"]:g}',
        eligible_candidates=int(pd.read_csv(root/'validation_candidates.csv').eligible.sum()),fusion_run=False)
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
