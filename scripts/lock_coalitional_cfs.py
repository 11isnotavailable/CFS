"""Lock all validation choices before any test teacher/evaluation."""
import argparse,json,hashlib,time
from pathlib import Path
import pandas as pd
import torch
from train_coalitional_cfs import name


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--evaluation-budget',action='store_true');a=p.parse_args();root=a.root
    prefix='EVALUATIONS' if a.evaluation_budget else 'WORKER'
    assert (root/f'{prefix}_0_COMPLETE').exists() and (root/f'{prefix}_1_COMPLETE').exists()
    suffix='_evaluations' if a.evaluation_budget else ''
    lock={}
    for job in json.loads((root/f'training_plan{suffix}.json').read_text())['jobs']:
        run=root/'runs'/name(job);assert (run/'COMPLETE.json').exists()
        history=pd.read_csv(run/'history.csv');assert list(history.step)==list(range(250,2001,250))
        best=history.loc[history.validation_mse.idxmin()];payload=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
        assert payload['step']==int(best.step) and abs(payload['validation_mse']-best.validation_mse)<1e-8
        lock[name(job)]=dict(step=payload['step'],validation_mse=payload['validation_mse'],sha256=hashlib.sha256((run/'model.pt').read_bytes()).hexdigest())
    out=root/f'selection_lock{suffix}.json';assert not out.exists();out.write_text(json.dumps(dict(models=lock,time=time.time(),test_used=False),indent=2));print('locked',len(lock))


if __name__=='__main__':main()
