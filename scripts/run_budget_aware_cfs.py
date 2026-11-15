"""Run the fixed12 continuations and lock selection without reading test data."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.budget_aware_objective import select_validation


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    (root/'runs').mkdir(exist_ok=False);records=[]
    for weight in (1.,1.5,2.,3.):
        for seed in (17,23,41):
            name=f'w{weight:g}_seed{seed}';out=root/'runs'/name
            with (root/'runs'/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable,str(root/'scripts/train_budget_aware_cfs.py'),'--source',str(root/'source'),
                    '--output',str(out),'--weight9',str(weight),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True)
            records.append(pd.read_csv(out/'training_history.csv'));print('completed',name,flush=True)
    table=pd.concat(records,ignore_index=True);lock,summary=select_validation(table)
    table.to_csv(root/'validation_trajectories.csv',index=False);summary.to_csv(root/'validation_candidates.csv',index=False)
    (root/'selection_lock.json').write_text(json.dumps(lock,indent=2))
    for weight,row in lock['representatives'].items():
        for seed in (17,23,41):
            dest=root/f'selected/w{weight}_seed{seed}';dest.mkdir(parents=True,exist_ok=False)
            shutil.copy2(root/f'runs/w{weight}_seed{seed}/epoch_{int(row["epoch"]):02d}.pt',dest/'model.pt')
    (root/'TRAINING_COMPLETE').write_text('12 runs, 10 epochs each; validation selection locked before test\n')
    print(json.dumps(lock),flush=True)


if __name__=='__main__':main()
