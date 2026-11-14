"""Finite pilot schedule, validation-only configuration lock, then three seeds."""
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    runs=root/'runs';runs.mkdir(exist_ok=True)
    def run(variant,strength,seed):
        name=f'{variant}_lambda{strength:g}_seed{seed}';out=runs/name
        if (out/'complete.json').exists():return out
        with (runs/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'scripts/train_cfs_regularization.py'),'--data',str(root/'baseline/learned_gating_inputs_20261115'),
                '--aux',str(root/'inputs'),'--output',str(out),'--variant',variant,'--strength',str(strength),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return out
    rows=[]
    for variant in ('A1','A2','A3'):
        for strength in (.01,.03,.1,.3,1.):
            out=run(variant,strength,17);history=pd.read_csv(out/'training_history.csv')
            best=history.loc[history.val_joint_regret.idxmin()].to_dict()
            rows.append(dict(variant=variant,strength=strength,**best))
            print('pilot complete',variant,strength,best['val_joint_regret'],flush=True)
    table=pd.DataFrame(rows);table.to_csv(root/'pilot_validation.csv',index=False)
    selected={v:table[table.variant==v].sort_values(['val_joint_regret','strength']).iloc[0].to_dict() for v in ('A1','A2','A3')}
    headline=min(selected,key=lambda v:selected[v]['val_joint_regret'])
    (root/'selection_lock.json').write_text(json.dumps(dict(selected=selected,headline=headline,
        rule='seed17 minimum mean tune candidate regret; no new held-out evaluation yet',test_used=False),indent=2))
    for variant,choice in selected.items():
        for seed in (23,41):run(variant,choice['strength'],seed)
    (root/'TRAINING_COMPLETE').write_text('21 new runs; selected configurations locked before test evaluation\n')


if __name__=='__main__':main()
