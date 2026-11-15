"""Uniform evaluation only after a validation-only weight/epoch lock exists."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    assert (root/'TRAINING_COMPLETE').exists();lock=json.loads((root/'selection_lock.json').read_text())
    baseline=root/'baseline';parts=[];reference=None
    for weight,selection in lock['representatives'].items():
        adapter=root/f'adapter_w{weight}';adapter.mkdir(exist_ok=False)
        for seed in (17,23,41):
            (adapter/f'gating_seed{seed}').symlink_to(baseline/f'runs/gating_seed{seed}',target_is_directory=True)
            (adapter/f'joint_only_r2_seed{seed}').symlink_to(root/f'selected/w{weight}_seed{seed}',target_is_directory=True)
        output=root/f'eval_w{weight}'
        with (root/f'eval_w{weight}.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'scripts/evaluate_learned_gating.py'),'--data',str(baseline/'learned_gating_inputs_20261115'),
                '--runs',str(adapter),'--checkpoint',str(baseline/'base.pth'),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        dense=np.load(output/'dense_logits.npz')['logits'];frame=pd.read_csv(output/'per_input.csv')
        if reference is None:
            reference=dense;parts.append(frame[frame.method!='R2_joint_only'])
        else:np.testing.assert_array_equal(reference,dense)
        new=frame[frame.method=='R2_joint_only'].copy();new['method']='W'+weight;parts.append(new)
        print('evaluated',weight,'epoch',selection['epoch'],'eligible',selection['eligible'],flush=True)
    for variant,method in [('A3','A3'),('A0','A0')]:
        directory=root/f'source/eval_{variant}'
        np.testing.assert_array_equal(reference,np.load(directory/'dense_logits.npz')['logits'])
        frame=pd.read_csv(directory/'per_input.csv');frame=frame[frame.method=='R2_joint_only'].copy();frame['method']=method;parts.append(frame)
    frame=pd.concat(parts,ignore_index=True);frame.to_csv(root/'routing_per_input.csv',index=False)
    frame.to_json(root/'routing_per_input.json',orient='records')
    (root/'EVALUATION_COMPLETE').write_text('Identical dense logits for all methods; no test-based re-selection\n')


if __name__=='__main__':main()
