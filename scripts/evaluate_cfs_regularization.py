"""Reuse the frozen gating evaluator; change only the selected router checkpoint."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    assert (root/'TRAINING_COMPLETE').exists()
    lock=json.loads((root/'selection_lock.json').read_text());baseline=root/'baseline';all_rows=[]
    configs={'A0':None,**lock['selected']}
    for variant,choice in configs.items():
        adapter=root/f'adapter_{variant}';adapter.mkdir(exist_ok=False)
        for seed in (17,23,41):
            (adapter/f'gating_seed{seed}').symlink_to(baseline/f'runs/gating_seed{seed}',target_is_directory=True)
            source=baseline/f'runs/joint_only_r2_seed{seed}' if variant=='A0' else root/f'runs/{variant}_lambda{choice["strength"]:g}_seed{seed}'
            (adapter/f'joint_only_r2_seed{seed}').symlink_to(source,target_is_directory=True)
        output=root/f'eval_{variant}'
        with (root/f'eval_{variant}.log').open('w') as log:
            subprocess.run([sys.executable,str(root/'scripts/evaluate_learned_gating.py'),'--data',str(baseline/'learned_gating_inputs_20261115'),
                '--runs',str(adapter),'--checkpoint',str(baseline/'base.pth'),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        frame=pd.read_csv(output/'per_input.csv');frame.loc[frame.method=='R2_joint_only','method']=variant
        if variant=='A0':
            all_rows.append(frame);reference=np.load(output/'dense_logits.npz')['logits']
        else:
            np.testing.assert_array_equal(reference,np.load(output/'dense_logits.npz')['logits'])
            all_rows.append(frame[frame.method==variant])
        print('evaluated',variant,flush=True)
    pd.concat(all_rows,ignore_index=True).to_csv(root/'routing_per_input.csv',index=False)
    (root/'EVALUATION_COMPLETE').write_text('All selected variants; identical dense logits\n')


if __name__=='__main__':main()
