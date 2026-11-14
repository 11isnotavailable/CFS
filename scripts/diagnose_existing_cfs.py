"""Endpoint gradient diagnostics for frozen original and joint-only baselines."""
import argparse,subprocess,sys,os
from pathlib import Path


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    for seed in (17,23,41):
        for name,checkpoint in [('A0',root/f'baseline/runs/joint_only_r2_seed{seed}/model.pt'),
                                ('A4',root/f'baseline/learned_gating_inputs_20261115/r2_seed{seed}.pt')]:
            subprocess.run([sys.executable,str(root/'scripts/train_cfs_regularization.py'),'--data',str(root/'baseline/learned_gating_inputs_20261115'),
                '--aux',str(root/'inputs'),'--output',str(root/f'endpoints/{name}_seed{seed}'),'--diagnose-checkpoint',str(checkpoint)],check=True)


if __name__=='__main__':main()
