"""Wait for the existing ViT job, then serially finish the authorized experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/cfs_rebuttal_20261114')


def run(args):
    print('RUN',args,flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--wait-vit-pid',type=int,required=True); a=p.parse_args()
    # Never overlap the two model families on GPU 1.
    proc=Path(f'/proc/{a.wait_vit_pid}/cmdline')
    while proc.exists() and b'run_rebuttal_pipeline.py' in proc.read_bytes():
        time.sleep(30)
    for name in ['vit_small','vit_base','vit_large']:
        if not (ROOT/'analysis'/name/'summary.csv').exists():
            raise RuntimeError(f'Previous ViT pipeline incomplete: {name}; inspect pipeline_vit.log')
        if not (ROOT/'analysis'/name/'consistency_checks.json').exists():
            run(['scripts/validate_matched_scaling.py','--result',str(ROOT/'analysis'/name)])
    if not (ROOT/'figures_vit').exists():
        run(['scripts/summarize_matched_scaling.py','--input','analysis','--output','figures_vit'])
        run(['scripts/report_matched_scaling.py','--figures','figures_vit','--analysis','analysis','--historical','historical'])
    print('VIT_RESULTS_COMPLETE',flush=True)
    # Downloads may still be finishing; fail rather than use incomplete assets.
    deadline=time.monotonic()+3600
    while not (ROOT/'data/fineweb_rebuttal_256x129.npz').exists():
        if time.monotonic()>deadline: raise RuntimeError('Preparation not finished within one hour; inspect preparation log')
        time.sleep(30)
    with (ROOT/'pipeline_qwen.log').open('a') as stream:
        subprocess.run([sys.executable,'-u','scripts/run_rebuttal_pipeline.py','--family','Qwen'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    for name in ['vit_small','vit_base','vit_large','qwen15b','qwen3b','qwen7b','qwen14b']:
        result=ROOT/'analysis'/name
        if not (result/'consistency_checks.json').exists():
            run(['scripts/validate_matched_scaling.py','--result',str(result)])
    run(['scripts/summarize_matched_scaling.py','--input','analysis','--output','figures'])
    run(['scripts/report_matched_scaling.py','--figures','figures','--analysis','analysis','--historical','historical'])
    with (ROOT/'environment_freeze.txt').open('w') as stream:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=stream,check=True)
    model_records={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'models').glob('*/provenance.json')}
    (ROOT/'vit_checkpoint_provenance.json').write_text(json.dumps(model_records,indent=2))
    subprocess.run(['tar','-czf','rebuttal_results.tar.gz','analysis','figures','outputs','scripts','src','tests','historical','qwen_manifest.json','vit_manifest.json','qwen_input_identity.json','vit_checkpoint_provenance.json','model_provenance.json','imagenet_provenance.json','environment_freeze.txt','data/fineweb_rebuttal_256x129.npz','data/fineweb_rebuttal_256x129.json'],cwd=ROOT,check=True)
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(status='complete',models=7,archive='rebuttal_results.tar.gz'),indent=2))
    print('ALL_EXPERIMENTS_COMPLETE',flush=True)


if __name__=='__main__': main()
