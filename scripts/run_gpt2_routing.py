"""Finite staged experiment; optional second layer decided by timing, not test."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layer',type=int,required=True)
    args=p.parse_args();root=args.root.resolve();layer=args.layer;logs=root/f'layer{layer}/logs';logs.mkdir(parents=True,exist_ok=True)
    def run(script,extra,complete,log):
        if complete.exists():return
        command=[sys.executable,str(root/'scripts'/script),'--root',str(root),'--layer',str(layer),*extra]
        with (logs/log).open('a') as stream:
            stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=root)
    smoke=json.loads((root/f'smoke_layer{layer}.json').read_text())
    estimated=(2048*2124+128*4104)/128*smoke['median_seconds_128']
    assert estimated<3600,'Measured suffix cost exceeds planned one-GPU-hour teacher budget'
    for phase,N,V,epochs in [('pilot',256,64,20),('full',2048,128,60)]:
        for split,count in [('train',N),('tune',V)]:
            run('collect_gpt2_routing_teacher.py',['--split',split,'--limit',str(count)],root/f'layer{layer}/pair/{split}/complete_{count}.json',f'{phase}_pair_{split}.log')
        run('train_gpt2_routing.py',['--phase',phase,'--kind','bootstrap','--seed','17'],root/f'layer{layer}/{phase}/runs/bootstrap_seed17/COMPLETE.json',f'{phase}_bootstrap.log')
        run('collect_gpt2_joint_teacher.py',['--phase',phase],root/f'layer{layer}/{phase}/joint/COMPLETE.json',f'{phase}_joint.log')
        kinds=('R2','Gating') if phase=='pilot' else ('R2','Gating','A0')
        seeds=(17,) if phase=='pilot' else (17,23,41)
        for kind in kinds:
            for seed in seeds:
                run('train_gpt2_routing.py',['--phase',phase,'--kind',kind,'--seed',str(seed)],root/f'layer{layer}/{phase}/runs/{kind}_seed{seed}/COMPLETE.json',f'{phase}_{kind}_{seed}.log')
        if phase=='pilot':
            # Resource/numerical gate only; do not require one method to win.
            records=[json.loads((root/f'layer{layer}/pilot/runs/{kind}_seed17/COMPLETE.json').read_text()) for kind in kinds]
            assert all(0<=r['best']<1e10 for r in records)
            (root/f'layer{layer}/PILOT_ACCEPTED.json').write_text(json.dumps(dict(reason='finite losses, correct cached interventions, bounded measured cost',estimated_teacher_seconds=estimated,records=records),indent=2))
        print('completed',layer,phase,flush=True)
    import torch
    lock={}
    for kind in ('R2','Gating','A0'):
        for seed in (17,23,41):
            payload=torch.load(root/f'layer{layer}/full/runs/{kind}_seed{seed}/model.pt',map_location='cpu',weights_only=False)
            lock[f'{kind}_{seed}']=dict(epoch=payload['best_epoch'],validation_hard_kl=payload['validation_metric'])
    lock_path=root/f'layer{layer}/selection_lock.json'
    if lock_path.exists():assert json.loads(lock_path.read_text())['models']==lock
    else:lock_path.write_text(json.dumps(dict(models=lock,test_used=False,primary='original R2',time=time.time()),indent=2))
    (root/f'layer{layer}/TRAINING_COMPLETE').write_text('Nine models locked on actual full-search validation hard KL\n')


if __name__=='__main__':main()
