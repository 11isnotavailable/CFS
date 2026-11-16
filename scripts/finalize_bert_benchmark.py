"""Record completed finite measurements and user's ViT stop; never starts training."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--vit-root',type=Path,required=True); a=p.parse_args()
    for mode in ['measurements','confirmation']:
        for gpu in [0,1]: assert (a.root/mode/f'COMPLETE_gpu{gpu}.json').exists()
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    old=json.loads((a.vit_root/'STATUS.json').read_text())
    stopped=dict(stage='stopped_by_user',recorded_at=now,reason='User requested stop ViT and benchmark BERT only',
        prior_status=old,checkpoints={n:[str(f.relative_to(a.vit_root)) for f in (a.vit_root/'runs'/n).glob('*.pt')] for n in ['V1','V2']})
    if not (a.vit_root/'STOPPED_BY_USER.json').exists():
        (a.vit_root/'STOPPED_BY_USER.json').write_text(json.dumps(stopped,indent=2))
    (a.vit_root/'STATUS.json').write_text(json.dumps(dict(stage='stopped_by_user',recorded_at=now,reason=stopped['reason']),indent=2))
    source=list((a.root/'src').glob('*.py'))+list((a.root/'scripts').glob('*.py'))
    result=dict(status='benchmark_complete',recorded_at=now,formal_training_started=False,
        model_checkpoints_saved=False,cfs_collected=False,measured_cases=24,confirmation_cases=4,
        gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv'],text=True),
        source_sha256={str(p.relative_to(a.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    (a.root/'COMPLETE.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
