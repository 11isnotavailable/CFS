"""Same64-image, same-definition CFS monitoring at the fixed epoch milestones."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.controlled_vit_imagenet import SPECS, controls, PROTOCOL, protocol_hash
from scripts.analyze_matched_scaling import analyze


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=SPECS,required=True)
    p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--probe-metadata',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epoch',type=int,required=True); a=p.parse_args()
    assert a.epoch in PROTOCOL['light_cfs']['epochs']
    a.output.mkdir(parents=True,exist_ok=True); done=a.output/'COMPLETE.json'
    import torch
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert payload['epoch']==a.epoch and payload['model_id']==a.model
    assert payload['protocol_hash']==protocol_hash()
    checkpoint_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if done.exists():
        assert json.loads(done.read_text())['checkpoint_sha256']==checkpoint_hash
        return
    root=Path(__file__).resolve().parents[1]; raw=a.output/'raw'; started=time.time()
    subprocess.run([sys.executable,'-u',str(root/'scripts/collect_controlled_vit_imagenet.py'),
        '--model',f'controlled_{a.model}','--checkpoint',str(a.checkpoint),'--data-root',str(a.data_root),
        '--output-dir',str(raw),'--num-samples','64','--indices-metadata',str(a.probe_metadata),
        '--layers',str(SPECS[a.model][1]-1),'--alpha-steps','31','--alpha-min','0','--alpha-max','3',
        '--shard-size','32','--variant-batch-size','128','--save-alpha-error-cube'],check=True,cwd=root)
    assert len(list(raw.glob('shard_*.npz')))==2
    analysis=a.output/'analysis'; analysis.mkdir(exist_ok=True)
    # Monitoring is light by sample count, not by altering the alpha grid or CFS.
    ns,_=controls(a.model)
    params=sum(t.numel() for t in payload['state_dict'].values())
    entry=dict(name=a.model,family='ViT',parameters=params,model_id=f'controlled_{a.model}',
        heads=SPECS[a.model][2],samples=64,layers=[SPECS[a.model][1]-1],N=ns,K=[],path=str(raw.resolve()))
    analyze(entry,root,analysis,500,PROTOCOL['light_cfs']['bootstrap'],42)
    done.write_text(json.dumps(dict(model=a.model,epoch=a.epoch,samples=64,checkpoint_sha256=checkpoint_hash,
        seconds=time.time()-started,alpha_steps=31,purpose='Fixed-time functional stability monitoring; no model-specific stopping or selection'),indent=2))


if __name__=='__main__': main()
