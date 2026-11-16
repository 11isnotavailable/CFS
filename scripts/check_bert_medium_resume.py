"""Disposable7+7 versus14-update continuation check; never used as production weights."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    out=root/'smoke';out.mkdir(exist_ok=True);done=out/'TRAIN_RESUME_CHECKS_PASSED.json'
    if done.exists():return
    assert (root/'data/READY.json').exists()
    attempt=out/f'resume_check_{time.time_ns()}';attempt.mkdir()
    for name,steps,resume in [('resume_a',7,False),('resume_a',7,True),('resume_b',14,False)]:
        dest=attempt/name
        cmd=[sys.executable,'-u',str(root/'scripts/train_bert_medium.py'),'--model','B1','--root',str(root),
            '--output',str(dest),'--stop-after-updates',str(steps)]+(['--resume'] if resume else [])
        with (attempt/f'{name}_{steps}_{resume}.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    left=torch.load(attempt/'resume_a/last.pt',map_location='cpu',weights_only=False)
    right=torch.load(attempt/'resume_b/last.pt',map_location='cpu',weights_only=False)
    error=0.
    for key,value in left['weights'].items():
        torch.testing.assert_close(value,right['weights'][key],rtol=0,atol=0)
        error=max(error,float((value-right['weights'][key]).abs().max()))
    for key in ['tokens','step','targets','document_visits','last_document']:assert left[key]==right[key]
    assert torch.equal(left['mask_rng'],right['mask_rng']) and torch.equal(left['rng_cuda'],right['rng_cuda'])
    done.write_text(json.dumps(dict(purpose='Disposable training/resume test; weights never reused',
        updates=14,exact_weight_equality=True,max_absolute_error=error,tokens=left['tokens'],masking_rng_equal=True,artifacts=str(attempt)),indent=2))
    print(done.read_text(),flush=True)
if __name__=='__main__':main()
