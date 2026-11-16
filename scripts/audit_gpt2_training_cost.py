"""Read historical training logs; distinguish logged throughput from completion wall time."""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); rows=[]; proof={}; budgets=[]
    for name in ['W576','W768','W1152','D9','D18','H6','H24']:
        root=a.root/name; config=json.loads((root/'training_config.json').read_text())
        completion=json.loads((root/'completion.json').read_text())
        logs=[json.loads(line) for line in (root/'train_log.jsonl').read_text().splitlines()]
        beginning=next(r for r in logs if r['event']=='starting')
        for label,threshold in [('100M',100000000),('300M',300000000),('1B',1000000000)]:
            reached=next(r for r in logs if r['event']=='train' and r['tokens_trained']>=threshold)
            seconds=(datetime.datetime.fromisoformat(reached['time'])-datetime.datetime.fromisoformat(beginning['time'])).total_seconds()
            budgets.append(dict(model=name,budget=label,actual_tokens=reached['tokens_trained'],step=reached['global_step'],
                logged_elapsed_gpu_hours=seconds/3600,
                definition='Time from starting log to first train update reaching budget; integer-second timestamps, before that update validation/checkpoint'))
        measured=[r for r in logs if r['event']=='train' and r.get('global_step',0)>10]
        mean_sec=np.mean([r['seconds'] for r in measured]); tokens=completion['tokens_trained']
        rows.append(dict(model=name,width=config['width'],depth=config.get('n_layer',12),heads=config.get('n_head',config['width']//64),
            sequence_length=config['protocol']['context_length'],input_tokens=tokens,updates=completion['global_step'],
            logged_mean_seconds_per_update=mean_sec,
            logged_input_tokens_per_second=config['protocol']['global_batch_tokens']/mean_sec,
            completion_wall_seconds=completion['wall_seconds'],gpu_hours=completion['wall_seconds']/3600,
            wall_input_tokens_per_second=tokens/completion['wall_seconds'],gpu=config['gpu_name'],
            logged_training_measurements=len(measured),start_events=sum(r['event']=='starting' for r in logs)))
        proof[name]=dict(completion=completion,training_config=config,
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'completion.json',root/'training_config.json',root/'train_log.jsonl']})
    frame=pd.DataFrame(rows); frame.to_csv(a.output/'costs.csv',index=False)
    frame.to_json(a.output/'costs.json',orient='records',indent=2)
    pd.DataFrame(budgets).to_csv(a.output/'observed_budget_costs.csv',index=False)
    (a.output/'evidence.json').write_text(json.dumps(proof,indent=2))
    (a.output/'data_metadata.json').write_bytes((a.root/'data/metadata.json').read_bytes())
    (a.output/'SUMMARY.json').write_text(json.dumps(dict(models=len(frame),total_tokens=int(frame.input_tokens.sum()),
        completion_gpu_hours=float(frame.gpu_hours.sum()),
        note='Summed successful completion wall times. Not total historical billing; excludes unlogged attempts/pilots and CFS.'),indent=2))
    print(frame.to_string(index=False))

if __name__=='__main__': main()
