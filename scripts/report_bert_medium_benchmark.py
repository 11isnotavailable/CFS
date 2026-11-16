import argparse,itertools,json
from pathlib import Path
import pandas as pd

def schedule(hours,cards):
    return min(max(sum(v for v,g in zip(hours,assignment) if g==gpu) for gpu in range(cards)) for assignment in itertools.product(range(cards),repeat=len(hours)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    rows=[]
    for stage in ['measurements','io_confirmation']:
        for file in sorted((root/stage).glob('B*.json')):
            value=json.loads(file.read_text());rows.append(dict(stage=stage,**{k:v for k,v in value.items() if not isinstance(v,(list,dict))}))
    frame=pd.DataFrame(rows);assert len(frame)==8
    frame.to_csv(root/'benchmark.csv',index=False)
    baseline=frame[frame.stage=='measurements'].set_index('model');rates=baseline.input_tokens_per_second.to_dict()
    time4=1.5*baseline.loc['B2','seconds_per_update']-.5*baseline.loc['B5','seconds_per_update']
    rates.update(B4=32768/time4,B6=rates['B2'],B7=rates['B2'])
    cost=pd.DataFrame([dict(model=k,tokens_per_second=v,gpu_hours_2B=2e9/v/3600,
        basis='measured' if k in baseline.index else 'depth extrapolation' if k=='B4' else 'B2 proxy, head dimension unmeasured') for k,v in sorted(rates.items())])
    hours=cost.gpu_hours_2B.tolist();cost.to_csv(root/'estimated_costs.csv',index=False)
    plan=dict(total_training_gpu_hours=sum(hours),optimal_two_gpu_hours=schedule(hours,2),optimal_three_gpu_hours=schedule(hours,3),
        limitations='Short windows; exclude preparation, clipping overhead, evaluation, checkpoint IO and CFS. B4/B6/B7 are estimates; third GPU hypothetical.')
    (root/'COST_ESTIMATE.json').write_text(json.dumps(plan,indent=2))
    report='# New BERT-Medium family preflight throughput\n\n'
    report+='Fresh random weights, actual WordPiece cache pipeline, batch256, sequence128, BF16 and Flash Attention;50 warmup +200 timed updates per case. '
    report+='The I/O confirmation adds host next-batch wait / asynchronous-copy enqueue timing; device copy completion is not separately isolated.\n\n'
    columns=['stage','model','parameters','input_tokens_per_second','seconds_per_update','gpu_utilization_mean','peak_allocated_gib','mean_host_input_wait_seconds','host_input_wait_fraction']
    report+=frame[columns].to_markdown(index=False,floatfmt='.6f')+'\n\n'+cost.to_markdown(index=False,floatfmt='.4f')+'\n\n'
    report+=json.dumps(plan,indent=2)+'\n\n'
    report+='The original small W256/D12 benchmark is not used for these estimates. Production also clips gradients and evaluates/saves milestones. '
    report+='Short cached input windows are not a guarantee of full-corpus IO or long-run throughput; actual logs record input wait and GPU-hours.\n'
    (root/'REPORT.md').write_text(report);print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
