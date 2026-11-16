"""Analyze a complete axis while other models are pending; mark outputs partial."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scripts.analyze_bert_medium import summarize
from src.bert_medium import AXES,SPECS

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--axis',choices=AXES,required=True);a=p.parse_args()
    out=a.root/'previews'/a.axis;out.mkdir(parents=True,exist_ok=True)
    summaries=[];boots={};inputs={};performance=[];stability=[];identities=[]
    for model in AXES[a.axis]:
        source=a.root/'cfs'/model
        assert (source/'2B/COMPLETE.json').exists()
        meta=json.loads((source/'2B/metadata.json').read_text());identities.append(meta['input_document_hashes'])
        summary,per,boot=summarize(model,source/'2B',out/model)
        summaries.append(summary);boots[model]=boot;inputs[model]=per
        done=json.loads((a.root/'runs'/model/'COMPLETE.json').read_text())
        assert done['tokens']==2000000000
        performance.append(dict(model=model,validation_ce=done['validation_ce'],validation_accuracy=done['validation_accuracy']))
        _,probe,_=summarize(model,source/'1B',out/model/'probe1B',full=False)
        ix=np.random.default_rng(42).integers(len(probe),size=(2000,len(probe)))
        for metric in probe:
            old=probe[metric].to_numpy();new=per.loc[probe.index,metric].to_numpy()
            dif=new-old;draw=np.nanmean(dif[ix],1)
            stability.append(dict(model=model,metric=metric,mean_1B=np.nanmean(old),mean_2B=np.nanmean(new),change=np.nanmean(dif),ci_low=np.nanquantile(draw,.025),ci_high=np.nanquantile(draw,.975)))
    assert all(identity==identities[0] for identity in identities)
    summary=pd.concat(summaries);summary.to_csv(out/'summary.csv',index=False)
    table=pd.DataFrame(performance).set_index('model').join(summary.pivot(index='model',columns='metric',values='mean'))
    table.to_csv(out/'table.csv');paired=[]
    for left,right in zip(AXES[a.axis][:-1],AXES[a.axis][1:]):
        assert inputs[left].index.equals(inputs[right].index)
        for metric in boots[left].columns.intersection(boots[right].columns):
            delta=boots[right][metric]-boots[left][metric]
            paired.append(dict(left=left,right=right,metric=metric,change=table.loc[right,metric]-table.loc[left,metric],ci_low=delta.quantile(.025),ci_high=delta.quantile(.975)))
    pd.DataFrame(paired).to_csv(out/'paired.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'stability.csv',index=False)
    (out/'STATUS.json').write_text(json.dumps(dict(partial_experiment=True,complete_axis=a.axis,models=AXES[a.axis],input_bootstrap=2000,training_seeds=1),indent=2))
    print(table[['validation_ce','raw','matched_2','matched_4','cover_full','rank_full','cover_4','rank_4','valid_source_ratio']].to_string())
    print(pd.DataFrame(paired).query("metric in ['raw','matched_2','cover_full','rank_full']").to_string(index=False))
if __name__=='__main__':main()
