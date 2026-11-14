"""Cross-check controls against original implementations on identical matrices."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.analyze_matched_scaling import load, matched
from scripts.analyze_qwen_cfs import expected_best_two


def validate(directory):
    provenance=json.loads((directory/'provenance.json').read_text())
    entry=provenance['entry']
    # Analysis records should use absolute source paths for independent checks.
    scores,valid,ids,layers,meta,hashes=load(entry,Path.cwd())
    assert hashes==provenance['sha256'], 'Source matrices changed'
    h=scores.shape[-1]
    oracle=[]; best2=[]; max_error=0.
    for j in range(len(layers)):
        candidates=scores[:,j][valid[:,j]]
        candidates=candidates[np.isfinite(candidates)].reshape(-1,h-1)
        original=np.array([expected_best_two(row) for row in candidates])
        actual=matched(candidates,2)
        max_error=max(max_error,float(np.max(np.abs(original-actual))))
        oracle.append(candidates.max(-1).mean()); best2.append(original.mean())
    summary=pd.read_csv(directory/'summary.csv').set_index('metric')
    for metric,expected in [('raw',np.mean(oracle)),('matched_2',np.mean(best2))]:
        np.testing.assert_allclose(summary.loc[metric,'mean'],expected,rtol=0,atol=1e-12)
    boundary=[]
    if h in entry['K']:
        for metric in ['cover','rank']:
            for col in ['mean','ci_low','ci_high']:
                np.testing.assert_allclose(summary.loc[f'{metric}_{h}',col],summary.loc[f'{metric}_full',col],rtol=0,atol=1e-12)
            boundary.append(metric)
        for p in directory.glob(f'subsets_*_K{h}.npz'):
            with np.load(p) as a: assert a['heads'].shape==(1,h)
    stability=pd.read_csv(directory/'sampling_stability.csv').groupby(['metric','draws'])['mean'].mean().unstack()
    changes=[]
    for metric,row in stability.iterrows():
        available=row.dropna()
        if len(available)>1:
            changes.append(dict(metric=metric,delta_100_to_final=float(available.iloc[-1]-available.iloc[0]),delta_250_to_final=float(available.iloc[-1]-available.iloc[-2]),subgraph_mc_se=float(summary.loc[metric,'subgraph_mc_se'])))
    result=dict(model=entry['name'],source_hashes_unchanged=True,matched_2_max_absolute_error=max_error,original_raw_and_matched_2_pass=True,K_equals_H_checks=boundary,valid_source_fraction=float(valid.mean()),sampling_stability=changes)
    output=directory/'consistency_checks.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--result',type=Path,required=True)
    validate(p.parse_args().result)
