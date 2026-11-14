"""Offline candidate-count controls. Never loads a model or changes source data.

Manifest paths are relative to the manifest. Each entry identifies an original
shard directory, family, model, parameters, N/K lists and optional identity hash.
Oracle point estimates retain the original per-layer valid-source weighting.
Bootstrap resamples whole inputs, retaining all layers and heads together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
from scripts.analyze_scaling_cfs import expected_best_of_k
from scripts.analyze_functional_capacity import (
    exact_minimum_cover, spectral_effective_rank, subset_table,
)
from scripts.analyze_qwen_cfs import minimum_cover


def matched(values, n):
    if not 1 <= n <= values.shape[-1] or not np.isfinite(values).all():
        raise ValueError("Need finite off-diagonal candidates and 1 <= N <= H-1")
    # Original implementation uses arbitrary-precision integer combinations;
    # ratios are safe for H <= 40 and do not use factorial floating arithmetic.
    return expected_best_of_k(values, n)


def subsets(heads, k, repeats, rng):
    if not 1 <= k <= heads or repeats < 2:
        raise ValueError("Require 1 <= K <= H and repeats >= 2")
    if k == heads:
        return np.arange(heads)[None, :]
    # Independent uniform subsets; no replacement WITHIN each draw.
    return np.sort([rng.choice(heads, k, replace=False) for _ in range(repeats)], axis=1)


def fixed_metrics(scores, valid, draws):
    k = draws.shape[1]
    table, cardinality = subset_table(k)
    unique, inverse = np.unique(draws, axis=0, return_inverse=True)
    sub = scores[unique[:, :, None], unique[:, None, :]]
    mask = valid[unique]
    cover = exact_minimum_cover(sub, mask, 0.5, table, cardinality) / k
    rank = spectral_effective_rank(sub, mask) / k
    return np.stack([cover[inverse], rank[inverse]], axis=-1)


def load(entry, base):
    directory = (base / entry["path"]).resolve()
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    files = sorted(directory.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"No full matrices: {directory}")
    values = {key: [] for key in ("single_drop", "substitution")}
    ids = []
    hashes = {}
    id_key = "sample" if entry["family"] == "Qwen" else "dataset_index"
    for p in files:
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p, allow_pickle=False) as data:
            for key in values:
                values[key].append(data[key].astype(np.float64))
            ids.extend(data[id_key].tolist())
    drop, sub = (np.concatenate(values[k]) for k in values)
    h = meta["heads"]
    if drop.ndim != 3 or sub.shape != (*drop.shape, h) or drop.shape[-1] != h:
        raise ValueError("Expected drops [input,layer,H], substitution [input,layer,H,H]")
    if len(set(ids)) != len(ids) or len(ids) != entry["samples"]:
        raise ValueError("Duplicate identities or incomplete sample set")
    layers = meta.get("selected_layers", meta.get("layers"))
    if layers != entry["layers"] or h != entry["heads"]:
        raise ValueError("Head/layer protocol mismatch")
    if meta["model"] != entry["model_id"]:
        raise ValueError("Checkpoint identity mismatch")
    if entry["family"] == "ViT" and meta.get("dataset_size") != 50000:
        raise ValueError("Expected original ImageNet-1K validation dataset")
    if entry["family"] == "Qwen" and meta.get("prediction_position_spec") != "random:8":
        raise ValueError("Expected formal random-eight position protocol")
    scores = 1 - sub / np.maximum(drop[..., None], 1e-12)
    scores[..., np.arange(h), np.arange(h)] = np.nan
    if not np.isfinite(scores[..., ~np.eye(h, dtype=bool)]).all():
        raise ValueError("Missing off-diagonal substitution measurements")
    order = np.argsort(ids)
    return scores[order], (drop > 1e-5)[order], np.asarray(ids)[order], layers, meta, hashes


def estimate(num, den, indices):
    # Average per-layer ratios. Preserves source-weighted original point estimate
    # when validity rates differ between inputs (especially ViT-L).
    return np.nanmean(num[indices].sum(axis=1) / den[indices].sum(axis=1), axis=-1)


def analyze(entry, base, out, repeats=500, bootstrap=2000, seed=42):
    scores, valid, ids, layers, meta, hashes = load(entry, base)
    n, l, h = valid.shape
    rng_boot = np.random.default_rng(seed)
    indices = rng_boot.integers(n, size=(bootstrap, n))
    identity = [entry["family"], entry["name"], entry["parameters"], h]
    details, summaries, bootrows = [], [], []
    numerators, denominators, mcvars = {}, {}, {}
    for metric in ["raw"] + [f"matched_{v}" for v in entry["N"]]:
        num = np.zeros((n, l)); den = valid.sum(-1).astype(float)
        for i in range(n):
            for j in range(l):
                candidates = scores[i, j][~np.eye(h, dtype=bool)].reshape(h, h-1)
                vals = candidates.max(-1) if metric == "raw" else matched(candidates, int(metric.split('_')[1]))
                num[i, j] = vals[valid[i, j]].sum()
        numerators[metric], denominators[metric] = num, den
        mcvars[metric] = np.zeros((n, l))
    for metric in ("cover_full", "rank_full"):
        numerators[metric] = np.zeros((n, l))
        denominators[metric] = np.ones((n, l))
        mcvars[metric] = np.zeros((n, l))
    stability = []
    for k in entry["K"]:
        for prefix in ("cover", "rank"):
            metric = f"{prefix}_{k}"
            numerators[metric] = np.zeros((n, l))
            denominators[metric] = np.ones((n, l))
            mcvars[metric] = np.zeros((n, l))
    for i in range(n):
        for j in range(l):
            s, v = scores[i, j], valid[i, j]
            numerators["cover_full"][i,j] = minimum_cover(s,v,0.5)/h
            numerators["rank_full"][i,j] = spectral_effective_rank(s[None],v[None])[0]/h
            for k in entry["K"]:
                # Stable across batch order and per-model invocations.
                model_seed = int.from_bytes(hashlib.sha256(entry["name"].encode()).digest()[:4], 'little')
                rng = np.random.default_rng(np.random.SeedSequence([seed,model_seed,int(ids[i]),layers[j],k]))
                draws = subsets(h,k,repeats,rng)
                vals = fixed_metrics(s,v,draws)
                for col,prefix in enumerate(("cover","rank")):
                    metric = f"{prefix}_{k}"
                    numerators[metric][i,j] = vals[:,col].mean()
                    mcvars[metric][i,j] = vals[:,col].var(ddof=1)/len(vals) if len(vals)>1 else 0
                    for count in sorted(set([min(100,len(vals)),min(250,len(vals)),len(vals)])):
                        stability.append(dict(model=entry['name'],input_id=int(ids[i]),layer=layers[j],metric=metric,draws=count,mean=float(vals[:count,col].mean())))
                np.savez_compressed(out / f"subsets_{i}_{j}_K{k}.npz",heads=draws,cover=vals[:,0],rank=vals[:,1])
        if i%8 == 0: print(entry['name'],i+1,n,flush=True)
    for metric,num in numerators.items():
        den = denominators[metric]
        point = estimate(num,den,np.arange(n)[None])[0]
        boot = estimate(num,den,indices)
        low,high = np.quantile(boot,[.025,.975])
        inp = np.nanmean(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0),axis=1)
        mcse = np.sqrt(mcvars[metric].sum())/(n*l)
        summaries.append(identity+[metric,point,low,high,np.nanstd(inp,ddof=1),mcse,n,repeats,bootstrap,seed])
        for b,value in enumerate(boot): bootrows.append([entry['name'],metric,b,value])
        for i in range(n):
            for j in range(l):
                details.append(identity+[int(ids[i]),layers[j],metric,num[i,j],den[i,j],num[i,j]/den[i,j] if den[i,j] else np.nan,np.sqrt(mcvars[metric][i,j]),seed,1 if metric.endswith('_full') or metric.split('_')[-1]==str(h) else repeats if metric.startswith(('cover_','rank_')) else 0])
    cols=['family','model','parameters','H']
    pd.DataFrame(details,columns=cols+['input_id','layer','metric','numerator','denominator','value','subgraph_mc_se','seed','draws']).to_csv(out/'per_input_layer.csv',index=False)
    summary=pd.DataFrame(summaries,columns=cols+['metric','mean','ci_low','ci_high','input_sd','subgraph_mc_se','inputs','repeats','bootstrap','seed'])
    summary.to_csv(out/'summary.csv',index=False)
    summary.to_json(out/'summary.json',orient='records',indent=2)
    summary[['model','metric','mean','ci_low','ci_high']].to_latex(out/'table.tex',index=False,float_format='%.6f')
    pd.DataFrame(bootrows,columns=['model','metric','replicate','value']).to_csv(out/'bootstrap.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'sampling_stability.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(dict(entry=entry,metadata=meta,sha256=hashes,identity_warning='Sample offsets alone do not prove input identity; use the frozen input hash and cohort provenance for paired comparisons.',seed=seed,bootstrap=bootstrap,ordered_input_ids=ids.tolist()),indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',action='append')
    p.add_argument('--repeats',type=int,default=500)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    entries=[e for e in manifest['models'] if not args.model or e['name'] in args.model]
    if not entries: raise ValueError('No requested models')
    for e in entries:
        # Fail rather than overwrite results, including partial runs.
        out=args.output/e['name']; out.mkdir(parents=True,exist_ok=False)
        analyze(e,args.manifest.parent,out,args.repeats,args.bootstrap,args.seed)


if __name__=='__main__': main()
