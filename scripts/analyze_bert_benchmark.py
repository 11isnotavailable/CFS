"""Summarize measured throughput and explicitly labeled seven-model cost projections."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def scheduled(hours,cards):
    best=float('inf'); assignment=None
    for choices in itertools.product(range(cards),repeat=len(hours)):
        loads=[0.]*cards
        for value,gpu in zip(hours,choices): loads[gpu]+=value
        if max(loads)<best: best=max(loads); assignment=choices
    return best,assignment

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args(); root=a.root
    raw=[json.loads(f.read_text()) for f in sorted((root/'measurements').glob('B*.json'))]
    assert len(raw)==24 and all('error' not in r for r in raw)
    columns=[k for k,v in raw[0].items() if not isinstance(v,(list,dict))]
    table=pd.DataFrame(raw)[columns]; table.to_csv(root/'measurements.csv',index=False)
    table.to_latex(root/'measurements.tex',index=False,float_format='%.5f')
    confirm=pd.DataFrame([json.loads(f.read_text()) for f in sorted((root/'confirmation').glob('B*.json'))])
    assert len(confirm)==4 and set(confirm.model)=={'B1','B2','B3','B5'}
    confirm[columns].to_csv(root/'confirmation.csv',index=False)
    # A shared batch for controlled comparisons, selected by total GPU cost of
    # the four measured models, not each model's independently favorable batch.
    end=table[table['mode']=='pipeline']; scores=end.groupby('batch_sequences').input_tokens_per_second.apply(lambda x:sum(1/x))
    batch=int(scores.idxmin()); assert batch==256
    actual=confirm.set_index('model'); initial=end[end.batch_sequences==batch].set_index('model')
    actual['repeat_change_percent']=100*(actual.input_tokens_per_second/initial.input_tokens_per_second-1)
    rates=actual.input_tokens_per_second.to_dict(); time2=batch*128/rates['B2']; time5=batch*128/rates['B5']
    time4=1.5*time2-.5*time5; assert time4>0
    rates.update(B4=batch*128/time4,B6=rates['B2'],B7=rates['B2'])
    rows=[]
    for name in sorted(rates):
        basis='measured200-update pipeline confirmation' if name in actual.index else 'linear-depth extrapolation from B2/B5' if name=='B4' else 'B2 throughput proxy; head dimension not benchmarked'
        rows.append(dict(model=name,input_tokens_per_second=rates[name],basis=basis,
            **{f'gpu_hours_{label}':tokens/rates[name]/3600 for label,tokens in [('100M',1e8),('300M',3e8),('1B',1e9)]}))
    cost=pd.DataFrame(rows); cost.to_csv(root/'projected_model_costs.csv',index=False)
    totals=[]
    for label in ['100M','300M','1B']:
        hours=cost[f'gpu_hours_{label}'].tolist(); total=sum(hours)
        two,plan2=scheduled(hours,2); three,plan3=scheduled(hours,3)
        totals.append(dict(input_tokens_per_model=label,total_gpu_hours=total,two_gpu_hours=two,three_gpu_hours=three,
            ideal_total_divided_by2=total/2,ideal_total_divided_by3=total/3,
            two_gpu_with30percent_allowance=two*1.3,three_gpu_with30percent_allowance=three*1.3,
            two_gpu_assignment=list(plan2),three_gpu_assignment=list(plan3)))
    pd.DataFrame(totals).to_csv(root/'projected_family_costs.csv',index=False)
    (root/'projected_family_costs.json').write_text(json.dumps(totals,indent=2))
    compare=actual[['input_tokens_per_second','seconds_per_update','gpu_utilization_mean','peak_allocated_gib','peak_reserved_gib','repeat_change_percent']].copy()
    resident=table[(table['mode']=='resident')&(table.batch_sequences==batch)].set_index('model')
    compare.insert(0,'resident_tokens_per_second',resident.input_tokens_per_second)
    compare['pipeline_vs_resident_percent']=100*(compare.input_tokens_per_second/compare.resident_tokens_per_second-1)
    compare.to_csv(root/'recommended_batch.csv')
    history=pd.read_csv(root/'gpt2_audit/costs.csv'); original=json.loads((root/'gpt2_audit/data_metadata.json').read_text())
    cache=json.loads((root/'data/READY.json').read_text())
    env=json.loads((root/'measurements/environment_gpu0.json').read_text())
    checks=[json.loads(f.read_text()) for f in (root/'measurements').glob('correctness*.json')]
    spent=sum(r['warmup_and_measured_gpu_hours'] for r in raw)+confirm.warmup_and_measured_gpu_hours.sum()
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(7.2,3))
    for ax,mode in zip(axs,['resident','pipeline']):
        for name in ['B1','B2','B3','B5']:
            points=table[(table.model==name)&(table['mode']==mode)].sort_values('batch_sequences')
            ax.plot(points.batch_sequences,points.input_tokens_per_second/1e6,'o-',label=name)
        ax.set_title(mode); ax.set_xlabel('Sequences / update'); ax.set_ylabel('Million input tokens / second')
        ax.set_xticks([256,512,1024]); ax.grid(alpha=.15)
    axs[1].legend(); fig.tight_layout(); fig.savefig(root/'throughput.pdf'); fig.savefig(root/'throughput.png',dpi=180); plt.close(fig)
    report='# BERT controlled-family throughput benchmark\n\n'
    report+='ViT training was stopped at the user\'s request. This experiment contains finite throughput measurements only: no BERT checkpoints, formal training, CFS or bootstrap.\n\n'
    report+='## Measured protocol\n\n'
    report+=f"Hardware: two {env['gpu']} GPUs, {env['total_vram']/2**30:.1f} GiB each; PyTorch {env['torch']}, Transformers {env['transformers']}. "
    report+='Two independent GPU jobs, no distributed training. B1=192/12/3, B2=256/12/4, B3=320/12/5, B5=256/18/4 (width/depth/heads). '
    report+='Randomly initialized HF BertForMaskedLM, FFN4x, post-LayerNorm, GELU, learned positions, tied vocabulary weights, dropout0.1, no NSP. '
    report+='BF16 autocast with FP32 weights/AdamW, fused AdamW, no torch.compile. Actual profiler operators confirm Flash Attention forward and backward. '
    report+='Only selected MLM hidden states enter the pointwise prediction transform, vocabulary projection and cross-entropy; the full encoder still processes every token. '
    report+='Dynamic Bernoulli15% masking over eligible tokens,80%MASK/10%random/10%unchanged, excluded special tokens.\n\n'
    report+='Each of24 cases used50 warmup +100 timed updates, including masking, zero-grad, forward, loss, backward and optimizer step. '
    report+='A further four end-to-end cases at common batch256 used50 warmup +200 timed updates because the initial windows were short. '
    report+='GPU-resident synthetic tokens omit host supply; pipeline tests include shuffled memory-mapped real WordPiece cache,2 persistent loading workers, pinned memory and host-to-device copy. '
    report+='Tokenization/preparation is offline and excluded from training throughput. nvidia-smi samples utilization every0.5seconds; short windows limit precision.\n\n'
    report+=f"FP32 dense-versus-selected loss/gradient checks passed on both GPUs; maximum gradient absolute difference {max(c['max_gradient_absolute_error'] for c in checks):.3g}. "
    report+='TF32 and dropout were disabled only for equivalence checks; benchmark dropout remains0.1. CPU unit tests also passed.\n\n'
    report+='## Throughput and batch choice\n\nCommon batch256 minimizes summed time for the four measured models. This is a throughput choice, not a validated convergence recipe. '
    report+='The table pairs original resident measurements with200-update real-data confirmation; repeat_change compares the latter to the original100-update pipeline run.\n\n'
    report+=compare.to_markdown(floatfmt='.4f')+'\n\n![Throughput](throughput.png)\n\n'
    report+='Larger batches did not uniformly improve throughput. Allocated tensor memory and allocator-reserved memory are different; both are reported. '
    report+='The24-case raw table includes every tested batch, timing distribution and telemetry; no slow result was filtered.\n\n'
    report+='## Original corpus and small real-data cache\n\n'
    report+=f"Original GPT-2 corpus: {original['dataset']} / {original['subset']}, revision `{original['revision']}` (not FineWeb-Edu). "
    report+=f"Its source manifest lists {len(original['source_files'])} Parquet shards totaling {sum(f['size'] for f in original['source_files']):,} bytes; the name10BT denotes the source subset, not the per-model budget. "
    report+=f"The frozen train stream has {original['train_tokens']:,} GPT-2 BPE tokens including one lookahead token; validation has {original['validation_tokens']:,}. "
    report+=f"Preparation recorded {original['processed_documents_including_last_batch'][0]:,} train and {original['processed_documents_including_last_batch'][1]:,} validation documents including the last tokenization batch; those counts are not exact distinct documents wholly consumed by training. "
    report+='Train/validation assignment hashes original UTF-8 text with BLAKE2b(seed42 + NUL + text), modulo1000, bucket0 for validation. There was no original test split.\n\n'
    report+=f"For this benchmark, read only {cache['bytes_downloaded']:,} source bytes by bounded HTTP ranges from `{cache['source_file']}`, the original seeded order's first shard. "
    report+=f"Use {cache['documents_read']:,} documents, exclude {cache['validation_documents_excluded']} validation document, produce {cache['training_blocks']:,} blocks / {cache['input_tokens']:,} input tokens. "
    report+='No full dataset download or processing. Pinned revision and per-range hashes are retained; the complete2GB shard hash was not reverified because it was not downloaded. '
    report+=f"Tokenizer-only `google-bert/bert-base-uncased` revision `{cache['tokenizer_revision']}`, vocabulary30,522; no pretrained weights loaded. "
    report+='Each block contains126 WordPiece text tokens within one document plus CLS/SEP. Short tails are dropped for this speed test. The small cache repeats and is not a final training corpus.\n\n'
    report+='## Seven-model cost projections\n\n'
    report+='Only B1/B2/B3/B5 were benchmarked. For planning the seven-model family, assume B4=W256/D9/H4, B6=W256/D12/H2, B7=W256/D12/H8. '
    report+='B4 uses linear depth-time extrapolation from measured D12/D18; B6/B7 use B2 as a proxy. Their head dimensions and kernels can change throughput, so these are not measurements. '
    report+='Input-token budgets count128 positions per sequence (including CLS/SEP), not masked prediction targets; content-only counts are126/128 as large. '
    report+='At15% eligible masking, supervised targets are approximately14.8% of input positions.\n\n'
    report+=cost.to_markdown(index=False,floatfmt='.5f')+'\n\n'
    report+=pd.DataFrame(totals).drop(columns=['two_gpu_assignment','three_gpu_assignment']).to_markdown(index=False,floatfmt='.4f')+'\n\n'
    report+='GPU-hours are steady-state training-only extrapolations. Wall times use optimal assignment of seven independent whole-model jobs; the ideal total/cards lower bound is also supplied. '
    report+='The third GPU is hypothetical: only two exist here and three-way CPU/memory contention was not tested. A30% allowance is an illustrative planning scenario, not a measured confidence interval. '
    report+='Preparation, validation, checkpointing, failures, CFS and final analysis are excluded; short-window speeds do not establish multi-hour thermal or IO stability.\n\n'
    report+='## Historical GPT-2 measured costs\n\n'
    report+=history[['model','width','depth','heads','sequence_length','input_tokens','gpu_hours','wall_input_tokens_per_second']].to_markdown(index=False,floatfmt='.3f')+'\n\n'
    report+=f"Seven successful runs total {history.gpu_hours.sum():.3f} GPU-hours for {history.input_tokens.sum():,} tokens (3,000,500,224 per model). "
    report+='Each log contains one starting event and5723 updates; global batch524288 tokens, sequence1024, gradient accumulation16 with microbatch32, BF16 and torch.compile. '
    report+='Completion wall time includes that run\'s validation/checkpoint overhead, but not all historical pilots/billing or CFS. '
    report+='BERT has much smaller widths, sequence128, different tokenizer/objective and sparse supervised vocabulary projection; the comparison is a practical job-cost comparison, not an architecture-controlled speedup claim.\n\n'
    report+='The historical GPU is recorded as PRO6000 Blackwell Server Edition; current GPUs are Workstation Edition. Historical Torch2.8.0/Transformers4.57.6 also differ from this run.\n\n'
    historical_budget=pd.read_csv(root/'gpt2_audit/observed_budget_costs.csv').groupby('budget').logged_elapsed_gpu_hours.sum()
    matched_budget=pd.DataFrame([dict(input_tokens_per_model=r['input_tokens_per_model'],
        historical_gpt2_observed_gpu_hours=historical_budget[r['input_tokens_per_model']],
        small_bert_projected_gpu_hours=r['total_gpu_hours']) for r in totals])
    matched_budget.to_csv(root/'historical_budget_comparison.csv',index=False)
    report+=matched_budget.to_markdown(index=False,floatfmt='.4f')+'\n\n'
    report+='Historical budget costs above come from first logged updates crossing each token threshold, not scaling the3B completion time. '
    report+='They use integer-second log timestamps and exclude validation/saving immediately after that update; exact reached token counts are in gpt2_audit/observed_budget_costs.csv. '
    report+='The same caveats about architecture, sequence length, tokenization, hardware and target density apply.\n\n'
    report+=f"Successful BERT timed+warmup windows consumed {spent:.4f} GPU-hours in total; correctness/profiling/import/IO setup and the initial failed numerical check are outside that sum. "
    report+='These measurements indicate the small seven-model BERT family is computationally feasible on the two current PRO6000 GPUs at the requested budgets. '
    report+='They do not establish convergence, functional stability, scientific comparability to GPT-2, or adequacy of100M/300M/1B tokens. No formal training is started.\n\n'
    report+='## Reproduction\n\n```bash\n# In /data/cfs_bert_benchmark_20261116, use the existing task Python.\nPY=/data/cfs_rebuttal_20261114/venv/bin/python\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false\n'
    report+='$PY scripts/prepare_bert_benchmark.py --root "$PWD" --gpt2-metadata gpt2_data_metadata.json --endpoint https://hf-mirror.com\n'
    report+='CUDA_VISIBLE_DEVICES=0 $PY scripts/benchmark_bert_controlled.py --root "$PWD" --gpu 0 --models B1 B3 &\n'
    report+='CUDA_VISIBLE_DEVICES=1 $PY scripts/benchmark_bert_controlled.py --root "$PWD" --gpu 1 --models B2 B5 &\nwait\n'
    report+='# Repeat with --updates 200 --batches 256 --modes pipeline --output-name confirmation\n```\n\n'
    report+='Use a new output directory to rerun; existing completed measurements are preserved. Analysis: `python scripts/analyze_bert_benchmark.py --root results/bert_throughput_20261116`. '
    report+='Implementation references: [HF BERT](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/bert), '
    report+='[PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).\n'
    (root/'REPORT.md').write_text(report,encoding='utf-8')
    print(compare.to_string()); print(pd.DataFrame(totals).to_string(index=False))

if __name__=='__main__': main()
