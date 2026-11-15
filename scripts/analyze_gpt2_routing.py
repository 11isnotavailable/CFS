"""Document-paired KL/NLL/PPL statistics and fixed cross-layer report."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--layers',default='11,8');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=False);layers=[int(v) for v in args.layers.split(',')]
    frames=[];dense=[];times=[];training=[];teacher=[]
    for layer in layers:
        d=root/f'layer{layer}/evaluation';a=pd.read_csv(d/'per_input.csv');a['layer']=layer;frames.append(a)
        b=pd.read_csv(d/'dense_per_input.csv');b['layer']=layer;dense.append(b)
        c=pd.read_csv(d/'timings.csv');c['layer']=layer;times.append(c)
        for run in (root/f'layer{layer}/full/runs').glob('*'):
            if not run.is_dir():continue
            c=json.loads((run/'config.json').read_text());done=json.loads((run/'COMPLETE.json').read_text())
            training.append({**c,**done})
        for split in ('train','tune'):
            paths=sorted((root/f'layer{layer}/pair/{split}').glob('[0-9]*.npz'))
            drops=np.stack([np.load(path)['drops'] for path in paths]);teacher.append(dict(layer=layer,split=split,documents=len(paths),valid_drop_fraction=float((drops>1e-5).mean()),
                minimum_drop=float(drops.min()),median_drop=float(np.median(drops)),maximum_drop=float(drops.max())))
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'per_input.csv',index=False);frame.to_json(out/'per_input.json',orient='records')
    draws=np.random.default_rng(42).integers(0,256,(2000,256));np.savez_compressed(out/'bootstrap_draws.npz',draws=draws)
    def stat(x,transform=False):
        x=np.asarray(x,float);assert len(x)==256 and np.isfinite(x).all()
        boot=x[draws].mean(-1);point=x.mean()
        if transform:boot=np.exp(boot);point=np.exp(point)
        lo,hi=np.quantile(boot,[.025,.975])
        return dict(mean=float(point),ci_low=float(lo),ci_high=float(hi),sd_inputs=float(x.std(ddof=1)) if not transform else np.nan)
    avg=frame.groupby(['layer','sample','method','keep'])[['kl','nll','nll_difference']].mean().reset_index()
    summary=[];seed_rows=[];paired=[];paired_inputs=[]
    for (layer,method,k,seed),g in frame.groupby(['layer','method','keep','seed']):
        g=g.sort_values('sample')
        for metric in ('kl','nll','ppl','nll_difference'):
            seed_rows.append(dict(layer=layer,method=method,keep=k,seed=seed,metric=metric,**stat(g['nll' if metric=='ppl' else metric],metric=='ppl')))
    seed_table=pd.DataFrame(seed_rows)
    for (layer,method,k),g in avg.groupby(['layer','method','keep']):
        g=g.sort_values('sample')
        for metric in ('kl','nll','ppl','nll_difference'):
            sd=seed_table.query('layer==@layer and method==@method and keep==@k and metric==@metric')['mean'].std(ddof=1)
            summary.append(dict(layer=layer,method=method,keep=k,metric=metric,seed_mean_sd=float(sd),**stat(g['nll' if metric=='ppl' else metric],metric=='ppl')))
    for (layer,k),g in avg.groupby(['layer','keep']):
        for metric in ('kl','nll'):
            table=g.pivot(index='sample',columns='method',values=metric)
            for reference in ('Taylor','Gating','A0'):
                paired.append(dict(layer=layer,keep=k,metric=metric,contrast=f'R2_minus_{reference}',**stat(table.R2-table[reference])))
                paired_inputs.extend(dict(layer=layer,keep=k,metric=metric,contrast=f'R2_minus_{reference}',sample=int(sample),difference=float(value))
                    for sample,value in (table.R2-table[reference]).items())
    d=pd.concat(dense);dense_summary=[]
    for layer,g in d.groupby('layer'):
        for metric in ('nll','ppl'):dense_summary.append(dict(layer=layer,metric=metric,**stat(g.sort_values('sample').nll,metric=='ppl')))
    summaries=pd.DataFrame(summary);pair_table=pd.DataFrame(paired)
    pd.DataFrame(paired_inputs).to_csv(out/'paired_per_input.csv',index=False)
    tables=dict(summary=summaries,paired=pair_table,per_seed=seed_table,dense=pd.DataFrame(dense_summary),teacher_validity=pd.DataFrame(teacher),training=pd.DataFrame(training))
    for name,t in tables.items():
        t.to_csv(out/f'{name}.csv',index=False);t.to_json(out/f'{name}.json',orient='records',indent=2);t.to_latex(out/f'{name}.tex',index=False,float_format='%.6f')
    time_frame=pd.concat(times);time_frame.to_csv(out/'timings_per_input.csv',index=False)
    time_summary=time_frame.groupby(['layer','method','scope']).seconds.agg(['mean','median','std']).reset_index();time_summary.to_csv(out/'timings.csv',index=False)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,len(layers),figsize=(3.5*len(layers),2.8),squeeze=False)
    colors={'Taylor':'#777777','Gating':'#0072B2','A0':'#009E73','R2':'#D55E00'}
    labels={'Taylor':'Taylor (supervised)','Gating':'Learned Gating','A0':'Joint-only R2','R2':'Original CFS R2'}
    for ax,layer in zip(axes[0],layers):
        for offset,(method,color) in enumerate(colors.items()):
            r=summaries.query('layer==@layer and method==@method and metric=="kl"').sort_values('keep')
            ax.errorbar(r.keep+(offset-1.5)*.08,r['mean'],yerr=np.stack([r['mean']-r.ci_low,r.ci_high-r['mean']]),fmt='o-',ms=3,capsize=2,label=labels[method],color=color)
        ax.set(title=f'Pretrained GPT-2, layer {layer}',xlabel='Retained heads K',ylabel='Next-token KL to dense',xticks=[3,6,9]);ax.legend(frameon=False,fontsize=6)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'figure_routing_kl.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    def md(t):return t.to_markdown(index=False,floatfmt='.6f')
    kl=summaries.query('metric=="kl"');table=kl.pivot(index=['layer','method'],columns='keep',values='mean').reset_index()
    table=pd.concat([table[table.layer==layer] for layer in layers],ignore_index=True)
    (out/'main_kl_table.tex').write_text(table.to_latex(index=False,float_format='%.6f'))
    ci_table=kl.copy();ci_table['mean [95% CI]']=ci_table.apply(lambda r:f'{r["mean"]:.5f} [{r.ci_low:.5f}, {r.ci_high:.5f}]',axis=1)
    ci_table.pivot(index=['layer','method'],columns='keep',values='mean [95% CI]').to_latex(out/'main_kl_ci_table.tex')
    runtime=pd.concat([pd.read_csv(root/f'layer{layer}/runtime/summary.csv') for layer in layers],ignore_index=True)
    runtime.to_csv(out/'runtime.csv',index=False);runtime.to_json(out/'runtime.json',orient='records',indent=2)
    provenance=json.loads((root/'provenance.json').read_text())
    conclusions=[];english=[]
    for layer in layers:
        r=summaries.query('layer==@layer and method=="R2" and metric=="kl"').sort_values('keep')
        english.append(f"At layer {layer}, R2 achieved mean KL {', '.join(f'{v:.6f}' for v in r['mean'])} for K=3/6/9, respectively.")
        for reference in ('Taylor','Gating','A0'):
            p=pair_table[(pair_table.layer==layer)&(pair_table.metric=='kl')&(pair_table.contrast==f'R2_minus_{reference}')]
            lower=list(p.loc[p.ci_high<0,'keep'].astype(int));higher=list(p.loc[p.ci_low>0,'keep'].astype(int));cross=list(p.loc[(p.ci_low<=0)&(p.ci_high>=0),'keep'].astype(int))
            conclusions.append(f'- Layer {layer}，R2 vs {reference}：配对KL区间全部低于零的K={lower}；全部高于零的K={higher}；跨零的K={cross}。')
            english.append(f"For R2 minus {reference} at layer {layer}, paired 95% document-bootstrap intervals were below zero at K={lower}, above zero at K={higher}, and included zero at K={cross}.")
    answers='''### 对研究问题的回答

**Q1（可迁移性）**：已将原R2的网络、CFS监督、结构化Utility与精确选择迁移到预训练GPT-2，完成独立训练与真实下一Token Hard Gating；必要的变化是语言前缀特征与固定预测位置。它验证了该协议在语言模型上的可实施性，不是保持原ViT权重的零样本迁移。

**Q2（相对效果）**：以下按输入配对区间逐项判断。区间跨零只表示本实验尚不能明确区分，不表示两者等效。多个预算/层的区间未作多重比较校正。

'''+ '\n'.join(conclusions)+'''

**Q3（CFS实际价值）**：跨模型路由行为保持能力应由R2对Taylor/Gating的结果判断；CFS额外监督的增量作用必须另看R2对A0，不能将结构化子集建模与CFS标签的贡献混为一谈。当前实现没有跳过Dense Attention核，不能声称推理加速。数据仅来自一个冻结FineWeb-Edu缓存、256个文档的一个预测位置，不能推广为普遍语言建模质量收益。

**与ViT的关系**：已有[ViT公平对照](../learned_gating_rebuttal_20261115/REPORT.md)中，原R2对Gating的点估计优势集中在K=9；本轮各层的预算依赖性见上述完整结果。仅比较相对模式，不比较不同输出空间的绝对KL，也不将某一预算模式写成跨任务定律。

### Reviewer WCQn: proposed response

We extended the original CFS R2 router to pretrained GPT-2 (124M), with independently trained routers at layers '''+', '.join(map(str,layers))+'''. The backbone remained frozen. All learned methods used the same causal prefix features and joint-subset teachers; the main evaluation used hard gating without method-specific compensation. Each learned method was trained with three fixed seeds on 2,048 documents, selected on 128 validation documents by actual routing KL, and evaluated on 256 held-out documents. '''+' '.join(english)+''' These intervals resample documents after averaging the three training seeds and are not multiplicity-adjusted. Taylor uses the true next-token label and a backward pass, so it is an analytical baseline. The Joint-only R2 ablation separates the structured subset architecture from additional CFS supervision; the comparisons above limit what can be attributed specifically to that supervision. This provides a cross-model, next-token routing validation, not evidence of universal superiority, full-corpus perplexity improvement, or realized inference acceleration.
'''
    expected={(11,'Gating'):([9],[]),(11,'A0'):([],[]),(8,'Gating'):([6,9],[3]),(8,'A0'):([6,9],[3]),
        (11,'Taylor'):([3,6,9],[]),(8,'Taylor'):([6,9],[])}
    pattern_matches=set(layers)=={11,8}
    for (layer,reference),(lower,higher) in expected.items():
        rows=pair_table[(pair_table.layer==layer)&(pair_table.metric=='kl')&(pair_table.contrast==f'R2_minus_{reference}')]
        pattern_matches=pattern_matches and list(rows.loc[rows.ci_high<0,'keep'])==lower and list(rows.loc[rows.ci_low>0,'keep'])==higher
    if pattern_matches:
        interpretation='''本轮支持**有条件的跨模型、跨层路由价值**。Layer11的R2三个预算均值均低于Taylor/Gating，但相对Gating只有K=9的配对区间完全低于零；相对A0三个预算均跨零。Layer8在K=6、9相对Gating和A0均有区间支持的改善，却在K=3明确差于两者。因此，额外CFS监督的收益依赖层与预算，不能总结为普遍优于学习门控。

辅助任务指标须另看NLL配对表；低KL不等于提升真实下一Token预测质量。'''
        def values(layer,method):
            return '/'.join(f'{v:.5f}' for v in summaries.query('layer==@layer and method==@method and metric=="kl"').sort_values('keep')['mean'])
        interval=pair_table.query('layer==11 and keep==9 and metric=="kl" and contrast=="R2_minus_Gating"').iloc[0]
        reply=f'''We extended the original CFS R2 router to pretrained GPT-2 (124M), independently at layers 11 and 8, without training the backbone. R2, parameter-matched Learned Gating (within 0.3%), and Joint-only R2 used identical causal prefix features and joint-subset teachers. We trained three seeds on 2,048 documents, selected checkpoints on 128 validation documents using actual hard-routing KL, and evaluated 256 held-out documents.

At layer 11, R2 achieved KL {values(11,'R2')} for K=3/6/9, compared with Gating's {values(11,'Gating')}. R2-minus-Gating's paired 95% CI excluded zero only at K=9: [{interval.ci_low:.5f}, {interval.ci_high:.5f}]. At layer 8, R2 achieved {values(8,'R2')}. It improved over both Gating and Joint-only R2 at K=6 and K=9, with paired intervals below zero, but performed worse than both at K=3. At layer 11, all R2-versus-Joint-only intervals included zero. R2 outperformed the supervised Taylor baseline at layer 11 and at layer 8's K=6/9; the layer-8 K=3 comparison was inconclusive.

These results extend routing validation beyond one vision model and one layer, while showing that the incremental value of CFS supervision depends on layer and budget. Intervals resample documents after averaging three training seeds and are not multiplicity-adjusted. We also report NLL/PPL and measured overhead; this experiment does not establish universal task-quality gains or realized inference acceleration.'''
        answers=answers.split('### Reviewer WCQn: proposed response')[0]+'### Reviewer WCQn: proposed response\n\n'+reply
    else:interpretation='以逐预算配对区间判断可迁移性与额外监督作用，不从点估计推导普遍优势。'
    (out/'REBUTTAL_WCQn.md').write_text(answers.split('### Reviewer WCQn: proposed response\n\n')[1],encoding='utf-8')
    text=f'''# Cross-model CFS Dynamic Routing: pretrained GPT-2

日期：2026-11-15。模型openai-community/gpt2，revision `{provenance['revision']}`，124,439,808参数；实际Config12层/12头/768宽。Backbone完全冻结。完成层：{layers}，每层独立训练Router与Teacher。主方法始终为原始R2，未根据测试结果改用A3或其他方法。

{interpretation}

## 主结果：真实Hard Gating Output KL

越低越好；平均三个固定训练seed17/23/41。Taylor是使用真实下一Token标签的梯度分析基线，不是无标签前向生成方法。

{md(table)}

![GPT-2 routing KL with document-bootstrap 95% confidence intervals](figure_routing_kl.png)

[矢量PDF](figure_routing_kl.pdf)。A0表示Joint-only R2；两层均为0-based索引。

完整KL与95%区间：

{md(kl[['layer','method','keep','mean','ci_low','ci_high']])}

### R2减参考方法的配对差

负值有利于R2。区间跨零不表示等效，也不从均值差直接宣称显著优势。

{md(pair_table)}

{answers}

## NLL与Perplexity

仅在每文档预先固定的第127位置预测第128个Token；不是全语料PPL。统一Token NLL求平均后取exp，不能平均逐Token exp(NLL)。

Dense：

{md(tables['dense'])}

Routed：

{md(summaries.query('metric in ["nll","ppl"]')[['layer','method','keep','metric','mean','ci_low','ci_high']])}

KL衡量保持Dense行为；NLL/PPL衡量对真实下一Token的损失，二者不必同向。逐Token配对NLL差、每seed结果、输入SD和训练seed均值SD均保存。

## 可复用实现与必要适配

复用原GPT2HeadGate在c_proj前缩放AV分支，保留输出投影Bias一次；原DynamicCFSRouter、subset_utilities、精确K子集搜索、loss_terms、joint_ranking_loss、七类锚点/local-swap candidate_pool均直接复用。R2损失为listwise+.5score+.5importance+.1alpha+structured+.5joint；A0/Gating为.5joint。没有用A3、残差网络或新的loss替代R2。

输入为ln_1输出的[current token; prefix mean; population variance]，维度2304，只使用当前预测位置及之前Token。ln_1站点对应ViT原norm1捕获站点；这是明确的语言特征适配。均值/标准差只从训练集计算并冻结。在线GPT2RuntimeRouter在ln_1 Hook内生成Head选择，不需要Dense logits、真实CFS或Joint-KL。它提供完整前缀到下一Token的接口，不宣称同次Forward中更早位置的共享Gate实现了逐Token因果Routing；不使用KV cache，不接受填充前缀。

每输入128Token无Padding，预测位置127，目标128。这与旧Controlled GPT-2的随机8位置平均不同；本轮是独立跨模型路由验证，不能冒充原Scaling数值复现。原GPT-2脚本的温度1、KL方向及clamp_min0保留。Backbone采用FP32且关闭TF32，目的是稳定小KL；所有方法统一精度。未改变CFS公式或数值分母1e-12，target仍nan->-1且clip[-1,1]，utility仍clip[0,1]。

Alpha训练16点/验证31点均匀[0,3]，沿用ViT Routing的训练/验证精度约定；旧GPT Scaling是16点。alpha=1对应删除Source后的不补偿候选，alpha=0同时删除两者。有效删除阈值1e-5仅作为诊断，未擅自增加原R2没有的训练筛选。

{md(tables['teacher_validity'])}

## 数据、训练与选择

使用既有FineWeb-Edu sample-10BT缓存parquet，按文档与129-token片段SHA去重，固定train2048/tune128/test256。一个文档一个位置，无同文档跨集合泄漏。来源是固定缓存文件的样本，不是官方独立语言模型基准；缓存原始dataset revision历史未记录，明确为null并保存文件SHA，模型revision及权重SHA完整保存。语料文本不用于测试后挑选。

32前缀真实模型Smoke后，pilot256训练/64验证/单seed；检查数值和资源后才扩大。每层先用pair-only seed17 bootstrap生成旧Router锚点，接着原七锚点+局部交换生成每K48个真实联合删除标签。bootstrap只用于Teacher候选，不是主结果。最终三种学习方法各3种子，从头训练60epochs，AdamW3e-4/decay1e-4/batch64，无架构、学习率或预算权重搜索。Pilot不依据测试选择方向。

为避免之前发现的候选池验证代理偏差，本轮每10epochs在全部220/924/220候选中真实路由，再用缓存后缀测量验证KL；按三预算均值最低选择，各方法完全相同。这里改进的是Checkpoint选择口径，R2数学目标/结构不变。九个模型锁定后才读取该层测试结果；Layer8由事先数值/成本检查决定，未根据Layer11测试胜负决定是否开展。

R2总/训练参数825,091；Gating822,460；A0总825,091、实际训练824,962（Alpha129参数冻结）。参数差约0.3%，报告为近似匹配，不能称完全相等。R2额外使用CFS/删除/Alpha监督；A0区分结构与额外监督贡献，不能把R2与Gating的全部差异都归因于CFS。

## 正确性、运行成本和限制

两层均验证Gate全1、单头删除、双头与零门控、Bias重构、完整Forward与缓存后缀、未来Token对前缀logit不影响。32输入Smoke的最终logit绝对差阈值3e-4。在线测试初始检查因大偏置logit的FP32批量GEMM累加误差触发绝对阈值而暂停；在32训练输入独立复查后，对在线与缓存输出采用atol=1e-4/rtol=1e-5，并额外要求KL差<2e-5，未调整模型选择或结果。测试初始4输入对三学习方法/3预算逐项核验Mask完全一致，误差见evaluation/checks.json；训练复查见precision_layer11.json。68项本地单元测试通过，新增5项GPT-2测试也在正式远端依赖环境通过；其中小型随机网络仅作数值测试，正式实验使用公开预训练权重。

Head scales作用在融合投影前，当前实现仍执行Dense QKV与原投影；减少逻辑Head预算不等于减少实际全部FLOPs，更不等于实现加速。下表是Router与精确搜索的实测开销，三个预算一起计时；Taylor计的是监督后缀Backward。Dense前缀捕获及归一化不在此计时，故不是优化部署的端到端吞吐比较。

{md(time_summary)}

另在32个固定训练前缀、每个重复3次、batch1上测量完整在线前向（含Backbone、特征归一化、Router、子集搜索与LM Head；不含模型构建、Tokenization和传输；seed17）。以下是当前未优化Python Hook实现的延迟，不是理论最低成本。中位数相对Dense>1表示更慢。

{md(runtime)}

所有统计按独立文档Bootstrap2,000次、seed42；先在文档内平均训练种子，不能将三个种子当作额外文档。Taylor没有训练seed，其重复记录使用完全相同Mask，批量不同位置可能有极小FP32差异，三次值也在文档内平均，不能解释为训练不稳定性。一个位置/文档，因此Token配对与文档配对在这里一致。区间未作多重比较校正，且条件于三个训练种子。不可跨ViT/GPT-2直接比较绝对KL大小。模型性能结论以完整配对表为准，不从单点优势推导因果或一般跨任务保证。

## 复现和交付

src/gpt2_routing.py：缓存干预与真正前向Routing；prepare_gpt2_routing.py：锁定模型/文档；smoke_gpt2_routing.py：真实检查和Benchmark；collect_gpt2_routing_teacher.py：可续跑Pairwise Teacher；collect_gpt2_joint_teacher.py：原候选规则与缓存Joint-KL；train_gpt2_routing.py：原R2与公平对照；run_gpt2_routing.py：有限阶段；evaluate_gpt2_routing.py：锁定测试；analyze_gpt2_routing.py：统计、图表与报告。所有输出独立保存。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4
mkdir -p RUN
cp -r scripts src tests docs RUN/
python scripts/prepare_gpt2_routing.py --root RUN --endpoint https://hf-mirror.com --parquet EXISTING_FINEWEB.parquet
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_gpt2_routing.py --root RUN --layer 11
CUDA_VISIBLE_DEVICES=1 python scripts/run_gpt2_routing.py --root RUN --layer 11
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_gpt2_routing.py --root RUN --layer 11
# 同样独立完成layer8的smoke/run/evaluate；按实测成本决定，不能以测试结果选择
python scripts/benchmark_gpt2_routing.py --root RUN --layer 11
python scripts/benchmark_gpt2_routing.py --root RUN --layer 8
python scripts/verify_gpt2_routing.py --root RUN
python scripts/analyze_gpt2_routing.py --root RUN --layers 11,8 --output RUN/analysis
```

须在RUN中提供repo的scripts/src；数据目录与非空训练输出不覆盖。Teacher逐文档原子保存，可跳过已核对身份的完成文档。训练仅对未开始的空目录重试；已完成阶段由COMPLETE标记跳过，不把缺失Optimizer状态伪装成精确恢复。

公开模型/接口资料：[模型配置](https://huggingface.co/openai-community/gpt2/blob/{provenance['revision']}/config.json)、[GPT-2接口](https://huggingface.co/docs/transformers/model_doc/gpt2)。实际加载Config和本地Transformers版本以归档记录为准。
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8');print(table.to_string(index=False));print(pair_table.query('metric=="kl"').to_string(index=False))


if __name__=='__main__':main()
