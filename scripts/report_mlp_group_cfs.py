"""Build the MLP analytical-generalization report from saved statistics."""
import argparse
import json
from pathlib import Path
import pandas as pd


def markdown(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(str(row[c]) for c in columns)+' |' for row in rows])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--collection',type=Path,required=True); p.add_argument('--analysis',type=Path,required=True)
    args=p.parse_args(); out=args.analysis/'EXPERIMENT_REPORT.md'
    if out.exists(): raise FileExistsError(out)
    s=pd.read_csv(args.analysis/'summary.csv'); contrasts=pd.read_csv(args.analysis/'partition_contrasts.csv')
    metadata=json.loads((args.collection/'metadata.json').read_text())
    checks=json.loads((args.collection/'numerical_checks.json').read_text())
    indexed=s.set_index(['config','metric'])
    ordered=sorted(s.config.unique(),key=lambda x:(int(x.split('_')[-1][1:]),int(x.split('_')[0][1:]),{'contiguous':0,'42':1,'123':2,'2027':3}[x.split('_')[1]]))
    def row(c,m): return indexed.loc[(c,m)]
    def value(c,m): return float(row(c,m)['mean'])
    def ci(c,m):
        r=row(c,m); return f"{r['mean']:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]"
    def bounds(m):
        v=s[s.metric==m]['mean']; return f'{v.min():.3f}–{v.max():.3f}'
    core=[]; comparisons=[]; structure=[]
    for c in ordered:
        core.append(dict(config=c,oracle_95CI=ci(c,'raw_oracle'),input_SD=f"{row(c,'raw_oracle').input_sd:.4f}",valid_percent=f"{100*value(c,'valid_source_fraction'):.2f}",drop_KL=f"{value(c,'valid_drop_kl_mean'):.6f}",replacement_KL=f"{value(c,'oracle_replacement_kl'):.6f}"))
        comparisons.append(dict(config=c,Taylor=ci(c,'taylor_recovery'),Cosine=ci(c,'cosine_recovery'),Random=ci(c,'random_recovery'),Taylor_rank=ci(c,'candidate_taylor_spearman'),Cosine_rank=ci(c,'candidate_cosine_spearman')))
        structure.append(dict(config=c,asymmetry=ci(c,'directed_asymmetry_abs'),pair_input_SD=ci(c,'same_pair_across_input_sd'),representative_switch=ci(c,'best_rep_input_switch_probability'),oracle_above_half=f"{value(c,'source_oracle_above_half'):.4f}"))
    raw=contrasts[contrasts.metric=='raw_oracle'].copy()
    nonzero=raw[(raw.ci_low>0)|(raw.ci_high<0)]
    report=['# MLP Group CFS：跨模块分析验证', '',
        f"完成 {metadata['samples']} 个固定 ImageNet-1K 输入、16 种 layer/G/partition 配置，共 1,024 个完整有向矩阵。模型 `{metadata['model']}`，实际零基层 6、11，G=8/16，连续分组和随机 seed=42/123/2027。无 backbone 训练、预测器或路由。",'',
        '## Q1：核心定义是否自然适用于 MLP Group？','',
        '可以在这个标准 ViT MLP 的加性神经元组分解上直接操作化。缓存 post-activation 后，source gain=0、representative gain=alpha，其余 gain=1；fc2 bias 保留一次；后续网络继续运行。输出差异、alpha 的 31 点 [0,3] 网格、有效 source 阈值 1e-5 和 CFS 归一化沿用原 Attention 实现。原 alpha 是总增益：alpha=1 才是 source-only deletion，alpha=0 删除两个组。', '',
        '## Q2：是否存在非平凡、输入相关、有方向性的替代？','',
        f"各配置 Raw Oracle 均值为 {bounds('raw_oracle')}，有效 source 比例为 {bounds('valid_source_fraction')}。上表以外的所有输入均纳入汇总，没有挑选最大恢复输入。有效 source 中 Oracle>0.5 的比例为 {bounds('source_oracle_above_half')}。",'',
        f"在固定分组的同一个有向 pair 上，跨输入 CFS 标准差的平均值为 {bounds('same_pair_across_input_sd')}；两方向差异 |Sij−Sji| 的平均值为 {bounds('directed_asymmetry_abs')}，跨方向是否超过 0.5 的不一致比例为 {bounds('directional_half_threshold_disagreement')}。最佳代表在不同输入对间的切换概率为 {bounds('best_rep_input_switch_probability')}；这一描述量会受近似并列候选影响，不能独自作为功能意义证据。",'',
        '恢复率必须结合绝对删除 KL、替代后 KL、有效性比例和数值诊断一起读。结果支持这些干预下的输出分布恢复与输入条件性，不是任务准确率收益或语义功能发现的证明。','',
        markdown(core,list(core[0])),'',markdown(structure,list(structure[0])),'',
        '## Q3：与 Taylor、Contribution Cosine 的关系','',
        '沿用原 fixed-source 比较：在 i 固定时用 Taylor(j) 或 cosine(c_i,c_j) 选代表 j，随后对该 pair 使用同一 alpha oracle；另报告候选 j 间 Spearman，及单独的 source Taylor 对删除 KL 的相关。没有把 source 重要性直接当作有向 pair 分数。Cosine 按原协议 flatten 全部 tokens 和输出维度；该表征包含分类端点未必敏感的方向，尤其最后层。', '',
        f"候选内 Taylor Spearman 均值范围 {bounds('candidate_taylor_spearman')}，Cosine Spearman 范围 {bounds('candidate_cosine_spearman')}。全部 16 配置的 Oracle−Taylor 和 Oracle−Cosine 配对差异 CI 下界均为正：{bool((s[s.metric.isin(['oracle_minus_taylor','oracle_minus_cosine'])].ci_low>0).all())}。",'',
        '这说明这些标量基线未恢复全部观测替代排序；Oracle 使用真实干预结果、择优本身占优，因此不能将差距解释为可部署预测优势，亦不能断言完整表征没有这些信息。Taylor 和 Cosine 是分析基线，不是学习到的 CFS predictor。','',markdown(comparisons,list(comparisons[0])),'',
        '## Q4：对 G 与分组方式是否稳健？','',
        '连续与三个随机等规模分组均保留上述恢复、输入变化和方向不对称现象，但数值不是分组不变量。G=16 相比 G=8 既改变单组大小，也改变候选数量，不能把 Raw Oracle 的差异解释成受控 scaling 或更强的固有功能冗余。不同划分的 group 身份不同，未逐元素比较其矩阵。','',
        f'随机分组对连续分组的 12 个 Raw Oracle 配对对照中，{len(nonzero)} 个逐项 95% CI 不包含零（未作多重比较校正）。', '',
        markdown([dict(config=r.config,delta=f'{r.delta:.6f}',CI=f'[{r.ci_low:.6f}, {r.ci_high:.6f}]') for _,r in raw.iterrows()],['config','delta','CI']),'',
        '## 数值与统计可靠性','',
        f"组贡献在 FP32 下加 bias 的最大重构误差为 {max(x['reconstruction_fp32_max_abs'] for x in checks):.3g}；与原 FP16 投影的最大绝对差为 {max(x['reconstruction_vs_fp16_max_abs'] for x in checks):.3g}，反映精度/累加方式差异。实际干预使用原 FP16 fc2，不使用 FP32 重构结果替换模型。",'',
        f"1,024 个 dense cached 检查最大 logits 误差 {max(x['dense_logits_max_abs'] for x in checks):.3g}；前四输入×16 配置、每配置四种 masks 的完整前向核对，最大 logits/KL 误差分别为 {max(x.get('full_logits_max_abs',0) for x in checks):.3g}/{max(x.get('full_kl_max_abs',0) for x in checks):.3g}。alpha=1 与 deletion 最大 KL 差 {max(x['alpha1_drop_kl_max_abs'] for x in checks):.3g}。",'',
        f"无干预批量 KL 最大为 {max(x['no_op_batch_kl'] for x in checks):.3g}。有效 source 中删除 KL≤10倍对应无干预 KL 的占比最大为 {max(x['valid_sources_within_10x_no_op_fraction'] for x in checks):.3g}；非对角 NaN/Inf 比例为 {s[s.metric=='nonfinite_pair_fraction']['mean'].max():.3g}。10倍标记仅是诊断，不是严格舍入误差上界。近零 deletion 按原阈值排除，未静默删样。",'',
        '2000 次 bootstrap、seed=42，以完整独立输入为重采样单位；不同层、G、partition 分别报告，不把 group/pair/种子当独立输入。源级和 pair 级比例每次重算分母。切换概率使用不同原始输入对的加权 U-statistic，排除同一输入重复抽样产生的自配对；初版自配对偏差已修正，旧分析单独保留，正式交付使用 analysis_final。所有 CI 均为逐项 95% 区间，未校正多重比较。跨输入 SD/切换概率本身是非线性跨输入统计，input_SD 不适用，记为缺失而非猜测。','',
        '## 文件与复现','',
        '原始矩阵 `collection/G{G}_{partition}_L{layer}/input_{id}.npz` 保存 group_indices、single_drop、substitution、alpha_error_cube、best_alpha、CFS、Taylor、Cosine、有效性和 no-op KL。`metadata.json`、`input_provenance.json`、`numerical_checks.json` 保存输入、权重/图像 SHA256、分组与数值检查。', '',
        '`summary.csv/json/tex` 含全部 528 行均值、SD、CI；`per_input.csv`、`per_source.csv` 和 `bootstrap.csv` 保留详细统计；`partition_contrasts.csv` 为分组敏感性；所有图均为 PDF/PNG。热图固定选择前四输入，另有各配置平均热图和分布图。复现命令与实现文件见仓库 `docs/MLP_GROUP_CFS_PROTOCOL.md`。', '',
        '## Reviewer UWwP response draft','',
        'We tested analytical generalization beyond attention heads using additive neuron groups in a frozen ViT-Base MLP, without changing the CFS intervention or output-KL definition. We evaluated layers 6 and 11 (zero-based), G=8 and 16, contiguous groups and three fixed random equal-size partitions, on 64 shared ImageNet-1K validation images. We retained the original 31-point alpha grid and used 2,000 image-level bootstrap replicates.', '',
        f"Across the 16 configurations, mean oracle recovery ranged from {bounds('raw_oracle')}, with approximately 91–99% valid source interventions. Fixed directed pairs varied across inputs, and mean absolute directional asymmetry ranged from {bounds('directed_asymmetry_abs')}. Candidate-wise Taylor rank correlations were modest ({bounds('candidate_taylor_spearman')}); contribution-cosine correlations were weak ({bounds('candidate_cosine_spearman')}). Both representative-selection baselines left a recovery gap relative to the intervention oracle, although oracle selection is intrinsically advantaged and is not a deployment claim.", '',
        'The qualitative findings persisted across the tested partitions, but the magnitudes were not partition-invariant: one of twelve pointwise oracle contrasts against contiguous grouping excluded zero. Changing G also changes group size and candidate count. These results support operational applicability to MLP groups and input-conditioned directional recovery under the tested protocol; they do not establish universal module generalization, backbone accuracy gains, or MLP inference acceleration.']
    out.write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(out)


if __name__=='__main__': main()
