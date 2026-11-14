"""Write a conservative report from completed results, without filling gaps."""
import argparse
import json
from pathlib import Path
import pandas as pd


def table(frame, columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(f'{row[c]:.6f}' if isinstance(row[c],float) else str(row[c]) for c in columns)+' |')
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--figures',type=Path,required=True); p.add_argument('--analysis',type=Path,required=True); p.add_argument('--historical',type=Path,required=True)
    a=p.parse_args()
    s=pd.read_csv(a.figures/'all_results.csv')
    expected={'Qwen':4,'ViT':3}
    lines=['# CFS Matched-N / Fixed-K 补充实验报告','',
        '这是重新下载公开权重后的复现实验。Qwen 四规模使用新冻结的同一批 FineWeb-Edu contexts；不将其与历史 Qwen-14B 拼接。ViT 使用原模型系列、ImageNet-1K 验证集、原索引选择与预处理方法。未训练任何模型。',
        '', '## 统计约定', '',
        'Matched-N 是均匀无放回候选子集最大值的精确期望。Fixed-K 使用原 Cover（严格 CFS>0.5）及谱熵有效 Rank，除以 K；每输入/层抽样 500 次，seed 42，K=H 仅一次。原有有效 source 阈值与裁剪规则保持不变。',
        '', '95% CI 为 2000 次输入级 percentile bootstrap。Oracle 在每层内按有效 source 等权、再按层等权；每次 bootstrap 重新计算分子/分母。相邻差异使用同一冻结输入的配对 bootstrap。Fixed-K 的子图 Monte Carlo SE 单独报告，不能把子图当作独立输入。所有 CI 均未作多重比较校正。',
        '', '## 完整结果', '']
    for family,count in expected.items():
        g=s[s.family==family]
        lines += [f'### {family}：{g.model.nunique()}/{count} 个模型完成','']
        if not g.empty:
            lines += [table(g,['model','metric','mean','ci_low','ci_high','input_sd','subgraph_mc_se']),'']
    contrast_path=a.figures/'paired_contrasts.csv'
    try: contrasts=pd.read_csv(contrast_path)
    except pd.errors.EmptyDataError: contrasts=pd.DataFrame()
    lines += ['## 相邻规模差异与单调性','']
    if not contrasts.empty: lines += [table(contrasts,['family','left','right','metric','delta','ci_low','ci_high']),'']
    trends=pd.read_csv(a.figures/'trends.csv')
    lines += [table(trends,['family','metric','models','monotonic_increasing','monotonic_decreasing']),'']
    checks=[]
    for path in sorted(a.analysis.glob('*/consistency_checks.json')):
        c=json.loads(path.read_text())
        checks.append(dict(model=c['model'],valid_source_fraction=c['valid_source_fraction'],matched_2_max_absolute_error=c['matched_2_max_absolute_error'],K_equals_H_checks=','.join(c['K_equals_H_checks']) or 'not applicable'))
    lines += ['## 与原实现及原论文表的一致性','']
    if checks: lines += [table(pd.DataFrame(checks),list(checks[0])),'']
    oldrows=[]
    mapping={'raw':'raw_oracle','matched_2':'matched_2','cover_full':'cover_per_head_tau_0.5','rank_full':'rank_per_head'}
    for family in expected:
        path=a.historical/f'natural_scaling_{family.lower()}.csv'
        if not path.exists(): continue
        old=pd.read_csv(path).set_index('parameters')
        for _,row in s[(s.family==family)&s.metric.isin(mapping)].iterrows():
            if row.parameters in old.index:
                value=old.loc[row.parameters,mapping[row.metric]]
                oldrows.append(dict(family=family,model=row.model,metric=row.metric,historical=value,rerun=row['mean'],delta=row['mean']-value))
    if oldrows:
        pd.DataFrame(oldrows).to_csv(a.figures/'historical_comparison.csv',index=False)
        lines += [table(pd.DataFrame(oldrows),list(oldrows[0])),'']
    lines += ['同矩阵对照检查实现一致性；与论文汇总表的差异还包含输入、硬件和数值环境差异。历史 Qwen 输入缺失，不能对历史表做逐输入配对，也不能把接近的数值写成完全复现。','',
              '## 抽样稳定性','',table(pd.read_csv(a.figures/'sampling_stability_summary.csv'),['model','metric','draws','mean']),'',
              '## Q1：固定候选数后的趋势','']
    for family,count in expected.items():
        group=s[s.family==family]
        if group.model.nunique()!=count:
            lines += [f'{family} 尚未全部完成，暂不判断完整 scaling 趋势。','']; continue
        for _,row in trends[(trends.family==family)&trends.metric.str.startswith('matched_')].iterrows():
            direction='单调增加' if row.monotonic_increasing else '单调减少' if row.monotonic_decreasing else '非单调'
            lines += [f"{family} {row.metric} 的均值随规模{direction}。"]
        lines += ['']
    lines += ['## Q2：固定子图规模后的趋势','']
    for family,count in expected.items():
        if s[s.family==family].model.nunique()!=count:
            lines += [f'{family} 尚未全部完成。']; continue
        selected=trends[(trends.family==family)&trends.metric.str.startswith(('cover_','rank_'))&~trends.metric.str.endswith('full')]
        for _,row in selected.iterrows():
            direction='单调增加' if row.monotonic_increasing else '单调减少' if row.monotonic_decreasing else '非单调'
            lines += [f"{family} {row.metric} 的均值随规模{direction}。"]
    lines += ['', '## Q3 与 Reviewer nLkz 的回应依据','']
    if not contrasts.empty:
        for family,count in expected.items():
            g=contrasts[contrasts.family==family]
            if g.empty: continue
            scope='完整系列' if s[s.family==family].model.nunique()==count else '目前已完成规模'
            lines += [f'{family}（{scope}）：']
            for label,prefix in [('固定候选数','matched_'),('固定子图 Cover','cover_'),('固定子图 Rank','rank_')]:
                subset=g[g.metric.str.startswith(prefix)&~g.metric.str.endswith('full')]
                uncertain=subset[(subset.ci_low<=0)&(subset.ci_high>=0)]
                lines += [f'{label}共有 {len(subset)} 个相邻规模/配置对照，其中 {len(uncertain)} 个配对 95% CI 包含零。']
                if len(uncertain):
                    pairs='；'.join(f'{r.left}→{r.right} {r.metric}' for _,r in uncertain.iterrows())
                    lines += [f'这些对照尚不足以支持明确的相邻规模差异：{pairs}。']
            lines += ['']
    lines += [
        '应逐个 N/K 对照上表和配对差异 CI：若控制后保持原方向，可称该已测趋势对相应候选数或子图规模控制稳健；若方向改变、非单调或相邻差异 CI 包含零，应如实弱化相应结论。不能仅凭均值排序或少数规模点的相关系数声称统计显著。',
        '', '这些控制排除了当前设置下的一部分候选数量解释，但没有固定模型深度、宽度、GQA、预训练或有效 source 比例。特别是 ViT-L 的 validity 过滤可能影响 Cover/K 和 Rank/K。结果仍是模型系列内的描述性关联，不是规模导致功能替代性的因果证据，也不是可执行联合压缩率或任务效用证明。',
        '', '图表：`figure1_matched_N.pdf/png`、`figure2_fixed_K.pdf/png`；完整数值 `all_results.csv/json/tex`，配对差异 `paired_contrasts.csv`，逐输入与全部 bootstrap replicate 位于各模型分析目录。']
    out=a.figures/'EXPERIMENT_REPORT.md'
    if out.exists(): raise FileExistsError(out)
    out.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
