"""Tail-loss audit and decision-focused interpretation of the fixed diagnostic."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve();out=root/'analysis'
    compact=pd.read_csv(out/'compact_summary.csv');summary=pd.read_csv(out/'diagnostic_summary.csv')
    raw=pd.read_csv(root/'collected/per_input.csv');cr=pd.read_csv(root/'collected/cross_method.csv')
    averaged=raw.groupby(['split','sample','method','keep']).mean(numeric_only=True).reset_index()
    tails=[];decomposed=[]
    for (split,method,k),g in averaged.groupby(['split','method','keep']):
        v=g.regret.to_numpy();q=np.quantile(v,[.5,.9,.95,.99]);count=max(1,int(np.ceil(len(v)*.1)))
        tails.append(dict(split=split,method=method,keep=k,n=len(v),median=q[0],p90=q[1],p95=q[2],p99=q[3],maximum=v.max(),
            worst10pct_regret_share=np.sort(v)[-count:].sum()/v.sum()))
    for (split,k),g in cr.groupby(['split','keep']):
        g=g.groupby('sample').mean(numeric_only=True);delta=g.A3_kl-g.R2_kl
        decomposed.append(dict(split=split,keep=k,n=len(g),mean_A3_minus_R2=delta.mean(),
            fraction_A3_better=float((delta< -1e-8).mean()),fraction_R2_better=float((delta>1e-8).mean()),
            positive_loss_contribution=float(delta.clip(lower=0).mean()),negative_loss_contribution=float(delta.clip(upper=0).mean()),
            maximum_A3_disadvantage=float(delta.max()),maximum_R2_disadvantage=float(-delta.min())))
    tails=pd.DataFrame(tails);decomposed=pd.DataFrame(decomposed)
    for name,t in [('regret_tail_descriptive',tails),('A3_R2_loss_decomposition',decomposed)]:
        t.to_csv(out/f'{name}.csv',index=False);t.to_json(out/f'{name}.json',orient='records',indent=2);t.to_latex(out/f'{name}.tex',index=False,float_format='%.7f')
    def md(df):return df.to_markdown(index=False,floatfmt='.6f')
    headline=compact[compact.method=='A3'][['split','keep','chosen_kl','oracle_kl','regret','exact_best','top5_retrieval','top5_regret_removable_fraction','far_regret_share','full_choice_known']]
    regret_rows=summary[summary.metric=='regret'][['split','method','keep','mean','ci_low','ci_high']]
    paired=pd.read_csv(out/'paired_methods.csv');paired=paired[(paired.metric=='chosen_kl') & (paired.left=='A3')]
    coverage=compact[['split','method','keep','full_choice_known']]
    union=pd.read_csv(out/'union_bounds.csv');union=union[(union.method=='A3') & (union.quantity=='oracle_KL_reduction')]
    cross=pd.read_csv(out/'disagreement_summary.csv');cross=cross[cross.metric.isin(['A3_R2_pool_disagree','A3_R2_full_disagree','A3_R2_full_swaps'])]
    queue=pd.read_csv(out/'unmeasured_training_candidate_queue.csv')
    text=f'''# Selection Regret / Candidate Ranking 诊断

日期：2026-11-15；来源commit3b56ba1。**没有训练模型，没有运行Backbone，没有补采Teacher，也没有读取旧测试或256张确认集。** 固定A3、R2、Gating和Joint-only R2(A0)，3种子，4096训练输入与64调参验证输入。

## 结论与下一步

**有明显的候选内决策改进空间，值得做一次有限的当前模型分歧候选补标诊断；证据不足以直接启动新一轮训练，也不支持把瓶颈限定为“最优前几名之间的小错排”。**

1. A3的K9验证池中最优命中率约28.1%，全局Spearman约0.816、总体pair accuracy约0.830，但真实前五名内部pair accuracy仅约0.603。预测Top5的Oracle重排可消除约77.5%的现有候选内regret。这是有标签的上界，不是已经实现的选择器收益。
2. K9验证池约96.1%的A3 regret来自所选候选的真实排名>5，而不是仅在真实前五名内相互错排。训练池对应约88.3%。应检验如何避免高损失错误、找回被低估的好候选，不能仅优化best-vs-second命中率。
3. R2的K9最优命中率更低，却有更低的平均KL。损失间距、错误大小和尾部输入都影响结论，单看排名正确率会误导。下面给出损失分解，未将尾部解释提升为已识别的训练因果机制。
4. 训练池确实覆盖不全，但不是随机48个候选：已经包含Oracle/旧Router等七类锚点及局部交换邻居。A3全候选赢家在训练池中的覆盖率K3/K6/K9约96.7%/80.1%/84.9%；R2约99.7%/93.5%/97.1%。候选覆盖风险集中于部分输入，不能仅从48/924的比例断言大部分决策都是外推。
5. 现有A3、R2、Gating、A0已经使用listwise Joint ranking目标。此次新问题是原始KL决策成本、局部/尾部错误及当前候选分布的适配，**不是第一次把MSE换成Ranking Loss**。上一轮新残差才是MSE目标。

下一步若继续，应先测一个冻结的小诊断集：按训练输入上的当前A3/R2分歧选定候选，与随机训练输入对照；仅补真实联合删除KL，核查分歧是否带来可重复的决策损失和未标注候选盲区。不要先扩大模型、调Loss或开始Adapter训练。本轮已输出 {len(queue)} 个去重的未测训练候选清单（每预算按平均Head交换数选64个输入），**未执行补标，也不能将该高分歧清单当作总体代表样本**。后续训练与跨模型验证的选择仍应结合该诊断的成本/收益；本轮不触发二者。

## 定义、协议与来源核对

每输入每预算使用同一48个已有真实KL候选。regret=D(所选)-min_48 D；原utility方向为越大越好，原裁剪/聚合不变。前m短名单修复只在模型预测Top-m中用真实标签选最好者。改善比例=总regret减少量/总regret，不平均不稳定的逐输入比值。全候选规模为220/924/220，48/总数只是集合大小比例，不是随机覆盖概率。

原Teacher代码`collect_joint_subset_teacher.py::candidate_pool`先插入七个锚点，优先填充pairwise CFS Oracle与旧Router锚点的一次交换邻居，不足才随机补齐。训练256个shard与验证4个shard的candidate_masks和joint_kl已与导出数据逐项完全匹配，见selection_teacher_provenance.json。旧Router锚点不是本轮所有A3/R2/Gating的当前决策，因此仍可能需要刷新困难候选。

`joint_ranking_loss`先对每输入、每预算KL按候选均值/标准差标准化，再形成温度0.5的soft目标，预测utility温度0.05，优化listwise交叉熵。它不直接等同于最小化部署argmax选择的原始KL，而且移除了输入内KL绝对尺度；这解释了为什么值得检查目标适配，**但当前诊断不能证明标准化或某一辅助项是失败原因**。

训练输入属于拟合内数据；64个tune输入用于历史Checkpoint选择。因此这些不是新的独立泛化结果。未复用已查看的测试/确认集来选择新方法，未利用真实标签构造一个部署路由器。

## A3概览

概率和比例均按0–1报告；全表及95% CI在analysis目录。

{md(headline)}

## 同一候选池的完整Regret与95% CI

{md(regret_rows)}

### A3减参考方法的配对KL差

负值有利于A3。由于相同候选Oracle，这也等于regret差。

{md(paired[['split','keep','right','mean','ci_low','ci_high']])}

## 命中率与平均损失并不等价

下表包括全部方法、训练/验证、所有预算。exact_best采用1e-8的数值容差；容差不代表功能等效。分数完全相同的pair计0.5，真实KL相同的pair不进入正确率；记录预测最高分并列数量，不人为打破并列创造信息。

{md(compact[['split','method','keep','exact_best','chosen_true_rank','spearman','pair_accuracy','top5_pair_accuracy','regret']])}

先对每输入平均三个seed，再作A3-R2损失分解：positive项为A3较差的损失贡献，negative项为A3较好的贡献，二者相加为总体配对差。它们是描述性分解，不是按结果筛选后再宣布模型优胜。

{md(decomposed)}

各方法regret的p50/p90/p95/p99、最大值及最差10%输入贡献另存regret_tail_descriptive.csv；这些尾部分位数是描述量，没有伪装成带Bootstrap区间的推断。

具体地，K9训练池中A3在约58.8%的输入上KL低于R2，R2在37.5%上更好，其余近似相同；但A3较差输入的平均正损失贡献0.001525，大于其较好输入的负贡献绝对值0.001192，合成平均差+0.000333。A3最差10%训练输入贡献约63.0%的regret，R2约55.5%。验证池A3在56.25%输入上更好，但平均差仍为+0.000957。这里都是三个seed先在每输入内平均后的描述；不能声称R2在所有尾部分位数都优于A3，也不能证明某一Loss项造成该分布。

## 模型分歧与Oracle重排上界

{md(cross[['split','keep','metric','mean','ci_low','ci_high']])}

从A3/R2/Gating的预测Top1/3/5的并集中由真实标签选择，能相对A3降低多少KL：

{md(union[['split','keep','shortlist_per_model','mean','ci_low','ci_high']])}

这不意味着增加两个Router就能自动取得该收益。选择哪个候选仍需未知的有效决策规则；不能将Oracle标签用于线上选择或冒充已训练ensemble。

## 全候选覆盖与尚未识别的问题

{md(coverage)}

对未在48池中的全搜索赢家，真实KL保留NaN，绝不代填。训练/验证pool内regret可以测量，但未标注候选的全搜索损失、全局Oracle regret及“候选偏移造成多少损失”仍不可识别。当前有标签池已经偏向旧策略的好候选，R2覆盖较高可能与旧锚点的相似性有关，但不是因果结论。全评分差、池外赢家清单和真实已知标签均保留。

## 统计与交付

独立输入Bootstrap2,000次，seed42；同一输入的三个seed先平均。不同Head/Mask/Pair不作为独立输入。保留每seed均值、seed均值SD、输入SD和全部配对区间；区间未作多重比较校正且条件于三个固定训练种子。少量无定义相关性记录NaN并报告有效输入数。

代码：src/selection_diagnostics.py（指标）；collect_selection_diagnostics.py（冻结评分与候选导出）；analyze_selection_diagnostics.py（Bootstrap/图表/待标清单）；verify_selection_diagnostics.py（源权重、选择和上界核验）；report_selection_diagnostics.py（报告与尾部分解）；tests/test_selection_diagnostics.py（完美/反序/局部错排/并列/Mask编码测试）。

audit_selection_teacher_provenance.py用于复核原Teacher shards与导出数据，保留全部shard哈希。完整本地测试套件63项通过；历史Checkpoint验证regret重建最大误差约1.1e-9，池评分与全候选对应评分最大差约1.2e-7。

图1：同池regret与95%CI。图2：预测短名单的Oracle可移除regret比例（描述性均值曲线，完整CI另存shortlist_bounds）。CSV/JSON/LaTeX完整表在analysis；逐输入/seed结果、预测分数和候选身份在collected。没有新增.pt权重。

```bash
# 独立目录内放scripts/src/tests，链接source到interaction归档，baseline到gating归档
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=4
python scripts/collect_selection_diagnostics.py --root . --output collected
python scripts/analyze_selection_diagnostics.py --root .
python scripts/verify_selection_diagnostics.py --root .
python scripts/report_selection_diagnostics.py --root .
```

## Objective summary for rebuttal planning

All existing routers already use a listwise joint-ranking objective. In a fixed diagnostic using existing labelled candidate pools, substantial selection regret remains despite relatively high global rank correlation. Oracle reranking of short predicted lists exposes improvement headroom, but most K=9 regret arises from selections outside the true top five, and better best-candidate hit rates do not necessarily yield lower mean KL. The historical pools already contain policy anchors and local hard candidates. These findings motivate a bounded audit of currently unlabelled disagreement candidates; they do not establish that a new ranking objective will improve deployment, nor justify a new training claim. No additional model was trained.
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8');print(decomposed.to_string(index=False));print(tails.query('keep==9').to_string(index=False))


if __name__=='__main__':main()
