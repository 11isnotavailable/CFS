"""Evidence-grounded report; do not turn measurement cost into a value judgment."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;out=root/'analysis'
    s=pd.read_csv(out/'summary.csv');o=pd.read_csv(out/'oracle.csv');paired=pd.read_csv(out/'paired.csv');d2=pd.read_csv(out/'pair_information_paired.csv')
    plan=json.loads((root/'training_plan.json').read_text());cost=plan['cost'];ratio=cost['joint']/cost['pair']
    def metric(method,metric,curve='labels',budget=20000):return s.query('pool==42 and method==@method and metric==@metric and curve==@curve and budget==@budget').iloc[0]
    def oracle(metric):return o.query('pool==42 and threshold==1e-5 and metric==@metric').iloc[0]
    def interval(row):return f'{row["mean"]:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]'
    def md(t):return t.to_markdown(index=False,floatfmt='.6f')
    r1=oracle('pair_oracle');r2=oracle('joint_oracle');gain=oracle('oracle_gain')
    main=s.query('pool==42 and method in ["pair","joint"] and metric in ["oracle","selected","nmse","spearman","normalized_regret"]')
    wide=main.pivot(index=['method','curve','budget','labels'],columns='metric',values='mean').reset_index()
    wide.to_csv(out/'main_table.csv',index=False);wide.to_latex(out/'main_table.tex',index=False,float_format='%.5f')
    info=s.query('pool==42 and budget==20000 and curve in ["information","pair_transfer"] and metric in ["nmse","spearman","selected","normalized_regret"]')
    itable=info.pivot(index='method',columns='metric',values='mean').reset_index();itable.to_latex(out/'pair_information_table.tex',index=False,float_format='%.5f')
    comparison=d2.query('pool==42 and budget==20000 and metric=="selected" and contrast=="direct_joint_minus_predicted_pair_mean"').iloc[0]
    benchmark=pd.read_csv(root/'benchmark/costs.csv');warm=benchmark.query('sample>0').groupby('method')[['seconds','seconds_per_label','variants','forward_calls','backward_calls','mean_recovery_excess','max_recovery_excess']].median().reset_index()
    events=pd.read_csv(out/'teacher_events.csv');event_table=events.groupby('kind')[['wall_seconds','gpu_event_seconds']].median().reset_index()
    full=pd.read_csv(out/'full_relation_space.csv');full_warm=full.query('sample>0').groupby('kind')[['relations','seconds','gpu_event_seconds']].median().reset_index()
    continuous=pd.read_csv(out/'continuous_sensitivity.csv');counts=pd.read_csv(out/'teacher_stage_wall.csv');total=pd.read_csv(out/'teacher_per_input_cost.csv')
    cpu=pd.read_csv(out/'predictor_cost.csv');valid=pd.read_csv(out/'validity.csv');train=pd.read_csv(out/'training.csv')
    train_table=train.groupby(['kind','curve','nominal_budget','labels'])[['unique_training_inputs','parameters','steps']].first().reset_index()
    reply=f'''Pairwise CFS and coalitional reconstruction measure different levels of redundancy. We evaluated the natural Joint-2 extension on the same frozen ViT-B block, using an identical label-independent four-head candidate pool per source, output KL, valid-deletion rule and alpha domain. Exact 31-point/31-by-31 grid searches gave Pairwise recovery {r1['mean']:.4f} and Joint-2 recovery {r2['mean']:.4f}; the paired gain was {interval(gain)} (95% input-bootstrap CI). Joint-2 therefore captures additional redundancy that the pairwise definition intentionally does not exhaust.

The relation space grows from 132 to 660 relations for 12 heads (1,560 to 29,640 for 40 heads). On our cached ViT suffix, Joint-2 labels cost approximately {ratio:.1f} times more per label in steady-state batched measurement; these are measured implementation costs, not a universal complexity constant. Both tasks used 668,801-parameter predictors and three fixed training seeds. At 20,000 labels, Joint-2 had lower NMSE ({metric('joint','nmse')['mean']:.3f} vs {metric('pair','nmse')['mean']:.3f}), with no clear normalized-regret difference. Thus we do not claim that coalitional relations are inherently harder to predict. Under estimated equal measurement cost and independently matched intervention-count budgets, Joint-2 prediction quality deteriorated, although its selected recovery still exceeded Pairwise recovery.

A simple mean of predicted Pairwise scores selected Joint-2 coalitions with recovery {metric('predicted_pair_mean','selected','pair_transfer')['mean']:.4f}, versus {metric('joint','selected')['mean']:.4f} for the direct Joint-2 predictor; their paired difference was {interval(comparison)}. This suggests a useful compact basis for coalition selection, not equivalence or absence of higher-order information. All selected-recovery comparisons remain conditioned on input-specific oracle coefficients; they do not establish deployable forward-only compensation or acceleration. These exploratory, non-multiplicity-adjusted results support Pairwise CFS as a cheaper, interpretable primitive, while recognizing Joint-2 as a valuable extension rather than a scientifically inferior alternative.'''
    (out/'REBUTTAL_nLkz.md').write_text(reply,encoding='utf-8')
    report=f'''# Pairwise CFS vs Joint-2：可预测性与成本

日期2026-11-16。已完成51个正式拟合、2个短程Pilot；Backbone未训练，原CFS/Routing/Scaling文件未改。主候选池seed42，敏感性123/2027。完整数据与模型保存，未筛选有利结果。

## 科学结论（Q1–Q5）

**Q1：Joint-2有额外Oracle恢复能力。** Pairwise={interval(r1)}，Joint-2={interval(r2)}，配对增益={interval(gain)}。有效Source中增益>1e-6的平均比例为{oracle('source_gain_fraction')['mean']:.2%}；输入均值增益>1e-6的样本比例为{oracle('input_gain_fraction')['mean']:.2%}。增益按输入分布见Figure1；它不是“Pairwise已穷尽冗余”的证据。

**Q2：测量成本更高，但本轮绝对开销仍可控。** 稳态批量单标签为Pairwise {cost['pair']*1e6:.2f}微秒、Joint-2 {cost['joint']*1e6:.2f}微秒，约{ratio:.2f}倍。主候选池48/72个关系，实际每输入测量约{cost['pair']*48*1000:.3f}/{cost['joint']*72*1000:.3f}毫秒。不能将组合数、单标签成本、完整空间成本混为一谈，也不称Joint-2不可计算。

**Q3：相同标签数并不支持Joint-2更难预测。** 20k标签下NMSE Pairwise={interval(metric('pair','nmse'))}，Joint-2={interval(metric('joint','nmse'))}；Joint-2更低。Spearman差与归一化Regret差的区间见配对表，不能用4/6候选的原始命中率直接比较难度。匹配估算GPU路径时间或严格网格干预点数后，Pairwise预测质量更好。这是有限架构/训练协议下的测量预算效率结果，不是联合关系不可学习的定理。

**Q4：额外Oracle能力确实转化为选择收益。** 等20k标签下选中Recovery为{metric('pair','selected')['mean']:.4f}/{metric('joint','selected')['mean']:.4f}；等估算成本时Joint-2为{metric('joint','selected','cost')['mean']:.4f}，严格干预预算时为{metric('joint','selected','evaluations')['mean']:.4f}，仍高于Pairwise。不能把NMSE较差写成实际Recovery也较差。所有选中Recovery均使用该输入该候选的Oracle Alpha，没有Alpha Predictor，**不是完整forward-only部署收益**。

等标签的“选中增益/Oracle增益”可能大于1：这是两个方法各自选择Regret不同导致的比值，不是有界的额外能力利用百分比。原始比值和CI保留在paired.csv，不裁剪或解读成超过100%的物理恢复。

**Q5：支持更紧凑、便宜的基础关系，不支持Pairwise全面占优。** 真Pairwise分数的小模型可以很好近似Joint-2标签，但使用测试时Oracle信息，不能视作免费部署。更实用的预测Pairwise分数均值，在相同20k标签下获得Joint-2选择Recovery {metric('predicted_pair_mean','selected','pair_transfer')['mean']:.4f}，直接Joint-2为{metric('joint','selected')['mean']:.4f}，直接减聚合的配对差{interval(comparison)}；区间跨零不证明等效。二者最终仍依赖Joint-2 Oracle Alpha。成本高不等于科学价值低。

## 主结果与学习曲线

表中cost是相同**估算的摊销后缀GPU路径时间**；evaluations严格匹配标签网格干预次数（仅整数取整有<1%差异），不是相同墙钟时间。pair行在两类预算对照中共用。

{md(wide)}

![Oracle与测量成本](figure1_oracle_cost.png)
![三种预算的预测曲线](figure2_predictability.png)
![Oracle到预测器差距](figure3_oracle_predictor_gap.png)

图均提供PDF。完整Mean/CI/输入SD/seed SD见summary.csv；逐seed见per_seed.csv；所有逐输入记录见per_input.csv/json。命中率/Top2 Recall之外同时提供random基线和chance-adjusted指标。

### 主池20k预算的配对比较

Joint2减Pairwise；NMSE与Regret越低越好，Recovery/Spearman越高越好。

{md(paired.query('pool==42 and budget==20000'))}

### Pairwise信息能解释多少联合关系？

oracle_pair_max/mean与pair_info使用**真实Pairwise Oracle分数**，是有额外测量成本的信息诊断。pair_info为1185参数对称MLP，同20k Joint标签、3seed；输入为两条Pairwise分数的均值和绝对差。predicted_pair_max/mean仅使用Pairwise预测器的输出，未用Joint标签拟合选择规则。max、mean事先同时报告，没有测试后挑选。NMSE差异与排序差异分别看：未校准的mean可能MSE很差而排序仍好。

{md(itable)}

{md(d2.query('pool==42 and budget==20000'))}

## 定义、原实现与正确性

复用src/head_intervention.py、model_loading.py、runtime_cfs_router.prehead_summary及原dense_kl。z'=z-c_i+(alpha_j-1)c_j+(alpha_k-1)c_k；alpha=1不改变替代Head，alpha=0删除该Head。Bias只计一次，不改其他贡献、LayerNorm或后续MLP。只删除一个Source。CFS=1-Dreplacement/max(Ddrop,1e-12)，D为温度1的KL(dense||intervened)，clamp_min0保持原实现；Ddrop>1e-5有效。先在有效Source内均值，再按独立输入统计。真实标签未额外裁剪。

原ViT-B AugReg `vit_base_patch16_224.augreg_in21k_ft_in1k`、原checkpoint与timm预处理；层11、12头。最后层后CLS不再与Patch交换信息，因此可缓存CLS Attention输入/融合Head输出，只重算投影、残差、MLP、Norm和分类Head。直接Forward与缓存后缀的无干预、单头删除、双缩放均数值一致；全部Joint候选都包含其两个Pairwise可行解。72项本地测试及4项生产环境新测试通过；verification.json记录全部审计。

本轮统一FP32、关闭TF32。旧Pairwise训练缓存是16点/FP16，旧Joint-Subset缓存是多头硬删除，均不能当作本轮统一31点缩放标签；复用了权重、划分、预处理及代码，必要部分重新测量。两方法没有精度差异。

### Alpha优化Benchmark

8训练输入比较完整31x31网格、从(1,1)梯度初始化、最佳Pairwise初始化、3起点60/120步。约束[0,3]，投影Adam lr0.08，逐步保留最好值。网格每标签31/961个候选，最大batch1024，前后向调用数见表。梯度方法连续可行域比分立网格更密，负excess表示其KL更小，不代表网格求解失败。网格更快，且对**原31点离散域**精确，所以选用网格而非近似优化；原有Pairwise可行解也保留，避免微小batch浮点噪声破坏包含性质。

以下排除首次setup调用，报告中位数；全部冷/热记录保留在benchmark/costs.csv。max_recovery_excess列在此为每输入最大值的中位数，原始逐输入最大误差可直接核对，不冒称全局最大值。

{md(warm)}

预定32测试输入上，两种方法都进一步连续优化；平均结果如下，仅作精度敏感性，不重新训练或选择主方案：

{md(continuous.drop(columns='sample').mean().rename('mean').reset_index())}

## 实测成本、完整关系空间与预测开销

GPU Event与同步墙钟计时（同一GPU、warm-up后8训练输入各3重复）：

{md(event_table)}

H=12完整132/660关系空间也实测，以下为输入1–3的稳态中位数；input0冷启动另保留：

{md(full_warm)}

H=40只报告组合数1560/29640，以及固定ViT单位成本的情景投影，**没有测量40头模型**，也不假设其宽度/吞吐相同。见relation_complexity.csv。完整标签采集各阶段墙钟时间（含缓存保存）如下；它不同于单标签摊销模型：

{md(counts)}

累计测量区间秒数：{total[['capture_seconds','drop_seconds','pair_seconds','joint_seconds']].sum().to_dict()}。这些阶段只运行一次，学习曲线离线抽取标签；不能将子集预算估算假称实际重复收集总时间。公平成本控制不包含共同的Dense捕获/Source-drop固定缓存，均单列。

预测成本，单关系与完整池分开。参数量完全相同，权重文件包括少量归一化元数据。FLOPs是Linear乘加按2 FLOPs计算，含一次共享global encoder，不含非线性、embedding查表、归一化和Backbone；不是完整模型FLOPs。实际延迟中两个完整池接近，不能仅按48/72关系数宣称1.5倍实测延迟。

{md(cpu)}

## 数据、训练预算与统计限制

1024训练图来自原4096训练划分，验证64沿用原划分，新测试256排除原4608与后续确认256，按路径和内容SHA检查。排除的是本路由实验系列，不声称公共图像在全部历史研究从未出现。候选池由seed42均匀无放回采样，每Source四个，固定且与标签无关。

PairScorePredictor level0仅提供预Attention2304维特征。为Joint无序集合加入candidate embedding均值/绝对差，两个模型完全相同668801参数；singleton用重复candidate。原PairScore只优化排序，本研究为校准CFS回归采用**相同标准化MSE**，不能称原训练损失完全没变。目标均值/方差来自该预算训练标签；原始输出保留，预测越界比例明确报告。两者共享全部训练特征的无标签归一化统计。AdamW3e-4、decay1e-4、batch128、2000步；每250步验证MSE选模型，三个seed17/23/41，无超参搜索。每种预算不同的独立训练输入数也保留：

{md(train_table)}

39个主模型在生成测试Teacher前锁定；额外12个严格干预次数对照在主测试程序启动之后、**人工查看测试指标之前**声明，仍仅用训练/验证选择，并单独锁定。COALITIONAL_COST_CONTROL_AMENDMENT.md完整记录；不把所有51个模型都说成测试标签产生前锁定。

2000次Bootstrap按独立输入，共享配对索引，先在输入内平均3seed。NMSE是aggregate MSE/训练常数在测试上的MSE，非每输入比值再均值。Normalized Regret用对应随机选择Regret归一化；最佳候选命中处理真实最优的数值并列；Top2用稳定排序。候选分数完全平坦时Spearman记0并报告比例。多预算/指标/候选池为探索性分析，未多重校正。seed CI条件于这三个训练seed，不把seed作为额外输入。

候选池123/2027仅对相同固定预测器评估，候选身份可能包含训练池之外的关系；这是候选池敏感性/组合外推，不能把它说成每池都独立训练后的学习效率。

{md(o.query('threshold==1e-5'))}

各池20k等标签预测结果（固定同一批模型）：

{md(s.query('budget==20000 and curve=="labels" and method in ["pair","joint"] and metric in ["selected","nmse","spearman","normalized_regret"]')[['pool','method','metric','mean','ci_low','ci_high']])}

删除阈值敏感性（主定义仍1e-5）：

{md(valid)}

全部1e-4/1e-3阈值的Oracle结果也在oracle.csv；不把弱删除样本静默移除。没有有效Source的输入只在对应阈值统计中排除，n_inputs及逐输入记录均保存。

## 复现与交付

独立目录`/data/cfs_coalitional_20261116`，保留源代码、images/data、教师、全部训练权重/日志。原Backbone路径由benchmark_coalitional_cfs.load读取，可按环境映射到同SHA文件。执行环境见environment.json。不要覆盖历史输出。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4
# 准备RUN/data, images和manifest；prepare_coalitional_cfs.py提供完整CLI
python scripts/benchmark_coalitional_cfs.py --root RUN
python scripts/collect_coalitional_cfs.py --root RUN --split train
python scripts/collect_coalitional_cfs.py --root RUN --split tune
python scripts/train_coalitional_cfs.py --root RUN --pilot
# 两个worker可分别绑定GPU0/1，先完成主模型
python scripts/train_coalitional_cfs.py --root RUN --worker 0 --workers 2
python scripts/train_coalitional_cfs.py --root RUN --worker 1 --workers 2
python scripts/lock_coalitional_cfs.py --root RUN
# 补充严格预算对照，两worker均增加--evaluation-budget，再独立lock
python scripts/train_coalitional_cfs.py --root RUN --worker 0 --workers 2 --evaluation-budget
python scripts/train_coalitional_cfs.py --root RUN --worker 1 --workers 2 --evaluation-budget
python scripts/lock_coalitional_cfs.py --root RUN --evaluation-budget
for pool in 42 123 2027; do
  python scripts/collect_coalitional_cfs.py --root RUN --split test --pool-seed $pool
  python scripts/evaluate_coalitional_cfs.py --root RUN --pool-seed $pool
done
python scripts/evaluate_coalitional_cfs.py --root RUN --evaluation-budget
python scripts/profile_coalitional_cfs.py --root RUN
python scripts/benchmark_coalitional_fullspace.py --root RUN
python scripts/verify_coalitional_cfs.py --root RUN
python scripts/analyze_coalitional_cfs.py --root RUN --output RUN/analysis
python scripts/report_coalitional_cfs.py --root RUN
```

新增文件均以coalitional命名：src下干预/预测器与输入级指标；scripts下prepare/benchmark/collect/train/lock/evaluate/profile/verify/analyze/report；tests下数值与统计测试。没有改原模型、路由或Scaling脚本。原始teacher与完整权重见ARTIFACTS.md，不只交付最终均值。

## Reviewer nLkz draft

{reply}
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8');print(reply)


if __name__=='__main__':main()
