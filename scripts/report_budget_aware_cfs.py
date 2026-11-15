"""Report validation failures and all fixed test diagnostics without re-selection."""
import argparse,json
from pathlib import Path
import pandas as pd


def md(frame):
    frame=frame.copy()
    for col in frame:
        if pd.api.types.is_float_dtype(frame[col]):frame[col]=frame[col].map(lambda v:f'{v:.7f}')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    lock=json.loads((root/'selection_lock.json').read_text());verified=json.loads((root/'verification.json').read_text())
    summary=pd.read_csv(root/'analysis/summary.csv');paired=pd.read_csv(root/'analysis/paired.csv');targets=pd.read_csv(root/'analysis/targets.csv')
    candidates=pd.read_csv(root/'validation_candidates.csv');probe=pd.read_csv(root/'analysis/probe_losses.csv');grad=pd.read_csv(root/'analysis/gradient_diagnostics.csv')
    selected=pd.DataFrame(lock['representatives'].values())
    primary='A3' if lock['fallback_to_A3'] else f'W{lock["primary"]["weight9"]:g}'
    result='没有找到满足验证集约束的更新；按预先规则保留原A3。' if lock['fallback_to_A3'] else f'按验证集预选{primary}，没有使用测试集改选。'
    hard=summary.query('metric=="kl" and compensation=="hard"')
    rows=[]
    for method in ('Gating','R2','A0','A3','W1','W1.5','W2','W3'):
        row={'Method':method}
        for keep in (3,6,9):
            r=hard.query('method==@method and keep==@keep').iloc[0];row[f'K={keep}']=f'{r["mean"]:.7f} [{r.ci_low:.7f}, {r.ci_high:.7f}]'
        rows.append(row)
    initial=probe.query('epoch==0 and weight9==1').groupby('loss')[['value','gradient_norm','weighted_gradient_norm']].mean().reset_index()
    initial_cos=grad.query('epoch==0 and weight9==1').groupby(['left','right']).cosine.agg(['mean','min','max']).reset_index()
    diagnostic=paired.query('metric=="kl" and compensation=="hard" and reference in ["A3","R2","Gating"]')
    meets=targets[targets.method.str.startswith('W')&targets.all_targets]
    target_text='四个锁定的诊断代表均未同时达到三个测试数值目标。' if meets.empty else '有诊断代表达到数值目标，但只有验证集合格的主配置才能被选用；测试结果不改变选择锁。'
    text=f'''# Budget-aware A3 短程微调

日期：2026-11-15。起点commit60bf8a7。**{result}** {target_text}

本轮仅执行方案一，没有启动Score融合、结构搜索、新Teacher或Backbone训练。原权重和结果不变。由于既有测试集已被用于提出本轮研究目标，这是一轮明确标注的后续探索；本轮内部仍严格只用验证集选择，不把它描述为从未观察过测试集的一次性确认实验。

## 1. 固定设置与预算

同一个825091参数的R2结构、2304维原始prehead输入、CFS+Alpha定义与λ=0.01。每个seed17/23/41从自身原A3的验证集选定Checkpoint开始。四组相对权重为(1,1,1)、(1,1,1.5)、(1,1,2)、(1,1,3)。各10epochs、batch64、lr3e-5、AdamW decay1e-4，无学习率或时长搜索。总12runs、7680updates。原checkpoint不含optimizer状态，因此所有run均重新初始化AdamW moments；这是短程fine-tuning，不是精确恢复原优化轨迹。

目标为 `0.5*((L3+L6+w9*L9)/(2+w9)+0.01*(L_CFS+0.1*L_alpha))`。Joint权重总和保持1，避免混淆总loss尺度变化；w9=1是等权继续训练对照。辅助定义、真实KL、mask池与原归一化均不变。模型架构与参数量未增加。

先测原A3实际起点的损失/梯度，再执行固定计划。下表为起点三个seed平均（共同global encoder+head embeddings），不是跨独立输入的置信区间：

{md(initial)}

起点梯度cosine（三个seed的均值与范围）：

{md(initial_cos)}

K9梯度并不更小；seed17的预算梯度范数约0.0824/0.1483/0.2037。各预算loss约3.536/3.698/3.618也不能简单用来比较有效优化贡献。重加权改变方向/取舍，不能被描述为已证实的梯度饥饿修复。epochs0/1/5/10的raw loss、梯度范数、加权贡献和cosine全部保存。

## 2. 只按验证集选择

验证数据为原64 tune输入和48候选mask/K；训练仍为原4096输入。按同一epoch平均三个seed的每预算regret，要求K3和K6不高于A3起点（容差1e-8），且K9更低。在合格项中按K9、平均regret、较小权重、较早epoch排序，选择一个所有seed共用的权重/epoch。若无合格项，保留A3。

原A3 tune平均regret：K3={lock['baseline_tune']['val_K3']:.8f}，K6={lock['baseline_tune']['val_K6']:.8f}，K9={lock['baseline_tune']['val_K9']:.8f}。

共40个weight/epoch候选，合格数为{verified['eligible_candidates']}。下表为每权重按预定规则锁定的代表：若无合格epoch，则仅用于诊断地选择最小K9 regret的epoch，不能替代主配置。

{md(selected)}

所有原始132行seed/weight/epoch验证轨迹在 `validation_trajectories.csv`，40个聚合候选在 `validation_candidates.csv`，选择结果在 `selection_lock.json`。各诊断代表可能对应不同微调长度，因此它们的测试差异不是纯粹的权重因果效应；相同epoch的验证轨迹可直接比较。没有为某个seed选择特有权重或使用测试标签选epoch。

## 3. 完整测试诊断

在selection lock生成后，统一评估四个固定代表，448输入、同一dense reference、原KL方向和预处理。原A3、Joint-only(A0)、原R2和Gating都保留。Mean [95% CI]：

{md(pd.DataFrame(rows))}

测试数值目标为K3≤0.10517、K6≤0.02480、K9<0.00700。下面同时显示原未舍入值和逐项是否达到；最后一列primary严格来自验证锁，不随测试表变化：

{md(targets)}

注意原A3的K6实际值约{float(hard.query('method=="A3" and keep==6')['mean'].iloc[0]):.8f}，显示为0.02480；与用户表中精确阈值的微小舍入差没有实质意义。对原A3精确值的比较应看配对差，而不是只看小数位后的布尔标记。

更新−参考方法的配对KL差（负值有利于更新）：

{md(diagnostic[['method','reference','keep','mean','ci_low','ci_high']])}

区间基于2000次独立输入Bootstrap，seed42；先对同一输入的三个训练seed平均，避免把seed当额外输入。各seed均值、SD和CI保留在 `per_seed_summary.csv`；所有区间未作多重比较校正，且条件于这三个训练种子。没有将均值胜出写成统计优势，没有以区间跨零宣称等效。

## 4. 解释与限制

等权继续训练对照也未通过验证约束，因此退化不能只归因于增加K9权重。原A3本来就是从60epoch轨迹中按验证集挑出的Checkpoint，继续训练不保证验证排序继续改善；fresh AdamW、64-input验证集和候选池排名的离散性也可能影响结果。这些是可能解释，不是已识别原因。

在测试诊断中W1.5的三个KL均值都略低于A3，但K9仍约0.00767，没有达到0.00700目标。其验证代表本身不合格，不能因这个测试观察而升格为主结果。验证指标是48候选集合内的regret，实际部署在全部220/924/220个子集上选择；两者不同，可能存在代理指标偏差，但本轮没有在看过测试后重新定义验证标准。

这次固定范围的尝试不能证明“原R2的K9优势必然来自Structured Auxiliary”，也不能排除其他小规模策略；但结果不足以支持把本轮更新替换为主方法。没有根据测试结果再扩展权重、学习率、训练长度或转向融合搜索。

共享原R2补偿的副结果与Accuracy完整保留在 `summary.csv/json/tex`、`paired.csv/json/tex`，但均不用于选择，也不用于替代未达标的shared-hard主结果。共享补偿器仍来自原R2，附加参数/计算边界与此前报告一致。没有实际backbone速度提升声明。

## 5. 可用于Rebuttal的客观说明

We conducted a bounded budget-aware continuation experiment on the unchanged 825,091-parameter A3 router: four prespecified K=9 weights (1, 1.5, 2, 3), three fixed seeds, and ten low-learning-rate epochs per run. Joint weights were normalized to preserve the overall loss scale. Checkpoint selection used only the existing validation set, requiring no degradation at K=3/6 and an improvement at K=9. {('No candidate satisfied these constraints, so we retained the original A3 rather than selecting a favorable test result.' if lock['fallback_to_A3'] else 'The eligible configuration was locked before held-out evaluation.')} All fixed representatives, including the equal-weight continuation control, are reported. This finite negative or mixed result does not establish that budget imbalance is the cause of the original R2/A3 trade-off, nor that auxiliary supervision uniquely explains the K=9 gap.

## 6. 交付、检查与复现

新增文件：`src/budget_aware_objective.py`定义归一化重加权和受约束验证选择；`train_budget_aware_cfs.py`微调与诊断；`run_budget_aware_cfs.py`固定12runs及选择锁；`evaluate_budget_aware_cfs.py`复用原评估；`analyze_budget_aware_cfs.py`配对统计/图表；`verify_budget_aware_cfs.py`核验源权重、固定范围与选择；`report_budget_aware_cfs.py`生成此报告。测试在 `tests/test_budget_aware_objective.py`，验证w9=1与A3目标相同、梯度比例正确、验证约束和fallback生效。

完整输出含12个锁定代表的权重、所有120个训练epoch的日志/验证记录、梯度诊断、777?条逐输入记录（精确数见verification）、Bootstrap抽样索引、CSV/JSON/LaTeX、PDF/PNG。所有epoch权重另保留在远端runs目录；本地紧凑归档只包含预定代表，不删除远端其他Checkpoint。依赖的A3、Teacher、Backbone仍使用此前归档，不重复复制。

Linux实验目录的 `source/` 指向上一轮cfs_interaction目录（其内部baseline需有效），`baseline/` 指向learned_gating目录。新输出路径不得已存在：

```bash
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4
python scripts/train_budget_aware_cfs.py --source source --output initial_audit --weight9 1 --seed 17 --epochs 0
python scripts/run_budget_aware_cfs.py --root .
python scripts/evaluate_budget_aware_cfs.py --root .
python scripts/analyze_budget_aware_cfs.py --root . --output analysis
python scripts/verify_budget_aware_cfs.py --root .
python scripts/report_budget_aware_cfs.py --root .
```

主图：`analysis/figure1_budget_finetuning.pdf/png`（完整对照和配对差），`analysis/figure2_validation_tradeoffs.pdf/png`（全部权重的验证轨迹）。方案二融合本轮未执行。
'''
    text=text.replace('777?条逐输入记录（精确数见verification）',f'{verified["routing_rows"]}条逐输入记录')
    (root/'REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
