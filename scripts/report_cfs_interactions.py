"""Render numerical evidence and bounded reviewer responses from completed runs."""
import argparse,json
from pathlib import Path
import pandas as pd


def md(frame):
    columns=list(frame.columns)
    return '\n'.join(['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])


def formatted(frame,columns=None):
    frame=frame.copy() if columns is None else frame[columns].copy()
    for col in frame:
        if pd.api.types.is_float_dtype(frame[col]):frame[col]=frame[col].map(lambda x:f'{x:.6f}')
    return md(frame)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    lock=json.loads((root/'selection_lock.json').read_text());verification=json.loads((root/'verification.json').read_text())
    summary=pd.read_csv(root/'routing_analysis/routing_summary.csv');paired=pd.read_csv(root/'routing_analysis/paired_routing.csv')
    pilot=pd.read_csv(root/'pilot_validation.csv');gradient=pd.read_csv(root/'figures/gradient_summary.csv')
    additive=pd.read_csv(root/'additive_final/additive_summary.csv');interaction=pd.read_csv(root/'interaction_analysis/interaction_summary.csv')
    correlations=pd.read_csv(root/'interaction_analysis/correlation_summary.csv');increment=pd.read_csv(root/'interaction_analysis/incremental_prediction.csv')
    prediction=pd.read_csv(root/'interaction_analysis/heldout_prediction_summary.csv');conditional=pd.read_csv(root/'routing_analysis/conditional_routing.csv')
    assoc=pd.read_csv(root/'routing_analysis/conditional_correlations.csv');decomp=pd.read_csv(root/'routing_analysis/pipeline_decomposition.csv')
    direct=pd.read_csv(root/'descriptive_checks/direct_auxiliary_ablation.csv');noise=pd.read_csv(root/'descriptive_checks/numerical_noise_diagnostic.csv')
    checks=json.loads((root/'interactions/checks.json').read_text())
    def routing_table(comp):
        rows=[]
        for method in ('Taylor','Gating','A0','R2','A1','A2','A3'):
            row={'Method':method}
            for keep in (3,6,9):
                r=summary.query('method==@method and keep==@keep and compensation==@comp and metric=="kl"').iloc[0]
                row[f'K={keep}']=f'{r["mean"]:.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}]'
            rows.append(row)
        return md(pd.DataFrame(rows))
    a3=paired.query('method=="A3" and metric=="kl" and compensation=="hard"')
    g=gradient[gradient.variant.isin(['A0','A3','A4'])&((gradient.left=='joint')|(gradient.right=='joint'))]
    g=g[~g.left.str.startswith('joint_K')&~g.right.str.startswith('joint_K')]
    c=conditional.query('method=="A3" and metric=="gating_minus_router"').copy()
    absolute=conditional[conditional.metric=='kl'].pivot(index=['keep','group'],columns='method',values='mean')
    c['gating_kl']=[absolute.loc[(r.keep,r.group),'Gating'] for r in c.itertuples()]
    c['A3_kl']=[absolute.loc[(r.keep,r.group),'A3'] for r in c.itertuples()]
    primary_assoc=assoc.query('outcome in ["gating_kl","gating_minus_A0","gating_minus_A3"] and feature=="complexity"')
    cfscorr=correlations.query('split=="test" and feature in ["cfs_mean","cfs_asymmetry","cosine","taylor_mean"]')
    inc9=increment[(increment.keep==9)].set_index('target')
    text=f'''# CFS Routing 监督优化与非加性功能交互分析

日期：2026-11-15。原始基线：commit `b494b13`。原始数据、Checkpoint 和论文正文未修改；未训练 Backbone。

**核心结论：得到预算相关的正面证据，但没有证明完整的因果机制链。** 验证集预选 A3（Joint+CFS+Alpha，λ=0.01）在 K=9 相比 Joint-only 降低 KL 约4.48%，配对95% CI不含零；K=3、6未确认增量改善。纯 Joint+CFS 的 A1 没有在任何 K 明确改善 Joint-only。CFS在控制重要性、相似度与上下文后，对K=9二阶交互有独立输入上的增量预测价值；交互复杂度与A3的相对Routing优势关联仍未分辨，不能将预测价值写成已验证的因果解释。

## 1. 审计与冻结方案

附件四行均值由已有CSV精确复现。此次全部基线重新执行，最大逐输入KL差为 {verification['baseline_max_abs_kl_difference']:.3g}（CSV浮点往返量级）。所有新消融与基线使用同一Dense logits，448输入、种子17/23/41、K=3/6/9。Imagenette2-160 train/tune/test为4096/64/448；ViT-B AugReg frozen，最后attention block11、12heads；原始checkpoint SHA256为 `683c6d82c1e832981a5fdce9ae66be4f02542b4bd7b7fc822b08f53c005de8df`。

R2是2304维prehead特征的pair scorer，825091参数；Gating是822460参数的预算无关加性head score，因此其Top-K集合跨K嵌套。输入使用原FP16缓存边界和训练集归一化。R2允许非嵌套子集，并枚举220/924/220个候选；不能把架构/搜索优势自动归因于CFS标签。沿用bicubic、224、crop_pct0.9、mean/std0.5。

原R2包含representative listwise CE、0.5 score SmoothL1、0.5 deletion CE、0.1 alpha loss、1.0 structured-subset loss和0.5 joint-set loss。新A1–A3刻意不含旧structured-subset辅助目标，因其同时依赖CFS与deletion，混入会破坏消融含义。所有pairwise原损失与原函数逐项一致，单元测试验证。源CFS训练值仍按原方法clip[-1,1]，alpha clip[0,3]，对角线规则不变；交互解释使用原始未裁剪CFS及both-source drop>1e-5过滤。

新目标为 `0.5 × [L_joint + λ(L_CFS + d×0.5L_del + a×0.1L_alpha)]`，L_CFS=原listwise+0.5原score regression；A1:(d,a)=(0,0)，A2:(1,0)，A3:(0,1)。A1/A2冻结未参与目标的alpha输出，824962优化参数；A3优化825091参数。A0、A4原权重复用。

每个新run仍为AdamW lr3e-4、decay1e-4、batch64、60epochs（3840updates），同一4096输入、每输入每K48条joint KL。15个seed17 pilots覆盖A1/A2/A3×λ{{0.01,0.03,0.1,0.3,1}}；λ0由已有A0提供。各变体只凭64个tune输入平均regret选λ，均选0.01，然后补种子23/41，共21个新run。headline在新测试前锁定A3；没有根据测试上A2在K3的更低均值改选模型。单run预算匹配，**调参预算和总监督成本并不与旧单配置基线相同**。未扩展λ网格，也未追加梯度手术或warmup搜索。

完整pilot验证结果（未选配置不做测试集排名）：

{formatted(pilot,['variant','strength','epoch','val_joint_regret','val_K3','val_K6','val_K9'])}

额外CFS标签来自既有缓存：train网格16点，tune/test实际31点，均为[0,3]；新优化仅用train标签，checkpoint选择仅用tune joint KL。每训练输入132个有向CFS标签，A2另用12个deletion值，A3另用132个oracle alpha标签。原始采集名义上需132×16=2112次pair intervention加12次deletion/input，4096输入约8,699,904次逻辑变体（不含dense/梯度/实现开销）；已有joint teacher为589824条mask KL。Alpha由同一搜索导出，无另一次搜索。这些不是实测GPU时间，缓存复用也不能称额外监督零成本。

## 2. Table A：完整Routing结果

Mean [95% CI]，先同输入平均三种子，再以独立输入做2000次Bootstrap，seed42。KL越低越好。所有逐项区间均未作多重比较校正，条件于这三个训练种子；种子均值SD、逐种子CI另存。

共享硬门控（主对照）：

{routing_table('hard')}

A3−参考方法的配对KL差，负值有利于A3：

{formatted(a3,['reference','keep','mean','ci_low','ci_high'])}

**Q1：辅助监督能否改善Joint-only？** 证据仅支持A3组合在K9的小幅改善：−0.000373 [−0.000730,−0.000072]。K3为−0.000455 [−0.001380,0.000504]；K6为+0.000667 [−0.000068,0.001528]，点估计反而更差。纯CFS的A1在K9为−0.000367 [−0.000786,0.000036]，仍跨零；A2各预算也跨零。因此不能宣称CFS单独监督已被证明带来稳定增益，也不能把A3区间更窄解释为alpha已被证明有效。

辅助目标直接对照（新增描述性比较，不用于选择）：

{formatted(direct.query('compensation=="hard"'),['contrast','keep','mean','ci_low','ci_high'])}

## 3. 梯度与训练目标诊断

在固定前64个训练输入上，eval模式、epoch0/1/10/30/60，对各目标在共同global encoder和head embeddings上的梯度记录范数与cosine。所有原始loss曲线、逐预算joint曲线和梯度在 `figures/selected_training_history.csv`、`all_gradient_diagnostics.csv` 保存。原A0/A4只能做保留checkpoint的端点诊断，不能伪称已记录历史训练轨迹。

下表是选定训练轨迹或端点的描述性统计，snapshots不是独立样本，不用它们构造伪置信区间。Figure 2梯度阴影表示三个种子的min/max范围，不是Bootstrap CI：

{formatted(g,['variant','left','right','snapshots','mean_cosine','min_cosine','max_cosine','negative_fraction','left_norm_mean','right_norm_mean'])}

原R2三个端点的joint与旧structured auxiliary均反向，平均cosine约−0.776；与整个原auxiliary bundle约−0.744。相比之下，joint与纯pairwise CFS约+0.183。因此不能简单断言“CFS loss本身导致原R2失败”。原R2端点的总joint与K9 joint方向也相反（约−0.288），存在预算取舍。A3训练probe中CFS与joint约40%的快照为负，说明冲突随训练阶段变化。

这些结果支持“原多目标可能存在优化取舍”的诊断，**不是性能下降的因果证明**：多目标最优点本来就可能存在梯度抵消；固定probe也不代表全训练分布。弱辅助权重在验证集优于强权重，但本次没有验证某一梯度处理算法能因果消除问题。

## 4. Shared compensation与系统归因

每个种子把同一个原R2的score/alpha及原 `compose_scales` 应用于所有mask。共享应用并不消除原补偿器与R2共同训练的偏向。A3在此使用的是**原R2补偿器**，不是把自己训练的alpha作为更强专用补偿；A3 alpha监督只通过router训练影响本次主selection对照。

{routing_table('shared_r2_merge')}

原完整R2 merge、A0/A1/A2/A3的同补偿变体，与Gating-hard的差异及选择/补偿算术分解全部在 `pipeline_decomposition.csv/json/tex`。分解不表示因果独立贡献。带原R2补偿的Gating/新router包含额外原R2参数与前向成本；不是同参数独立部署。旧forward成本和Top-K/穷举差异见b494b13报告；不声称backbone wall-clock加速。Accuracy、逐种子波动和准确率差的完整结果见 `routing_summary`、`paired_routing`。

## 5. Table B：真实非加性与CFS预测信息

B1对已有4096 train与64 tune的48-mask缓存做每输入、每K四折mask预测。重复mask归同一fold；训练集中心化、pseudoinverse最小范数，固定K的权重和为零，截距由训练均值恢复。理论有效秩12，逐输入实际rank保留；少量原anchor池fold缺秩，未隐去这些输入。新uniform-mask测试样本全部达到rank12。这里是同输入的held-out mask诊断，不是可部署router：它需要该输入的36个teacher masks。

完整误差（列出RMSE、训练误差与常数预测误差；其他MAE/SD同文件）：

{formatted(additive[additive.metric.isin(['rmse','train_rmse','constant_rmse'])],['split','source','keep','metric','n','mean','ci_low','ci_high'])}

**Q2：加性建模是否有限？** 在128个测试输入的uniform48-mask上，held-out RMSE为0.10718/0.08210/0.03557，训练RMSE为0.07022/0.05458/0.02387。加性拟合明显优于常数，但并非精确拟合。不能仅凭预测误差证明非加性；B2直接测量提供了独立证据。

旧穷举oracle只保留最优mask，没有全部KL。新增测量固定64 tune+随机种子42选出的128 test，对每输入每K的66对head抽一个不含i,j、大小K−2的T，真正评估四个mask。共38016个quartet，去重后140128个mask forward，只重放最后block；另存每输入每K48个uniform mask供B1。未训练模型，也未重算pairwise CFS。

`I=D(T+i+j)-D(T+i)-D(T+j)+D(T)`。正值表示加入j后i的边际KL变化变大；若两者各自降低KL，可称KL改善的边际递减。负值相反。KL不一定单调，不能仅凭符号叫语义冗余/协同；输出KL自身的非线性就能产生交互。T在输入间也变化，不能把所有跨输入差异归因于输入本身。

{formatted(interaction.query('split=="test"'),['keep','metric','n','mean','sd','ci_low','ci_high'])}

全部66个Head Pair的统计见 `descriptive_checks/all_head_pair_statistics.csv`，保留所有pair而非最佳案例。前4输入的dense与四个直接干预检查误差均为0；全192输入dense重构最大误差 {max(r['dense_max_error'] for r in checks):.3g}。批形状导致的最大noop KL为 {noise.noop_kl.max():.8f}，mean |I|为0.09883/0.04127/0.01643。`numerical_noise_diagnostic.csv`保留接近4×noop的比例；noop只是数值诊断，不是严格的四项误差上界，未据此筛选结果。

B3使用两个source drop均>1e-5的pairs，测试保留95.82%，tune约95.00%。原始CFS值用于Spearman；回归只对CFS特征做signed log1p以控制负尾。Contribution cosine来自同一模型的post-WO贡献、全token展平、无bias；Taylor与deletion沿用原缓存。

逐输入内Spearman的平均及输入Bootstrap区间：

{formatted(cfscorr,['keep','feature','target','mean','ci_low','ci_high'])}

相关并非CFS独占：Taylor通常与交互幅度有更强的单变量相关性。增量检验更重要：固定ridge α=1，64 tune输入拟合、128 disjoint test输入预测。所有arm共享pair identity与背景mask控制；基础特征为Cosine、Taylor pair均值/差、log-deletion均值/差；+CFS添加双向均值、max、非对称度。分别预测signed I与|I|，所有标准化只用tune，绝不按test选正则化或特征。

MSE(controls)−MSE(controls+CFS)，正值表示CFS改善：

{formatted(increment,['keep','target','n','mean','ci_low','ci_high'])}

**Q3：CFS是否提供增量信息？** K9支持：signed I的MSE改善 {inc9.loc['interaction','mean']:.8f} [{inc9.loc['interaction','ci_low']:.8f},{inc9.loc['interaction','ci_high']:.8f}]，|I|改善 {inc9.loc['magnitude','mean']:.8f} [{inc9.loc['magnitude','ci_low']:.8f},{inc9.loc['magnitude','ci_high']:.8f}]。K3/6点估计为正但区间跨零。完整controls/CFS-only/+CFS误差、符号预测和恒正基线均保存。由于I多数为正，不能只凭普通sign accuracy很高宣称有效识别符号；balanced accuracy与恒正基线见 `sign_prediction_summary.csv`。

## 6. Table C：交互与Routing优势是否连起来？

复杂度预定义为每输入66对head的mean |I|，每K分别用64 tune输入中位数定high/low阈值；不使用test性能定义分组。以下保留全448、测量子集128及其两组。表中是Gating−A3，正值有利于A3。

{formatted(c,['keep','group','n','gating_kl','A3_kl','mean','ci_low','ci_high'])}

完整分组表同时含Gating/A0/原R2/A1/A2/A3的绝对KL；全组均报告。连续关系如下：

{formatted(primary_assoc,['keep','outcome','n','rho','ci_low','ci_high'])}

**Q4：能否证明这些信息转化为Routing优势？** 有限支持，不能完成强机制论证。复杂度与Gating KL的Spearman为0.442/0.625/0.822，区间都不含零，表明更强交互伴随更难的加性路由。但复杂度与A3相对优势的相关为0.122/0.057/0.145，区间均跨零；high组虽有较大的优势点估计，也未稳定分辨。A0在K3/K9、A1在K9等有部分条件关联，反而说明架构本身也可能解释收益。`high_minus_low_advantage.csv`提供直接组间差；不得用“某组显著、另一组不显著”冒充组间显著差。

因此可报告三个分别成立的观察：真实KL非加性；CFS在K9提供额外交互预测信息；预选弱CFS+alpha监督在K9改善Joint-only。不能据此声称三者存在已验证的中介因果链。B1/B2复杂度和真实CFS在这里是oracle分析量，绝非部署时免费可得的routing信号。没有根据高交互test分组训练或选择新的策略。

## 7. Reviewer WCQn response

We retained the parameter-matched learned gating baseline and conducted a validation-only auxiliary-loss ablation on the unchanged R2 architecture. All new runs use the same 4,096/64/448 split, joint-subset teachers, 60-epoch schedule and three seeds. The preselected Joint+CFS+Alpha model (λ=0.01) achieves hard-gating KL 0.10517/0.02480/0.00795 at K=3/6/9, versus 0.11154/0.02699/0.00808 for learned gating and 0.10562/0.02413/0.00833 for Joint-only R2. Relative to Joint-only, its K=9 paired difference is −0.000373 [−0.000730,−0.000072], while K=3/6 improvements are not resolved. The CFS-only auxiliary does not show a resolved gain over Joint-only. Thus we report a modest, budget-dependent benefit of the selected auxiliary bundle, not a universal advantage attributable solely to CFS. Extra pairwise supervision and the larger validation search budget are explicitly accounted for. All intervals are unadjusted 95% input-bootstrap intervals conditional on the three seeds.

## 8. Reviewer UWwP response

We measured 38,016 fixed-background second differences of actual subset KL on 64 tuning and 128 disjoint held-out inputs, using only frozen last-block interventions. An additive model has held-out-mask RMSE 0.1072/0.0821/0.0356 at K=3/6/9, and directly measured interactions confirm non-additivity. In disjoint-input ridge prediction controlling Taylor importance, contribution cosine, deletion effects, pair identity and background masks, CFS improves K=9 interaction prediction: the MSE reductions are 0.00002677 [0.00000906,0.00004457] for signed interactions and 0.00005111 [0.00002425,0.00007715] for their magnitude (exact values in the table). Gains at K=3/6 remain unresolved. Interaction complexity correlates with additive gating error, but its association with the selected CFS-regularized router's relative advantage remains unresolved. These results establish a restricted analytical and predictive use for CFS; they do not establish causal mediation, universally superior routing, or physical inference acceleration.

## 9. 代码、归档与复现

新增代码及用途：

| 文件 | 用途 |
| --- | --- |
| src/cfs_regularization.py | 与原函数数值一致的辅助目标、梯度诊断 |
| src/subset_interactions.py | 精确mask quartet、去重mask CV与可识别加性拟合 |
| scripts/prepare_cfs_regularization.py | 原辅助标签、固定测试索引和tune图像导出 |
| scripts/train_cfs_regularization.py | 原架构上的A1–A3训练及固定probe诊断 |
| scripts/run_cfs_regularization.py | 有限pilot、验证集锁定、补齐种子 |
| scripts/diagnose_existing_cfs.py | A0/A4原权重端点诊断，无更新 |
| scripts/collect_subset_interactions.py | 有限末层真实mask干预 |
| scripts/analyze_additive_subsets.py | held-out mask拟合及输入bootstrap |
| scripts/analyze_cfs_interactions.py | CFS/重要性/相似度相关和独立输入预测 |
| scripts/evaluate_cfs_regularization.py | 复用已有统一评估器及补偿接口 |
| scripts/analyze_cfs_routing.py | 三种子routing、配对CI、固定分组 |
| scripts/summarize_cfs_interaction_checks.py | 完整pair表、直接消融、sign与数值检查 |
| scripts/plot_cfs_interaction_results.py | 四张PDF/PNG图与全部gradient汇总 |
| scripts/verify_cfs_interaction_results.py | 选择锁、mask构造、训练完整性和基线核查 |
| scripts/report_cfs_interactions.py | 此报告的数字与表格生成 |
| tests/test_cfs_regularization.py | 原辅助损失一致性和梯度余弦 |
| tests/test_subset_interactions.py | quartet真实性、已知交互、加性恢复与重复mask隔离 |

53个项目测试通过。保留21个新checkpoint及日志、所有pilot验证结果、38016真实quartet的原mask/KL、56448条routing记录、逐input/pair预测、完整统计与图表；原baseline仍在独立目录。两处运行问题已记录：初版离线脚本反复解压同一NPZ而被停止，改为一次载入后执行相同统计；初次端点诊断缺少CUBLAS确定性环境变量，失败后在新目录按原设置重跑。均未产生用于选择的测试结果，也未改变目标或采样。

本地导出（仓库根目录；必须指定新输出目录）：

```powershell
python scripts/prepare_cfs_regularization.py --workspace D:/tries2 --output D:/tries2/outputs/cfs_regularization_inputs_new
python -m pytest -q -p no:cacheprovider
```

Linux复现目录应包含本次scripts/src、`inputs/`导出，以及只读 `baseline/` 指向b494b13完整实验目录；baseline中使用原base.pth。下列输出目录须未存在，完整复现不覆盖旧结果：

```bash
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4
python scripts/train_cfs_regularization.py --data baseline/learned_gating_inputs_20261115 --aux inputs --output smoke_A1 --epochs 2
python scripts/run_cfs_regularization.py --root .
python scripts/diagnose_existing_cfs.py --root .
python scripts/collect_subset_interactions.py --root . --output interaction_smoke --smoke
python scripts/collect_subset_interactions.py --root . --output interactions
python scripts/analyze_additive_subsets.py --root . --output additive_final --measurements interactions
python scripts/analyze_cfs_interactions.py --root . --output interaction_analysis
python scripts/evaluate_cfs_regularization.py --root .
python scripts/analyze_cfs_routing.py --root . --output routing_analysis
python scripts/summarize_cfs_interaction_checks.py --root . --output descriptive_checks
python scripts/plot_cfs_interaction_results.py --root . --output figures
python scripts/verify_cfs_interaction_results.py --root .
python scripts/report_cfs_interactions.py --root .
```

四张主图分别为 `figures/figure1_regularization_ablation`、`figure2_loss_gradient_diagnostics`、`figure3_cfs_interaction_prediction`、`figure4_interaction_conditioned_routing`，均有PDF/PNG。Table A/C的完整CSV/JSON/LaTeX在routing_analysis；Table B和全部对照在interaction_analysis、additive_final；descriptive_checks为额外透明性核查，不改变预注册主比较。
'''
    # Numerical reviewer statement is generated from the actual prediction table.
    text=text.replace('0.00002677 [0.00000906,0.00004457]',f'{inc9.loc["interaction","mean"]:.8f} [{inc9.loc["interaction","ci_low"]:.8f},{inc9.loc["interaction","ci_high"]:.8f}]')
    text=text.replace('0.00005111 [0.00002425,0.00007715]',f'{inc9.loc["magnitude","mean"]:.8f} [{inc9.loc["magnitude","ci_low"]:.8f},{inc9.loc["magnitude","ci_high"]:.8f}]')
    (root/'REPORT.md').write_text(text,encoding='utf-8')
    for directory in ('interaction_analysis','additive_final'):
        for name in (['interaction_summary','correlation_summary','heldout_prediction_summary','incremental_prediction'] if directory=='interaction_analysis' else ['additive_summary']):
            frame=pd.read_csv(root/directory/f'{name}.csv');frame.to_json(root/directory/f'{name}.json',orient='records',indent=2)
            frame.to_latex(root/directory/f'{name}.tex',index=False,float_format='%.6f')


if __name__=='__main__':main()
