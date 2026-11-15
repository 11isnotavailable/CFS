# Budget-aware A3 短程微调

日期：2026-11-15。起点commit60bf8a7。**没有找到满足验证集约束的更新；按预先规则保留原A3。** 四个锁定的诊断代表均未同时达到三个测试数值目标。

本轮仅执行方案一，没有启动Score融合、结构搜索、新Teacher或Backbone训练。原权重和结果不变。由于既有测试集已被用于提出本轮研究目标，这是一轮明确标注的后续探索；本轮内部仍严格只用验证集选择，不把它描述为从未观察过测试集的一次性确认实验。

## 1. 固定设置与预算

同一个825091参数的R2结构、2304维原始prehead输入、CFS+Alpha定义与λ=0.01。每个seed17/23/41从自身原A3的验证集选定Checkpoint开始。四组相对权重为(1,1,1)、(1,1,1.5)、(1,1,2)、(1,1,3)。各10epochs、batch64、lr3e-5、AdamW decay1e-4，无学习率或时长搜索。总12runs、7680updates。原checkpoint不含optimizer状态，因此所有run均重新初始化AdamW moments；这是短程fine-tuning，不是精确恢复原优化轨迹。

目标为 `0.5*((L3+L6+w9*L9)/(2+w9)+0.01*(L_CFS+0.1*L_alpha))`。Joint权重总和保持1，避免混淆总loss尺度变化；w9=1是等权继续训练对照。辅助定义、真实KL、mask池与原归一化均不变。模型架构与参数量未增加。

先测原A3实际起点的损失/梯度，再执行固定计划。下表为起点三个seed平均（共同global encoder+head embeddings），不是跨独立输入的置信区间：

| loss | value | gradient_norm | weighted_gradient_norm |
| --- | --- | --- | --- |
| alpha | 0.2058636 | 0.0653039 | 0.0000327 |
| cfs | 2.3776059 | 0.0864319 | 0.0004322 |
| joint_K3 | 3.5415733 | 0.1028890 | 0.0171482 |
| joint_K6 | 3.6971484 | 0.1094960 | 0.0182493 |
| joint_K9 | 3.6273493 | 0.1808380 | 0.0301397 |
| weighted_joint | 3.6220237 | 0.0501251 | 0.0250625 |

起点梯度cosine（三个seed的均值与范围）：

| left | right | mean | min | max |
| --- | --- | --- | --- | --- |
| alpha | cfs | 0.1770074 | -0.0240485 | 0.3786023 |
| alpha | joint_K3 | 0.0670105 | 0.0106810 | 0.1172091 |
| alpha | joint_K6 | 0.0038602 | -0.0280886 | 0.0665575 |
| alpha | joint_K9 | 0.0940021 | -0.0057957 | 0.2145458 |
| alpha | weighted_joint | 0.1513752 | 0.1060923 | 0.2404765 |
| cfs | joint_K3 | -0.0431891 | -0.2079130 | 0.1891558 |
| cfs | joint_K6 | 0.0899218 | -0.0327109 | 0.1762003 |
| cfs | joint_K9 | -0.1015969 | -0.2669440 | 0.1777973 |
| cfs | weighted_joint | -0.0848392 | -0.1678438 | 0.0192188 |
| joint_K3 | joint_K6 | 0.1166541 | -0.2117945 | 0.5159187 |
| joint_K3 | joint_K9 | -0.2321057 | -0.6892665 | 0.5137970 |
| joint_K3 | weighted_joint | 0.4680327 | 0.2481758 | 0.7608057 |
| joint_K6 | joint_K9 | -0.6239709 | -0.6702018 | -0.5607939 |
| joint_K6 | weighted_joint | 0.0715559 | -0.2972610 | 0.5399351 |
| joint_K9 | weighted_joint | 0.5070642 | 0.2016474 | 0.7169911 |

K9梯度并不更小；seed17的预算梯度范数约0.0824/0.1483/0.2037。各预算loss约3.536/3.698/3.618也不能简单用来比较有效优化贡献。重加权改变方向/取舍，不能被描述为已证实的梯度饥饿修复。epochs0/1/5/10的raw loss、梯度范数、加权贡献和cosine全部保存。

## 2. 只按验证集选择

验证数据为原64 tune输入和48候选mask/K；训练仍为原4096输入。按同一epoch平均三个seed的每预算regret，要求K3和K6不高于A3起点（容差1e-8），且K9更低。在合格项中按K9、平均regret、较小权重、较早epoch排序，选择一个所有seed共用的权重/epoch。若无合格项，保留A3。

原A3 tune平均regret：K3=0.01192148，K6=0.00841106，K9=0.00383460。

共40个weight/epoch候选，合格数为0。下表为每权重按预定规则锁定的代表：若无合格epoch，则仅用于诊断地选择最小K9 regret的epoch，不能替代主配置。

| weight9 | epoch | val_K3 | val_K6 | val_K9 | mean_regret | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000000 | 6 | 0.0147772 | 0.0109891 | 0.0040402 | 0.0099355 | False |
| 1.5000000 | 1 | 0.0153133 | 0.0089832 | 0.0040755 | 0.0094573 | False |
| 2.0000000 | 3 | 0.0146449 | 0.0095475 | 0.0040003 | 0.0093975 | False |
| 3.0000000 | 10 | 0.0150279 | 0.0105205 | 0.0039882 | 0.0098455 | False |

所有原始132行seed/weight/epoch验证轨迹在 `validation_trajectories.csv`，40个聚合候选在 `validation_candidates.csv`，选择结果在 `selection_lock.json`。各诊断代表可能对应不同微调长度，因此它们的测试差异不是纯粹的权重因果效应；相同epoch的验证轨迹可直接比较。没有为某个seed选择特有权重或使用测试标签选epoch。

## 3. 完整测试诊断

在selection lock生成后，统一评估四个固定代表，448输入、同一dense reference、原KL方向和预处理。原A3、Joint-only(A0)、原R2和Gating都保留。Mean [95% CI]：

| Method | K=3 | K=6 | K=9 |
| --- | --- | --- | --- |
| Gating | 0.1115365 [0.0975148, 0.1268990] | 0.0269924 [0.0226395, 0.0314269] | 0.0080800 [0.0065228, 0.0097850] |
| R2 | 0.1190073 [0.1037138, 0.1350883] | 0.0281352 [0.0240766, 0.0324784] | 0.0069980 [0.0056469, 0.0084921] |
| A0 | 0.1056227 [0.0917461, 0.1201822] | 0.0241335 [0.0202009, 0.0282368] | 0.0083268 [0.0067973, 0.0100778] |
| A3 | 0.1051673 [0.0913031, 0.1197498] | 0.0248006 [0.0207983, 0.0291227] | 0.0079536 [0.0064643, 0.0095895] |
| W1 | 0.1047997 [0.0910087, 0.1195664] | 0.0244053 [0.0204078, 0.0286334] | 0.0082116 [0.0065372, 0.0100408] |
| W1.5 | 0.1049432 [0.0911157, 0.1195534] | 0.0241008 [0.0201762, 0.0282225] | 0.0076694 [0.0061933, 0.0092926] |
| W2 | 0.1052489 [0.0913574, 0.1197858] | 0.0242021 [0.0202338, 0.0283507] | 0.0078032 [0.0062999, 0.0095382] |
| W3 | 0.1055423 [0.0918123, 0.1201097] | 0.0238456 [0.0198545, 0.0280769] | 0.0079744 [0.0062808, 0.0098293] |

测试数值目标为K3≤0.10517、K6≤0.02480、K9<0.00700。下面同时显示原未舍入值和逐项是否达到；最后一列primary严格来自验证锁，不随测试表变化：

| method | kl3 | kl6 | kl9 | pass_K3 | pass_K6 | pass_K9 | all_targets | primary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.1056227 | 0.0241335 | 0.0083268 | False | True | False | False | False |
| A3 | 0.1051673 | 0.0248006 | 0.0079536 | True | False | False | False | True |
| Gating | 0.1115365 | 0.0269924 | 0.0080800 | False | False | False | False | False |
| R2 | 0.1190073 | 0.0281352 | 0.0069980 | False | False | True | False | False |
| Taylor | 0.2104781 | 0.0512980 | 0.0137709 | False | False | False | False | False |
| W1 | 0.1047997 | 0.0244053 | 0.0082116 | True | True | False | False | False |
| W1.5 | 0.1049432 | 0.0241008 | 0.0076694 | True | True | False | False | False |
| W2 | 0.1052489 | 0.0242021 | 0.0078032 | False | True | False | False | False |
| W3 | 0.1055423 | 0.0238456 | 0.0079744 | False | True | False | False | False |

注意原A3的K6实际值约0.02480064，显示为0.02480；与用户表中精确阈值的微小舍入差没有实质意义。对原A3精确值的比较应看配对差，而不是只看小数位后的布尔标记。

更新−参考方法的配对KL差（负值有利于更新）：

| method | reference | keep | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| W1 | A3 | 3 | -0.0003676 | -0.0012966 | 0.0005024 |
| W1 | R2 | 3 | -0.0142076 | -0.0209289 | -0.0085990 |
| W1 | Gating | 3 | -0.0067369 | -0.0122443 | -0.0021513 |
| W1.5 | A3 | 3 | -0.0002241 | -0.0010779 | 0.0006386 |
| W1.5 | R2 | 3 | -0.0140641 | -0.0206090 | -0.0084681 |
| W1.5 | Gating | 3 | -0.0065933 | -0.0119931 | -0.0020100 |
| W2 | A3 | 3 | 0.0000817 | -0.0008341 | 0.0011006 |
| W2 | R2 | 3 | -0.0137583 | -0.0202223 | -0.0082628 |
| W2 | Gating | 3 | -0.0062876 | -0.0117488 | -0.0017541 |
| W3 | A3 | 3 | 0.0003751 | -0.0008963 | 0.0016961 |
| W3 | R2 | 3 | -0.0134649 | -0.0198204 | -0.0080498 |
| W3 | Gating | 3 | -0.0059942 | -0.0116607 | -0.0013549 |
| W1 | A3 | 6 | -0.0003954 | -0.0009059 | 0.0001038 |
| W1 | R2 | 6 | -0.0037300 | -0.0059454 | -0.0015774 |
| W1 | Gating | 6 | -0.0025872 | -0.0049100 | -0.0004366 |
| W1.5 | A3 | 6 | -0.0006998 | -0.0013091 | -0.0002165 |
| W1.5 | R2 | 6 | -0.0040344 | -0.0061908 | -0.0019891 |
| W1.5 | Gating | 6 | -0.0028916 | -0.0051364 | -0.0007240 |
| W2 | A3 | 6 | -0.0005986 | -0.0012029 | -0.0000785 |
| W2 | R2 | 6 | -0.0039332 | -0.0061037 | -0.0017809 |
| W2 | Gating | 6 | -0.0027904 | -0.0050184 | -0.0006963 |
| W3 | A3 | 6 | -0.0009551 | -0.0017461 | -0.0002530 |
| W3 | R2 | 6 | -0.0042897 | -0.0063927 | -0.0022576 |
| W3 | Gating | 6 | -0.0031469 | -0.0055263 | -0.0009613 |
| W1 | A3 | 9 | 0.0002580 | -0.0001631 | 0.0008930 |
| W1 | R2 | 9 | 0.0012136 | 0.0003071 | 0.0022271 |
| W1 | Gating | 9 | 0.0001317 | -0.0004838 | 0.0007272 |
| W1.5 | A3 | 9 | -0.0002843 | -0.0004627 | -0.0001252 |
| W1.5 | R2 | 9 | 0.0006713 | -0.0004726 | 0.0017765 |
| W1.5 | Gating | 9 | -0.0004106 | -0.0010375 | 0.0001690 |
| W2 | A3 | 9 | -0.0001504 | -0.0004331 | 0.0001588 |
| W2 | R2 | 9 | 0.0008052 | -0.0003749 | 0.0019600 |
| W2 | Gating | 9 | -0.0002768 | -0.0009132 | 0.0003295 |
| W3 | A3 | 9 | 0.0000208 | -0.0005235 | 0.0007094 |
| W3 | R2 | 9 | 0.0009763 | 0.0000276 | 0.0020206 |
| W3 | Gating | 9 | -0.0001056 | -0.0007278 | 0.0004909 |

区间基于2000次独立输入Bootstrap，seed42；先对同一输入的三个训练seed平均，避免把seed当额外输入。各seed均值、SD和CI保留在 `per_seed_summary.csv`；所有区间未作多重比较校正，且条件于这三个训练种子。没有将均值胜出写成统计优势，没有以区间跨零宣称等效。

## 4. 解释与限制

等权继续训练对照也未通过验证约束，因此退化不能只归因于增加K9权重。原A3本来就是从60epoch轨迹中按验证集挑出的Checkpoint，继续训练不保证验证排序继续改善；fresh AdamW、64-input验证集和候选池排名的离散性也可能影响结果。这些是可能解释，不是已识别原因。

在测试诊断中W1.5的三个KL均值都略低于A3，但K9仍约0.00767，没有达到0.00700目标。其验证代表本身不合格，不能因这个测试观察而升格为主结果。验证指标是48候选集合内的regret，实际部署在全部220/924/220个子集上选择；两者不同，可能存在代理指标偏差，但本轮没有在看过测试后重新定义验证标准。

这次固定范围的尝试不能证明“原R2的K9优势必然来自Structured Auxiliary”，也不能排除其他小规模策略；但结果不足以支持把本轮更新替换为主方法。没有根据测试结果再扩展权重、学习率、训练长度或转向融合搜索。

共享原R2补偿的副结果与Accuracy完整保留在 `summary.csv/json/tex`、`paired.csv/json/tex`，但均不用于选择，也不用于替代未达标的shared-hard主结果。共享补偿器仍来自原R2，附加参数/计算边界与此前报告一致。没有实际backbone速度提升声明。

## 5. 可用于Rebuttal的客观说明

We conducted a bounded budget-aware continuation experiment on the unchanged 825,091-parameter A3 router: four prespecified K=9 weights (1, 1.5, 2, 3), three fixed seeds, and ten low-learning-rate epochs per run. Joint weights were normalized to preserve the overall loss scale. Checkpoint selection used only the existing validation set, requiring no degradation at K=3/6 and an improvement at K=9. No candidate satisfied these constraints, so we retained the original A3 rather than selecting a favorable test result. All fixed representatives, including the equal-weight continuation control, are reported. This finite negative or mixed result does not establish that budget imbalance is the cause of the original R2/A3 trade-off, nor that auxiliary supervision uniquely explains the K=9 gap.

## 6. 交付、检查与复现

新增文件：`src/budget_aware_objective.py`定义归一化重加权和受约束验证选择；`train_budget_aware_cfs.py`微调与诊断；`run_budget_aware_cfs.py`固定12runs及选择锁；`evaluate_budget_aware_cfs.py`复用原评估；`analyze_budget_aware_cfs.py`配对统计/图表；`verify_budget_aware_cfs.py`核验源权重、固定范围与选择；`report_budget_aware_cfs.py`生成此报告。测试在 `tests/test_budget_aware_objective.py`，验证w9=1与A3目标相同、梯度比例正确、验证约束和fallback生效。

完整输出含12个锁定代表的权重、所有120个训练epoch的日志/验证记录、梯度诊断、72576条逐输入记录、Bootstrap抽样索引、CSV/JSON/LaTeX、PDF/PNG。所有epoch权重另保留在远端runs目录；本地紧凑归档只包含预定代表，不删除远端其他Checkpoint。依赖的A3、Teacher、Backbone仍使用此前归档，不重复复制。

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
