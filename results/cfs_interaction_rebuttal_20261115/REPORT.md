# CFS Routing 监督优化与非加性功能交互分析

日期：2026-11-15。原始基线：commit `b494b13`。原始数据、Checkpoint 和论文正文未修改；未训练 Backbone。

**核心结论：得到预算相关的正面证据，但没有证明完整的因果机制链。** 验证集预选 A3（Joint+CFS+Alpha，λ=0.01）在 K=9 相比 Joint-only 降低 KL 约4.48%，配对95% CI不含零；K=3、6未确认增量改善。纯 Joint+CFS 的 A1 没有在任何 K 明确改善 Joint-only。CFS在控制重要性、相似度与上下文后，对K=9二阶交互有独立输入上的增量预测价值；交互复杂度与A3的相对Routing优势关联仍未分辨，不能将预测价值写成已验证的因果解释。

## 1. 审计与冻结方案

附件四行均值由已有CSV精确复现。此次全部基线重新执行，最大逐输入KL差为 4.44e-16（CSV浮点往返量级）。所有新消融与基线使用同一Dense logits，448输入、种子17/23/41、K=3/6/9。Imagenette2-160 train/tune/test为4096/64/448；ViT-B AugReg frozen，最后attention block11、12heads；原始checkpoint SHA256为 `683c6d82c1e832981a5fdce9ae66be4f02542b4bd7b7fc822b08f53c005de8df`。

R2是2304维prehead特征的pair scorer，825091参数；Gating是822460参数的预算无关加性head score，因此其Top-K集合跨K嵌套。输入使用原FP16缓存边界和训练集归一化。R2允许非嵌套子集，并枚举220/924/220个候选；不能把架构/搜索优势自动归因于CFS标签。沿用bicubic、224、crop_pct0.9、mean/std0.5。

原R2包含representative listwise CE、0.5 score SmoothL1、0.5 deletion CE、0.1 alpha loss、1.0 structured-subset loss和0.5 joint-set loss。新A1–A3刻意不含旧structured-subset辅助目标，因其同时依赖CFS与deletion，混入会破坏消融含义。所有pairwise原损失与原函数逐项一致，单元测试验证。源CFS训练值仍按原方法clip[-1,1]，alpha clip[0,3]，对角线规则不变；交互解释使用原始未裁剪CFS及both-source drop>1e-5过滤。

新目标为 `0.5 × [L_joint + λ(L_CFS + d×0.5L_del + a×0.1L_alpha)]`，L_CFS=原listwise+0.5原score regression；A1:(d,a)=(0,0)，A2:(1,0)，A3:(0,1)。A1/A2冻结未参与目标的alpha输出，824962优化参数；A3优化825091参数。A0、A4原权重复用。

每个新run仍为AdamW lr3e-4、decay1e-4、batch64、60epochs（3840updates），同一4096输入、每输入每K48条joint KL。15个seed17 pilots覆盖A1/A2/A3×λ{0.01,0.03,0.1,0.3,1}；λ0由已有A0提供。各变体只凭64个tune输入平均regret选λ，均选0.01，然后补种子23/41，共21个新run。headline在新测试前锁定A3；没有根据测试上A2在K3的更低均值改选模型。单run预算匹配，**调参预算和总监督成本并不与旧单配置基线相同**。未扩展λ网格，也未追加梯度手术或warmup搜索。

完整pilot验证结果（未选配置不做测试集排名）：

| variant | strength | epoch | val_joint_regret | val_K3 | val_K6 | val_K9 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 0.010000 | 27.000000 | 0.008432 | 0.012342 | 0.010038 | 0.002917 |
| A1 | 0.030000 | 29.000000 | 0.009043 | 0.013968 | 0.009518 | 0.003643 |
| A1 | 0.100000 | 29.000000 | 0.009046 | 0.015198 | 0.007502 | 0.004438 |
| A1 | 0.300000 | 36.000000 | 0.010499 | 0.015845 | 0.010992 | 0.004661 |
| A1 | 1.000000 | 25.000000 | 0.013427 | 0.019575 | 0.014708 | 0.005998 |
| A2 | 0.010000 | 31.000000 | 0.008024 | 0.011078 | 0.010223 | 0.002770 |
| A2 | 0.030000 | 27.000000 | 0.008844 | 0.011440 | 0.012695 | 0.002396 |
| A2 | 0.100000 | 45.000000 | 0.009520 | 0.016924 | 0.007841 | 0.003794 |
| A2 | 0.300000 | 21.000000 | 0.010446 | 0.013140 | 0.014967 | 0.003230 |
| A2 | 1.000000 | 37.000000 | 0.012018 | 0.021113 | 0.012208 | 0.002734 |
| A3 | 0.010000 | 22.000000 | 0.007418 | 0.010868 | 0.008533 | 0.002854 |
| A3 | 0.030000 | 22.000000 | 0.008530 | 0.011718 | 0.008998 | 0.004875 |
| A3 | 0.100000 | 29.000000 | 0.010450 | 0.015154 | 0.011842 | 0.004354 |
| A3 | 0.300000 | 27.000000 | 0.009984 | 0.014550 | 0.012412 | 0.002990 |
| A3 | 1.000000 | 23.000000 | 0.012607 | 0.015855 | 0.015900 | 0.006065 |

额外CFS标签来自既有缓存：train网格16点，tune/test实际31点，均为[0,3]；新优化仅用train标签，checkpoint选择仅用tune joint KL。每训练输入132个有向CFS标签，A2另用12个deletion值，A3另用132个oracle alpha标签。原始采集名义上需132×16=2112次pair intervention加12次deletion/input，4096输入约8,699,904次逻辑变体（不含dense/梯度/实现开销）；已有joint teacher为589824条mask KL。Alpha由同一搜索导出，无另一次搜索。这些不是实测GPU时间，缓存复用也不能称额外监督零成本。

## 2. Table A：完整Routing结果

Mean [95% CI]，先同输入平均三种子，再以独立输入做2000次Bootstrap，seed42。KL越低越好。所有逐项区间均未作多重比较校正，条件于这三个训练种子；种子均值SD、逐种子CI另存。

共享硬门控（主对照）：

| Method | K=3 | K=6 | K=9 |
| --- | --- | --- | --- |
| Taylor | 0.210478 [0.183057, 0.241383] | 0.051298 [0.042772, 0.061774] | 0.013771 [0.010342, 0.018293] |
| Gating | 0.111537 [0.097515, 0.126899] | 0.026992 [0.022640, 0.031427] | 0.008080 [0.006523, 0.009785] |
| A0 | 0.105623 [0.091746, 0.120182] | 0.024134 [0.020201, 0.028237] | 0.008327 [0.006797, 0.010078] |
| R2 | 0.119007 [0.103714, 0.135088] | 0.028135 [0.024077, 0.032478] | 0.006998 [0.005647, 0.008492] |
| A1 | 0.105891 [0.091983, 0.120459] | 0.025010 [0.020844, 0.029380] | 0.007960 [0.006440, 0.009638] |
| A2 | 0.104976 [0.091207, 0.119505] | 0.025094 [0.020951, 0.029512] | 0.008042 [0.006401, 0.009872] |
| A3 | 0.105167 [0.091303, 0.119750] | 0.024801 [0.020798, 0.029123] | 0.007954 [0.006464, 0.009590] |

A3−参考方法的配对KL差，负值有利于A3：

| reference | keep | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| A0 | 3 | -0.000455 | -0.001380 | 0.000504 |
| Gating | 3 | -0.006369 | -0.011816 | -0.001997 |
| R2 | 3 | -0.013840 | -0.020230 | -0.008206 |
| A0 | 6 | 0.000667 | -0.000068 | 0.001528 |
| Gating | 6 | -0.002192 | -0.004400 | -0.000116 |
| R2 | 6 | -0.003335 | -0.005520 | -0.001142 |
| A0 | 9 | -0.000373 | -0.000730 | -0.000072 |
| Gating | 9 | -0.000126 | -0.000760 | 0.000450 |
| R2 | 9 | 0.000956 | -0.000182 | 0.002080 |

**Q1：辅助监督能否改善Joint-only？** 证据仅支持A3组合在K9的小幅改善：−0.000373 [−0.000730,−0.000072]。K3为−0.000455 [−0.001380,0.000504]；K6为+0.000667 [−0.000068,0.001528]，点估计反而更差。纯CFS的A1在K9为−0.000367 [−0.000786,0.000036]，仍跨零；A2各预算也跨零。因此不能宣称CFS单独监督已被证明带来稳定增益，也不能把A3区间更窄解释为alpha已被证明有效。

辅助目标直接对照（新增描述性比较，不用于选择）：

| contrast | keep | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| A3_minus_A1 | 3 | -0.000724 | -0.002921 | 0.001060 |
| A2_minus_A1 | 3 | -0.000916 | -0.003767 | 0.001230 |
| A3_minus_A2 | 3 | 0.000192 | -0.001392 | 0.001889 |
| A3_minus_A1 | 6 | -0.000209 | -0.001174 | 0.000624 |
| A2_minus_A1 | 6 | 0.000085 | -0.000691 | 0.000819 |
| A3_minus_A2 | 6 | -0.000294 | -0.001386 | 0.000771 |
| A3_minus_A1 | 9 | -0.000007 | -0.000308 | 0.000282 |
| A2_minus_A1 | 9 | 0.000082 | -0.000443 | 0.000608 |
| A3_minus_A2 | 9 | -0.000088 | -0.000686 | 0.000474 |

## 3. 梯度与训练目标诊断

在固定前64个训练输入上，eval模式、epoch0/1/10/30/60，对各目标在共同global encoder和head embeddings上的梯度记录范数与cosine。所有原始loss曲线、逐预算joint曲线和梯度在 `figures/selected_training_history.csv`、`all_gradient_diagnostics.csv` 保存。原A0/A4只能做保留checkpoint的端点诊断，不能伪称已记录历史训练轨迹。

下表是选定训练轨迹或端点的描述性统计，snapshots不是独立样本，不用它们构造伪置信区间。Figure 2梯度阴影表示三个种子的min/max范围，不是Bootstrap CI：

| variant | left | right | snapshots | mean_cosine | min_cosine | max_cosine | negative_fraction | left_norm_mean | right_norm_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | alpha | joint | 3 | -0.045162 | -0.066903 | -0.006808 | 1.000000 | 0.070676 | 0.047812 |
| A0 | cfs | joint | 3 | 0.027096 | -0.074685 | 0.094058 | 0.333333 | 0.198438 | 0.047812 |
| A0 | deletion | joint | 3 | 0.140233 | -0.054003 | 0.241245 | 0.333333 | 0.357457 | 0.047812 |
| A0 | joint | original_auxiliary_total | 3 | 0.378463 | 0.264164 | 0.484857 | 0.000000 | 0.047812 | 0.882675 |
| A0 | joint | original_structured | 3 | 0.450305 | 0.361519 | 0.518101 | 0.000000 | 0.047812 | 0.675529 |
| A3 | alpha | joint | 15 | 0.042212 | -0.088657 | 0.329678 | 0.333333 | 0.068185 | 0.067788 |
| A3 | cfs | joint | 15 | 0.025954 | -0.428557 | 0.463825 | 0.400000 | 0.127523 | 0.067788 |
| A3 | deletion | joint | 15 | 0.146415 | -0.362479 | 0.758398 | 0.333333 | 0.244377 | 0.067788 |
| A4 | alpha | joint | 3 | 0.044372 | -0.087484 | 0.138374 | 0.333333 | 0.065924 | 0.701012 |
| A4 | cfs | joint | 3 | 0.182910 | 0.055222 | 0.305592 | 0.000000 | 0.172049 | 0.701012 |
| A4 | deletion | joint | 3 | 0.143685 | 0.014743 | 0.233751 | 0.000000 | 0.134261 | 0.701012 |
| A4 | joint | original_auxiliary_total | 3 | -0.743934 | -0.842945 | -0.621567 | 1.000000 | 0.701012 | 0.381458 |
| A4 | joint | original_structured | 3 | -0.776046 | -0.841445 | -0.649824 | 1.000000 | 0.701012 | 0.421669 |

原R2三个端点的joint与旧structured auxiliary均反向，平均cosine约−0.776；与整个原auxiliary bundle约−0.744。相比之下，joint与纯pairwise CFS约+0.183。因此不能简单断言“CFS loss本身导致原R2失败”。原R2端点的总joint与K9 joint方向也相反（约−0.288），存在预算取舍。A3训练probe中CFS与joint约40%的快照为负，说明冲突随训练阶段变化。

这些结果支持“原多目标可能存在优化取舍”的诊断，**不是性能下降的因果证明**：多目标最优点本来就可能存在梯度抵消；固定probe也不代表全训练分布。弱辅助权重在验证集优于强权重，但本次没有验证某一梯度处理算法能因果消除问题。

## 4. Shared compensation与系统归因

每个种子把同一个原R2的score/alpha及原 `compose_scales` 应用于所有mask。共享应用并不消除原补偿器与R2共同训练的偏向。A3在此使用的是**原R2补偿器**，不是把自己训练的alpha作为更强专用补偿；A3 alpha监督只通过router训练影响本次主selection对照。

| Method | K=3 | K=6 | K=9 |
| --- | --- | --- | --- |
| Taylor | 0.154732 [0.134110, 0.176835] | 0.040980 [0.034337, 0.048862] | 0.013095 [0.010576, 0.016269] |
| Gating | 0.088674 [0.075927, 0.102532] | 0.027013 [0.022405, 0.031844] | 0.008209 [0.006700, 0.009861] |
| A0 | 0.084520 [0.072268, 0.096602] | 0.025434 [0.021242, 0.029880] | 0.009094 [0.007368, 0.011108] |
| R2 | 0.085139 [0.073288, 0.097218] | 0.023228 [0.019603, 0.027136] | 0.006323 [0.005015, 0.007869] |
| A1 | 0.086619 [0.074247, 0.098631] | 0.025954 [0.021665, 0.030489] | 0.008662 [0.007014, 0.010618] |
| A2 | 0.084894 [0.072799, 0.097234] | 0.025482 [0.021443, 0.029852] | 0.008763 [0.006956, 0.010799] |
| A3 | 0.085199 [0.072757, 0.097431] | 0.026076 [0.021777, 0.030675] | 0.008673 [0.007014, 0.010571] |

原完整R2 merge、A0/A1/A2/A3的同补偿变体，与Gating-hard的差异及选择/补偿算术分解全部在 `pipeline_decomposition.csv/json/tex`。分解不表示因果独立贡献。带原R2补偿的Gating/新router包含额外原R2参数与前向成本；不是同参数独立部署。旧forward成本和Top-K/穷举差异见b494b13报告；不声称backbone wall-clock加速。Accuracy、逐种子波动和准确率差的完整结果见 `routing_summary`、`paired_routing`。

## 5. Table B：真实非加性与CFS预测信息

B1对已有4096 train与64 tune的48-mask缓存做每输入、每K四折mask预测。重复mask归同一fold；训练集中心化、pseudoinverse最小范数，固定K的权重和为零，截距由训练均值恢复。理论有效秩12，逐输入实际rank保留；少量原anchor池fold缺秩，未隐去这些输入。新uniform-mask测试样本全部达到rank12。这里是同输入的held-out mask诊断，不是可部署router：它需要该输入的36个teacher masks。

完整误差（列出RMSE、训练误差与常数预测误差；其他MAE/SD同文件）：

| split | source | keep | metric | n | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | new_uniform_masks | 3 | rmse | 128 | 0.107181 | 0.091682 | 0.123484 |
| test | new_uniform_masks | 3 | train_rmse | 128 | 0.070215 | 0.060208 | 0.080900 |
| test | new_uniform_masks | 3 | constant_rmse | 128 | 0.330169 | 0.283450 | 0.378523 |
| test | new_uniform_masks | 6 | rmse | 128 | 0.082097 | 0.068613 | 0.096422 |
| test | new_uniform_masks | 6 | train_rmse | 128 | 0.054578 | 0.045859 | 0.063775 |
| test | new_uniform_masks | 6 | constant_rmse | 128 | 0.168423 | 0.138865 | 0.197564 |
| test | new_uniform_masks | 9 | rmse | 128 | 0.035572 | 0.028173 | 0.043747 |
| test | new_uniform_masks | 9 | train_rmse | 128 | 0.023874 | 0.018907 | 0.029323 |
| test | new_uniform_masks | 9 | constant_rmse | 128 | 0.068407 | 0.054532 | 0.083547 |
| train | cached_anchor_pool | 3 | rmse | 4096 | 0.124168 | 0.120924 | 0.127492 |
| train | cached_anchor_pool | 3 | train_rmse | 4096 | 0.080748 | 0.078579 | 0.082935 |
| train | cached_anchor_pool | 3 | constant_rmse | 4096 | 0.310636 | 0.302072 | 0.319222 |
| train | cached_anchor_pool | 6 | rmse | 4096 | 0.040676 | 0.039281 | 0.042085 |
| train | cached_anchor_pool | 6 | train_rmse | 4096 | 0.023968 | 0.023134 | 0.024760 |
| train | cached_anchor_pool | 6 | constant_rmse | 4096 | 0.087123 | 0.084256 | 0.090010 |
| train | cached_anchor_pool | 9 | rmse | 4096 | 0.016694 | 0.015982 | 0.017408 |
| train | cached_anchor_pool | 9 | train_rmse | 4096 | 0.010289 | 0.009862 | 0.010723 |
| train | cached_anchor_pool | 9 | constant_rmse | 4096 | 0.043534 | 0.041732 | 0.045250 |
| tune | cached_anchor_pool | 3 | rmse | 64 | 0.119820 | 0.098477 | 0.142542 |
| tune | cached_anchor_pool | 3 | train_rmse | 64 | 0.076742 | 0.063276 | 0.090889 |
| tune | cached_anchor_pool | 3 | constant_rmse | 64 | 0.287177 | 0.228811 | 0.351477 |
| tune | cached_anchor_pool | 6 | rmse | 64 | 0.038281 | 0.029622 | 0.047459 |
| tune | cached_anchor_pool | 6 | train_rmse | 64 | 0.023775 | 0.018358 | 0.029613 |
| tune | cached_anchor_pool | 6 | constant_rmse | 64 | 0.076147 | 0.057984 | 0.095485 |
| tune | cached_anchor_pool | 9 | rmse | 64 | 0.018499 | 0.012615 | 0.025213 |
| tune | cached_anchor_pool | 9 | train_rmse | 64 | 0.010215 | 0.007240 | 0.013572 |
| tune | cached_anchor_pool | 9 | constant_rmse | 64 | 0.034987 | 0.024806 | 0.045894 |
| tune | new_uniform_masks | 3 | rmse | 64 | 0.131483 | 0.110016 | 0.155620 |
| tune | new_uniform_masks | 3 | train_rmse | 64 | 0.088344 | 0.073737 | 0.104634 |
| tune | new_uniform_masks | 3 | constant_rmse | 64 | 0.387815 | 0.321047 | 0.461477 |
| tune | new_uniform_masks | 6 | rmse | 64 | 0.094538 | 0.075932 | 0.114303 |
| tune | new_uniform_masks | 6 | train_rmse | 64 | 0.063402 | 0.050669 | 0.076419 |
| tune | new_uniform_masks | 6 | constant_rmse | 64 | 0.179088 | 0.140813 | 0.220398 |
| tune | new_uniform_masks | 9 | rmse | 64 | 0.035434 | 0.026782 | 0.044743 |
| tune | new_uniform_masks | 9 | train_rmse | 64 | 0.024078 | 0.018090 | 0.030823 |
| tune | new_uniform_masks | 9 | constant_rmse | 64 | 0.062882 | 0.047113 | 0.080169 |

**Q2：加性建模是否有限？** 在128个测试输入的uniform48-mask上，held-out RMSE为0.10718/0.08210/0.03557，训练RMSE为0.07022/0.05458/0.02387。加性拟合明显优于常数，但并非精确拟合。不能仅凭预测误差证明非加性；B2直接测量提供了独立证据。

旧穷举oracle只保留最优mask，没有全部KL。新增测量固定64 tune+随机种子42选出的128 test，对每输入每K的66对head抽一个不含i,j、大小K−2的T，真正评估四个mask。共38016个quartet，去重后140128个mask forward，只重放最后block；另存每输入每K48个uniform mask供B1。未训练模型，也未重算pairwise CFS。

`I=D(T+i+j)-D(T+i)-D(T+j)+D(T)`。正值表示加入j后i的边际KL变化变大；若两者各自降低KL，可称KL改善的边际递减。负值相反。KL不一定单调，不能仅凭符号叫语义冗余/协同；输出KL自身的非线性就能产生交互。T在输入间也变化，不能把所有跨输入差异归因于输入本身。

| keep | metric | n | mean | sd | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | complexity | 128 | 0.098832 | 0.089681 | 0.083777 | 0.115093 |
| 3 | signed_mean | 128 | 0.096392 | 0.090268 | 0.081314 | 0.112696 |
| 3 | positive_fraction | 128 | 0.893939 | 0.102198 | 0.876062 | 0.910991 |
| 3 | valid_fraction | 128 | 0.958215 | 0.094588 | 0.941048 | 0.973369 |
| 6 | complexity | 128 | 0.041271 | 0.038201 | 0.035113 | 0.047889 |
| 6 | signed_mean | 128 | 0.037899 | 0.038018 | 0.031739 | 0.044517 |
| 6 | positive_fraction | 128 | 0.817708 | 0.126780 | 0.796396 | 0.840436 |
| 6 | valid_fraction | 128 | 0.958215 | 0.094588 | 0.941048 | 0.973369 |
| 9 | complexity | 128 | 0.016433 | 0.014869 | 0.013969 | 0.019037 |
| 9 | signed_mean | 128 | 0.012509 | 0.012513 | 0.010425 | 0.014681 |
| 9 | positive_fraction | 128 | 0.738281 | 0.135248 | 0.715433 | 0.762902 |
| 9 | valid_fraction | 128 | 0.958215 | 0.094588 | 0.941048 | 0.973369 |

全部66个Head Pair的统计见 `descriptive_checks/all_head_pair_statistics.csv`，保留所有pair而非最佳案例。前4输入的dense与四个直接干预检查误差均为0；全192输入dense重构最大误差 0。批形状导致的最大noop KL为 0.00000731，mean |I|为0.09883/0.04127/0.01643。`numerical_noise_diagnostic.csv`保留接近4×noop的比例；noop只是数值诊断，不是严格的四项误差上界，未据此筛选结果。

B3使用两个source drop均>1e-5的pairs，测试保留95.82%，tune约95.00%。原始CFS值用于Spearman；回归只对CFS特征做signed log1p以控制负尾。Contribution cosine来自同一模型的post-WO贡献、全token展平、无bias；Taylor与deletion沿用原缓存。

逐输入内Spearman的平均及输入Bootstrap区间：

| keep | feature | target | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| 3 | cfs_asymmetry | interaction | 0.088481 | 0.058234 | 0.117719 |
| 3 | cfs_asymmetry | magnitude | 0.102812 | 0.073168 | 0.131148 |
| 3 | cfs_mean | interaction | 0.075501 | 0.040192 | 0.110349 |
| 3 | cfs_mean | magnitude | 0.110212 | 0.074977 | 0.145096 |
| 3 | cosine | interaction | 0.174672 | 0.132248 | 0.218026 |
| 3 | cosine | magnitude | 0.164083 | 0.122567 | 0.206366 |
| 3 | taylor_mean | interaction | 0.358409 | 0.321201 | 0.394380 |
| 3 | taylor_mean | magnitude | 0.410526 | 0.375779 | 0.444437 |
| 6 | cfs_asymmetry | interaction | 0.092390 | 0.062784 | 0.119416 |
| 6 | cfs_asymmetry | magnitude | 0.124209 | 0.095595 | 0.150899 |
| 6 | cfs_mean | interaction | 0.098672 | 0.063179 | 0.133696 |
| 6 | cfs_mean | magnitude | 0.184367 | 0.151056 | 0.215423 |
| 6 | cosine | interaction | 0.174962 | 0.137076 | 0.213294 |
| 6 | cosine | magnitude | 0.145864 | 0.103911 | 0.187775 |
| 6 | taylor_mean | interaction | 0.294650 | 0.260926 | 0.327122 |
| 6 | taylor_mean | magnitude | 0.426976 | 0.396999 | 0.457372 |
| 9 | cfs_asymmetry | interaction | 0.110395 | 0.083418 | 0.137535 |
| 9 | cfs_asymmetry | magnitude | 0.159853 | 0.131131 | 0.186334 |
| 9 | cfs_mean | interaction | 0.129528 | 0.093715 | 0.164946 |
| 9 | cfs_mean | magnitude | 0.297101 | 0.265103 | 0.325891 |
| 9 | cosine | interaction | 0.178738 | 0.143443 | 0.215134 |
| 9 | cosine | magnitude | 0.129069 | 0.087977 | 0.170923 |
| 9 | taylor_mean | interaction | 0.239399 | 0.204701 | 0.272507 |
| 9 | taylor_mean | magnitude | 0.468521 | 0.438273 | 0.498741 |

相关并非CFS独占：Taylor通常与交互幅度有更强的单变量相关性。增量检验更重要：固定ridge α=1，64 tune输入拟合、128 disjoint test输入预测。所有arm共享pair identity与背景mask控制；基础特征为Cosine、Taylor pair均值/差、log-deletion均值/差；+CFS添加双向均值、max、非对称度。分别预测signed I与|I|，所有标准化只用tune，绝不按test选正则化或特征。

MSE(controls)−MSE(controls+CFS)，正值表示CFS改善：

| keep | target | n | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| 3 | interaction | 128 | 0.000278 | -0.000114 | 0.000673 |
| 3 | magnitude | 128 | 0.000307 | -0.000079 | 0.000699 |
| 6 | interaction | 128 | 0.000063 | -0.000038 | 0.000163 |
| 6 | magnitude | 128 | 0.000092 | -0.000012 | 0.000197 |
| 9 | interaction | 128 | 0.000027 | 0.000009 | 0.000045 |
| 9 | magnitude | 128 | 0.000051 | 0.000024 | 0.000077 |

**Q3：CFS是否提供增量信息？** K9支持：signed I的MSE改善 0.00002669 [0.00000903,0.00004492]，|I|改善 0.00005102 [0.00002448,0.00007739]。K3/6点估计为正但区间跨零。完整controls/CFS-only/+CFS误差、符号预测和恒正基线均保存。由于I多数为正，不能只凭普通sign accuracy很高宣称有效识别符号；balanced accuracy与恒正基线见 `sign_prediction_summary.csv`。

## 6. Table C：交互与Routing优势是否连起来？

复杂度预定义为每输入66对head的mean |I|，每K分别用64 tune输入中位数定high/low阈值；不使用test性能定义分组。以下保留全448、测量子集128及其两组。表中是Gating−A3，正值有利于A3。

| keep | group | n | gating_kl | A3_kl | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | all448 | 448 | 0.111537 | 0.105167 | 0.006369 | 0.001997 | 0.011816 |
| 3 | measured128 | 128 | 0.109738 | 0.107473 | 0.002265 | -0.003238 | 0.008252 |
| 3 | low | 86 | 0.104631 | 0.103185 | 0.001446 | -0.005889 | 0.009103 |
| 3 | high | 42 | 0.120196 | 0.116253 | 0.003943 | -0.003576 | 0.011218 |
| 6 | all448 | 448 | 0.026992 | 0.024801 | 0.002192 | 0.000116 | 0.004400 |
| 6 | measured128 | 128 | 0.028661 | 0.026048 | 0.002613 | -0.001160 | 0.007478 |
| 6 | low | 75 | 0.010717 | 0.010208 | 0.000510 | -0.000255 | 0.001306 |
| 6 | high | 53 | 0.054053 | 0.048465 | 0.005588 | -0.003151 | 0.017260 |
| 9 | all448 | 448 | 0.008080 | 0.007954 | 0.000126 | -0.000450 | 0.000760 |
| 9 | measured128 | 128 | 0.008516 | 0.007991 | 0.000525 | -0.000696 | 0.001917 |
| 9 | low | 65 | 0.000652 | 0.000933 | -0.000280 | -0.000720 | 0.000080 |
| 9 | high | 63 | 0.016629 | 0.015273 | 0.001356 | -0.000959 | 0.004167 |

完整分组表同时含Gating/A0/原R2/A1/A2/A3的绝对KL；全组均报告。连续关系如下：

| keep | outcome | n | rho | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| 3 | gating_kl | 128 | 0.441883 | 0.288167 | 0.571740 |
| 3 | gating_minus_A0 | 128 | 0.174459 | 0.022739 | 0.333165 |
| 3 | gating_minus_A3 | 128 | 0.121853 | -0.036097 | 0.279464 |
| 6 | gating_kl | 128 | 0.625132 | 0.500817 | 0.727579 |
| 6 | gating_minus_A0 | 128 | 0.040652 | -0.144959 | 0.217713 |
| 6 | gating_minus_A3 | 128 | 0.057380 | -0.125060 | 0.231573 |
| 9 | gating_kl | 128 | 0.821787 | 0.755132 | 0.866942 |
| 9 | gating_minus_A0 | 128 | 0.268196 | 0.096670 | 0.434879 |
| 9 | gating_minus_A3 | 128 | 0.145322 | -0.042226 | 0.332825 |

**Q4：能否证明这些信息转化为Routing优势？** 有限支持，不能完成强机制论证。复杂度与Gating KL的Spearman为0.442/0.625/0.822，区间都不含零，表明更强交互伴随更难的加性路由。但复杂度与A3相对优势的相关为0.122/0.057/0.145，区间均跨零；high组虽有较大的优势点估计，也未稳定分辨。A0在K3/K9、A1在K9等有部分条件关联，反而说明架构本身也可能解释收益。`high_minus_low_advantage.csv`提供直接组间差；不得用“某组显著、另一组不显著”冒充组间显著差。

因此可报告三个分别成立的观察：真实KL非加性；CFS在K9提供额外交互预测信息；预选弱CFS+alpha监督在K9改善Joint-only。不能据此声称三者存在已验证的中介因果链。B1/B2复杂度和真实CFS在这里是oracle分析量，绝非部署时免费可得的routing信号。没有根据高交互test分组训练或选择新的策略。

## 7. Reviewer WCQn response

We retained the parameter-matched learned gating baseline and conducted a validation-only auxiliary-loss ablation on the unchanged R2 architecture. All new runs use the same 4,096/64/448 split, joint-subset teachers, 60-epoch schedule and three seeds. The preselected Joint+CFS+Alpha model (λ=0.01) achieves hard-gating KL 0.10517/0.02480/0.00795 at K=3/6/9, versus 0.11154/0.02699/0.00808 for learned gating and 0.10562/0.02413/0.00833 for Joint-only R2. Relative to Joint-only, its K=9 paired difference is −0.000373 [−0.000730,−0.000072], while K=3/6 improvements are not resolved. The CFS-only auxiliary does not show a resolved gain over Joint-only. Thus we report a modest, budget-dependent benefit of the selected auxiliary bundle, not a universal advantage attributable solely to CFS. Extra pairwise supervision and the larger validation search budget are explicitly accounted for. All intervals are unadjusted 95% input-bootstrap intervals conditional on the three seeds.

## 8. Reviewer UWwP response

We measured 38,016 fixed-background second differences of actual subset KL on 64 tuning and 128 disjoint held-out inputs, using only frozen last-block interventions. An additive model has held-out-mask RMSE 0.1072/0.0821/0.0356 at K=3/6/9, and directly measured interactions confirm non-additivity. In disjoint-input ridge prediction controlling Taylor importance, contribution cosine, deletion effects, pair identity and background masks, CFS improves K=9 interaction prediction: the MSE reductions are 0.00002669 [0.00000903,0.00004492] for signed interactions and 0.00005102 [0.00002448,0.00007739] for their magnitude (exact values in the table). Gains at K=3/6 remain unresolved. Interaction complexity correlates with additive gating error, but its association with the selected CFS-regularized router's relative advantage remains unresolved. These results establish a restricted analytical and predictive use for CFS; they do not establish causal mediation, universally superior routing, or physical inference acceleration.

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
