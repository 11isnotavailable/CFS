# CFS-conditioned latent interaction residual feasibility

日期：2026-11-15。冻结来源 a7c87f5。**验证选择：A3；是否通过预设提升条件：False。**

这轮实现没有取得预期Routing增益。独立确认集上单位强度CFS残差的K3/K6/K9均值为0.116239/0.031450/0.009817，原A3为0.105810/0.027364/0.009669。K3/K6相对A3的配对差区间为正；K9差异未分辨。原R2在该确认集K9为0.006436，仍较强。A3在旧测试三个预算均优于Gating的点估计现象，没有在新确认集K9上保持；不能把旧测试均值结论扩大为普遍优势。

## 1. 实际执行范围与科学边界

仅做方向 A 的离线可行性实验；A3、原 R2、ViT-B Backbone 全部冻结。三种轻量残差、三个固定种子17/23/41，各20epochs，lr3e-4，batch64；不搜索网络、Loss权重、学习率或时长。每种新增 14,945 个参数，完整A3+残差为 840,036 个，比825,091增加 1.811%。新增模型不是原825K参数匹配主实验。

已有38,016个真实四元组来自旧tune/test，**没有拿来训练**。本轮只复用4,096个训练输入的Joint-KL标签。每输入/预算48个唯一Mask固定拆为36拟合+12诊断。冻结A3此前见过这些标签，因此这里只能称为“对残差预测器留出的Mask”，不是完整管线完全未见的数据。旧测试448张已被多轮查看，仅作探索。新增256张确认图在拟合前锁定，排除旧4,608个输入的路径和SHA256；来自同一Imagenette验证总体，不代表跨域外推。

该实现是**保留集合条件化的潜在成对残差**：训练目标是联合KL预测误差，成对输出未被真实二阶差分唯一识别，不能称作已学得每一对Head的物理/因果交互。没有把任意背景下的四元组相加。结果不改变原始CFS定义，也不证明独立补偿近似是误差的唯一原因。

## 2. 数学形式与公平对照

原结构效用U仍调用项目的裁剪和聚合实现。每预算仅在36个拟合Mask上拟合D0=a-bU，b>=1e-8，保持原排序。预测D_new=D0+gamma_K*sd_K*r(x,R)。CFS版对删除对使用共享小MLP，输入包含同一个保留集合R、冻结输入表示、重要性、Head身份以及八个有向CFS关系特征，取成对输出平均。它等价于带gamma_K/P系数的成对和，不假定可物理相加。分数可为负，记录原值，未为了好看而裁剪KL预测。

Plain是相同参数量的普通集合MLP；Shuffled保留网络和每个输入图的数值分布，但按固定随机置换破坏Head对齐。所有方案使用相同冻结编码、原效用、数据、初始化种子、优化预算、校准规则。Plain仍通过冻结A3编码间接包含CFS训练信息；对照检验显式图结构的增量，而不是全部CFS信息的存在与否。成对网络计算量高于Plain，参数相同不代表延迟相同。

## 3. 验证锁定与补充诊断

各方法按三个seed平均tune标准化MSE选一个共同epoch，再对每预算从0/.25/.5/1中按tune选择真实候选KL最小的gamma，平局取较小gamma。验证仅有48个候选/预算，实际排序枚举220/924/220个；保留这个代理偏差限制。锁定结果如下，测试/确认不参与：

```json
{
  "variants": {
    "plain": {
      "epoch": 20,
      "gamma": [
        0.0,
        0.0,
        0.0
      ],
      "tune_kl": [
        0.09508334845304489,
        0.01863328367471695,
        0.0056198057718575
      ],
      "base_tune_kl": [
        0.09508334845304489,
        0.01863328367471695,
        0.0056198057718575
      ]
    },
    "cfs": {
      "epoch": 19,
      "gamma": [
        0.0,
        0.0,
        0.0
      ],
      "tune_kl": [
        0.09508334845304489,
        0.01863328367471695,
        0.0056198057718575
      ],
      "base_tune_kl": [
        0.09508334845304489,
        0.01863328367471695,
        0.0056198057718575
      ]
    },
    "shuffled": {
      "epoch": 19,
      "gamma": [
        0.0,
        0.0,
        0.0
      ],
      "tune_kl": [
        0.09508334845304489,
        0.01863328367471695,
        0.0056198057718575
      ],
      "base_tune_kl": [
        0.09508334845304489,
        0.01863328367471695,
        0.0056198057718575
      ]
    }
  },
  "primary": "A3",
  "eligible": false,
  "test_used": false,
  "confirmation_used": false,
  "selection_metric": "Tune only; identical policy for every variant"
}
```

全部方案选择了gamma=0，均回退原A3。看到此验证结果后、**读取任何本轮测试/确认输出之前**，新增固定gamma=1的统一诊断（`*_raw`），见独立补充协议。它不参与选择，三个方法/预算全部报告，不能根据其测试结果升级为主模型。没有追加训练。

## 4. 真实硬门控KL：Mean [95% CI]

所有方法使用同一ViT-B Checkpoint、预处理、最后一层Head门控和KL(p_dense||p_routed)。原三个基线在旧测试上与历史结果核对一致，dense logits逐元素一致。下表的plain/cfs/shuffled是锁定策略，*_raw是单位强度诊断。

### test

| method       | K=3                              | K=6                              | K=9                              |
|:-------------|:---------------------------------|:---------------------------------|:---------------------------------|
| A3           | 0.1051673 [0.0913031, 0.1197498] | 0.0248006 [0.0207983, 0.0291227] | 0.0079536 [0.0064643, 0.0095895] |
| R2           | 0.1190073 [0.1037138, 0.1350883] | 0.0281352 [0.0240766, 0.0324784] | 0.0069980 [0.0056469, 0.0084921] |
| Gating       | 0.1115365 [0.0975148, 0.1268990] | 0.0269924 [0.0226395, 0.0314269] | 0.0080800 [0.0065228, 0.0097850] |
| plain        | 0.1051673 [0.0913031, 0.1197498] | 0.0248006 [0.0207983, 0.0291227] | 0.0079536 [0.0064643, 0.0095895] |
| cfs          | 0.1051673 [0.0913031, 0.1197498] | 0.0248006 [0.0207983, 0.0291227] | 0.0079536 [0.0064643, 0.0095895] |
| shuffled     | 0.1051673 [0.0913031, 0.1197498] | 0.0248006 [0.0207983, 0.0291227] | 0.0079536 [0.0064643, 0.0095895] |
| plain_raw    | 0.1204830 [0.1061733, 0.1355681] | 0.0279335 [0.0237422, 0.0324387] | 0.0096679 [0.0079236, 0.0114721] |
| cfs_raw      | 0.1190147 [0.1048661, 0.1340209] | 0.0269826 [0.0230905, 0.0309826] | 0.0090205 [0.0075677, 0.0106355] |
| shuffled_raw | 0.1190082 [0.1044608, 0.1338769] | 0.0265428 [0.0227607, 0.0304560] | 0.0092237 [0.0077201, 0.0108821] |
### confirmation

| method       | K=3                              | K=6                              | K=9                              |
|:-------------|:---------------------------------|:---------------------------------|:---------------------------------|
| A3           | 0.1058096 [0.0876483, 0.1248914] | 0.0273637 [0.0212701, 0.0342504] | 0.0096689 [0.0066536, 0.0131937] |
| R2           | 0.1148799 [0.0963004, 0.1349965] | 0.0308723 [0.0245992, 0.0377886] | 0.0064363 [0.0052264, 0.0077970] |
| Gating       | 0.1071228 [0.0892090, 0.1268186] | 0.0290018 [0.0227088, 0.0361678] | 0.0089926 [0.0065962, 0.0117189] |
| plain        | 0.1058096 [0.0876483, 0.1248914] | 0.0273637 [0.0212701, 0.0342504] | 0.0096689 [0.0066536, 0.0131937] |
| cfs          | 0.1058096 [0.0876483, 0.1248914] | 0.0273637 [0.0212701, 0.0342504] | 0.0096689 [0.0066536, 0.0131937] |
| shuffled     | 0.1058096 [0.0876483, 0.1248914] | 0.0273637 [0.0212701, 0.0342504] | 0.0096689 [0.0066536, 0.0131937] |
| plain_raw    | 0.1165034 [0.0984520, 0.1364698] | 0.0302756 [0.0243890, 0.0370338] | 0.0101861 [0.0074585, 0.0134668] |
| cfs_raw      | 0.1162390 [0.0979264, 0.1357035] | 0.0314501 [0.0243182, 0.0399423] | 0.0098169 [0.0074187, 0.0127757] |
| shuffled_raw | 0.1182575 [0.0991726, 0.1384477] | 0.0307314 [0.0239155, 0.0389242] | 0.0098361 [0.0074678, 0.0128476] |

### 关键配对差（CFS单位强度减参考；负数有利于CFS）

| cohort       |   keep | reference    |       mean |     ci_low |    ci_high |
|:-------------|-------:|:-------------|-----------:|-----------:|-----------:|
| confirmation |      3 | A3           |  0.0104294 |  0.0042786 |  0.0169094 |
| confirmation |      3 | plain_raw    | -0.0002643 | -0.0036116 |  0.0031463 |
| confirmation |      3 | shuffled_raw | -0.0020184 | -0.0042012 | -0.0003643 |
| confirmation |      6 | A3           |  0.0040864 |  0.0008008 |  0.0081090 |
| confirmation |      6 | plain_raw    |  0.0011745 | -0.0019195 |  0.0054873 |
| confirmation |      6 | shuffled_raw |  0.0007187 |  0.0000379 |  0.0014695 |
| confirmation |      9 | A3           |  0.0001480 | -0.0021560 |  0.0018543 |
| confirmation |      9 | plain_raw    | -0.0003692 | -0.0016166 |  0.0005684 |
| confirmation |      9 | shuffled_raw | -0.0000192 | -0.0006150 |  0.0005334 |
| test         |      3 | A3           |  0.0138474 |  0.0100319 |  0.0178781 |
| test         |      3 | plain_raw    | -0.0014683 | -0.0042892 |  0.0016390 |
| test         |      3 | shuffled_raw |  0.0000065 | -0.0012389 |  0.0012476 |
| test         |      6 | A3           |  0.0021819 |  0.0005378 |  0.0037902 |
| test         |      6 | plain_raw    | -0.0009510 | -0.0031423 |  0.0012408 |
| test         |      6 | shuffled_raw |  0.0004397 | -0.0002557 |  0.0012946 |
| test         |      9 | A3           |  0.0010668 |  0.0004849 |  0.0016298 |
| test         |      9 | plain_raw    | -0.0006475 | -0.0014893 |  0.0000512 |
| test         |      9 | shuffled_raw | -0.0002033 | -0.0007046 |  0.0002520 |

## 5. 留出Mask的预测与排序

以下为残差留出Mask的MSE差。完整MAE、选择regret、基线误差和所有配对区间保存在prediction_summary/paired。降低绝对KL预测误差可能主要改善输入难度/整体尺度估计，**不能直接推出同一输入内子集排序改善**。关键增量应比较CFS与Plain/Shuffled，而非只比较校准后的基础模型。

|   keep | contrast            |       mean |     ci_low |    ci_high |
|-------:|:--------------------|-----------:|-----------:|-----------:|
|      3 | cfs_minus_plain     |  0.0137984 |  0.0122044 |  0.0155358 |
|      3 | cfs_minus_shuffled  | -0.0043045 | -0.0047065 | -0.0039013 |
|      3 | plain_minus_base    | -0.1663932 | -0.1761049 | -0.1573658 |
|      3 | cfs_minus_base      | -0.1525948 | -0.1609566 | -0.1446854 |
|      3 | shuffled_minus_base | -0.1482903 | -0.1562556 | -0.1407075 |
|      6 | cfs_minus_plain     |  0.0014549 |  0.0012965 |  0.0016130 |
|      6 | cfs_minus_shuffled  | -0.0002269 | -0.0002558 | -0.0002002 |
|      6 | plain_minus_base    | -0.0094110 | -0.0100460 | -0.0087922 |
|      6 | cfs_minus_base      | -0.0079560 | -0.0084668 | -0.0074465 |
|      6 | shuffled_minus_base | -0.0077291 | -0.0082177 | -0.0072451 |
|      9 | cfs_minus_plain     |  0.0002585 |  0.0002260 |  0.0002921 |
|      9 | cfs_minus_shuffled  | -0.0000594 | -0.0000660 | -0.0000529 |
|      9 | plain_minus_base    | -0.0017093 | -0.0018461 | -0.0015811 |
|      9 | cfs_minus_base      | -0.0014509 | -0.0015582 | -0.0013467 |
|      9 | shuffled_minus_base | -0.0013914 | -0.0014962 | -0.0012919 |

K9的关键反例：CFS残差相对校准A3的留出Mask MSE约下降0.001451，但选中Mask的regret约增加0.000612。同时CFS的MSE比普通MLP高约0.000258、比Shuffled低约0.000059。因而虽然真实图对齐相对打乱图对预测有一定增量，却没有形成超越普通残差、再改善原Router的完整证据链。确认集K9上CFS与Plain/Shuffled的配对KL差区间均跨零，不据此宣布二者等效。

## 6. 不确定性、成本与结论限制

独立输入为重采样单位，2,000次Bootstrap，seed42；先在每输入内平均三个训练seed，再重采样。旧测试和确认队列分开分析。per_seed表给每个种子均值/区间，summary给输入SD与seed均值SD。区间条件于这三个种子，未作多重比较校正。没有把Mask、Head Pair或种子当作额外独立输入。

九次训练记录总耗时 55.62 秒（不含特征缓存、数据加载和模型评估）；推理各阶段实测耗时在timings.csv。离线候选打分循环额外计算了冻结encoder，并且*_raw与锁定策略共享计算；这些计时不是优化后的端到端部署延迟。本实验没有推理加速声明。

在预设验证选择下没有得到可以替换A3的残差方法。若单位强度测试/确认的某个点估计更好，也不能反过来选择系数。是否存在CFS结构的增量应读上面的配对对照和独立确认，而不能把非零残差、较低MSE或个别预算均值作为充分证据。当前范围不支持继续叠加方向B，更不能据此把潜在分解称为已验证的高阶因果交互算法。

## 7. Rebuttal wording

We evaluated a bounded, frozen-router residual extension using only existing training joint-deletion labels. Parameter-matched plain, CFS-conditioned pair-decomposed, and shuffled-CFS residuals were trained with identical budgets. The pair terms are latent joint-loss corrections, not identified causal interaction effects. Validation selected zero residual strength for all three methods and budgets, so the primary router remains A3. We report fixed unit-strength diagnostics on both the previously inspected test set and 256 additional confirmation images, without reselecting models. This study distinguishes improvement in absolute joint-KL prediction from improvement in subset selection; it does not establish a deployable higher-order routing advantage or inference acceleration.

## 8. 文件与复现

新增src/higher_order_residual.py、prepare_residual_confirmation.py、train_higher_order_residual.py、evaluate_higher_order_residual.py、analyze_higher_order_residual.py、verify_higher_order_residual.py和对应测试。旧Attention/MLP/Scaling实现不变。主协议与pre-test amendment同时保留。完整权重、逐Mask预测、逐输入KL、确认图片哈希、验证锁和Bootstrap抽样均归档。完整本地测试套件59项通过。

在新实验目录放置scripts/src，source链接到上轮cfs_interaction，baseline链接到learned_gating；confirmation复制本轮锁定的图片与confirmation.npz，勿带features.npz。使用原实验Python环境和base.pth：

```bash
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4
python scripts/train_higher_order_residual.py --root . --stage all
python scripts/evaluate_higher_order_residual.py --root . --stage all
python scripts/analyze_higher_order_residual.py --root .
python scripts/verify_higher_order_residual.py --root .
```

输出目录必须是新的；原始数据和结果不覆盖。分模型拟合可用--stage train --kind cfs --seed 17；所有九个拟合完成后运行--stage lock和--stage diagnostic。图1为实际重排序的配对差，图2分开呈现预测MSE与子集选择regret。
