"""Render the predeclared learned-gating experiment's numerical report."""
import argparse
import json
from pathlib import Path
import pandas as pd


def markdown(frame):
    columns=list(frame.columns)
    lines=['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;analysis=root/'analysis_final'
    summary=pd.read_csv(analysis/'summary.csv');paired=pd.read_csv(analysis/'paired_bootstrap.csv')
    historical=pd.read_csv(root/'evaluation_v2/historical_comparison.csv')
    def table(comp,metric):
        rows=[]
        for method in ('Taylor','Gating','R2','R2_joint_only'):
            row={'Method':method}
            for k in (3,6,9):
                r=summary.query('method==@method and keep==@k and compensation==@comp and metric==@metric').iloc[0]
                scale=100 if metric=='correct' else 1
                row[f'K={k}']=f'{r["mean"]*scale:.5f} [{r.ci_low*scale:.5f}, {r.ci_high*scale:.5f}]'
            rows.append(row)
        return markdown(pd.DataFrame(rows))
    compact=paired[paired.metric=='kl'][['contrast','keep','mean','ci_low','ci_high']].copy()
    for col in ('mean','ci_low','ci_high'):compact[col]=compact[col].map(lambda x:f'{x:.6f}')
    cost=pd.read_csv(analysis/'router_costs.csv')
    seedspread=summary[(summary.metric=='kl')&(summary.compensation=='hard')][['method','keep','mean','sd_seed_means']].copy()
    for col in ('mean','sd_seed_means'):seedspread[col]=seedspread[col].map(lambda x:f'{x:.6f}')
    text=f'''# Learned Dynamic Gating：Reviewer WCQn 对照实验

完成日期：2026-11-15。结论：**普通 Learned Gating 是强基线；原始 CFS R2 并未在所有预算下具有独立的选头优势。** 共享硬门控下 Gating 在 K=3 的 KL 更低，K=6 未分辨出差异，K=9 原 R2 更低。共享 R2 补偿后，K=6、9 的区间支持原 R2 更低的 KL，但补偿器自身来自 R2 的联合训练。不能把这些结果全部归因于 pairwise CFS 标签。

## 1. 审计与冻结协议

使用原 Routing 的 Imagenette2-160 划分 4096/64/448，ViT-B AugReg 原始 `base.pth`，最后 Attention block 11，12 heads，K=3/6/9，种子17/23/41。与前面 ImageNet Scaling/MLP 实验的数据集不同。原始权重 SHA256 为 `683c6d82c1e832981a5fdce9ae66be4f02542b4bd7b7fc822b08f53c005de8df`。

输入复用原始 CLS + patch mean + population variance 缓存，FP16 存储边界后转 FP32，使用同一训练集 mean/std。代码实际从 Attention wrapper 输入提取，即本 ViT 的 norm1 之后；不是任意新定义的 residual 特征。预处理224、bicubic、crop_pct=0.9、mean/std=0.5。使用原 ImageNet 1000 类标签评价 top-1。

R1 为 `2304→256→256` 编码器 + 32维 head embedding + `384→256→128` pair MLP，输出 pair score、importance 与 alpha。R2 保留 R1 的 representative/listwise、score、deletion importance、alpha、structured subset 损失，并加入权重0.5的 joint-set ranking；原始825091参数与三种子权重全部保留。

Gating 为 `2304→304→384→12`、GELU、dropout0.1，共822460参数，比原 R2 少0.319%。统一模型支持全部 K；子集效用为 head score 求和，部署使用 Top-K，恰好保留 K heads。不同 K 的集合因此嵌套，而 R2 的最优子集不必嵌套。

两者复用每输入每 K 的48个候选硬门控 KL，但**原 R2 额外使用 pairwise/deletion/alpha 监督**。新增 R2_joint_only：原结构重新初始化，只用与 Gating 完全相同的 Joint-KL 损失；alpha输出129参数不参与训练，共824962优化参数。此对照单独命名，不替换原 R2。

所有新增训练：AdamW lr=3e-4、weight_decay=1e-4、batch64、60 epochs，每种子3840次更新；无调参搜索、无 test early stopping。每输入每预算内标准化原 KL，SD floor=1e-6，softmax(-z/0.5)为目标，预测效用温度0.05，listwise损失乘0.5。用64个 tune 输入的候选集 KL regret 选择 checkpoint。训练脚本不加载 test；导出的 test 包不包含 Teacher KL。候选池含原 CFS/R1 anchors，所有方法共享此固定池，但不能视为与 CFS 完全无关的采样分布。

## 2. 主要对照：共享硬门控

同一 dense reference、同一 K、保留 head 系数1、删除系数0。下表为三种子平均、448个独立输入、2000次输入 Bootstrap 的 Mean [95% CI]，KL越低越好。

{table('hard','kl')}

种子均值的样本标准差（不把种子当作额外输入）：

{markdown(seedspread)}

## 3. 共享 learned compensation 与完整 pipeline

对所有 selector 的 mask 应用同一种子的 frozen 原 R2 score/alpha 与同一 `compose_scales`：保留head初始1，累加 alpha−1，裁剪[0,6]。这是共享**应用**，补偿器训练仍与 R2 相关。Gating加此补偿不是独立的 CFS-free 部署系统：还需要原R2，总参数为1647551（原R2自身仅825091）。硬门控是无此偏向的主比较。

{table('shared_r2_merge','kl')}

原论文保守主部署是 hard routing；learned merge 是补偿归因变体。本报告同时保留原完整 learned-merge pipeline 与独立 Gating hard 的比较，见 `full_pipeline_table.csv/.tex`。

Gating-hard − R2-merge 的 KL 差为0.026397、0.003765、0.001757。它在算术上分别等于：

| K | Gating-hard − R2-hard（选择差） | R2-hard − R2-merge（R2补偿收益） | 合计 |
| --- | --- | --- | --- |
| 3 | -0.007471 | 0.033868 | 0.026397 |
| 6 | -0.001143 | 0.004907 | 0.003765 |
| 9 | 0.001082 | 0.000675 | 0.001757 |

这只是给定模型与mask下的加法分解，不是独立因果贡献或可跨selector迁移的补偿收益。尤其K=3的系统优势来自补偿，原R2的纯选择反而更差。Gating共享补偿在K=6、9的平均收益接近零，不能忽略补偿与选择的相互作用。

## 4. Accuracy

Dense accuracy为81.9196%，输入Bootstrap 95% CI [78.5714%,85.2679%]。下表单位为百分比。

共享硬门控：

{table('hard','correct')}

共享 R2 补偿：

{table('shared_r2_merge','correct')}

Gating-hard在K=3比R2-hard高1.8601个百分点，配对区间[0.8185,3.0506]个百分点；其他预算的此准确率差异未分辨。完整R2-merge与Gating-hard的准确率差异均未分辨。KL改善不能直接等同于任务准确率改善。

## 5. 配对统计

差异方向按名称 left−right；Gating−R2 的KL负值有利于Gating。先对同一输入的三个种子取均值，再用同一组输入索引配对重采样。CI条件于这三个训练种子，不覆盖训练随机性的完整总体；每种子统计另存。以下为未作多重比较校正的逐项95%区间，不作全族显著性声明。

{markdown(compact)}

文件 `summary.csv/json` 保留输入标准差、种子均值标准差及CI；`per_seed_summary.csv` 与 `per_seed_paired_bootstrap.csv` 分别保留每种子均值/CI及配对差；原始Bootstrap draws和统计复制值也保存。

## 6. 历史一致性与当前前向特征

三个原R2在冻结输入特征上的全部选头mask复现一致（0 mismatches）；重新计算pair score的最大绝对差小于2e-6。原始模型与数据未改变。Dense accuracy与历史一致。全部历史对照中平均KL漂移绝对值最大为 {historical.mean_delta.abs().max():.8f}；最大逐输入漂移 {historical.max_abs_delta.max():.6f}，部分接近分类边界的配置有1/448准确率差异。历史值和当前值分列保存在 `historical_comparison.csv`，不混用。

当前GPU/库环境的FP16前向特征与旧缓存并非逐bit相同：最大差0.21875，平均绝对差0.000699，训练SD归一化后RMSE=0.002061；同环境batch1/16也可出现差异。未证明漂移的唯一来源，故不声称精确在线特征复现。

为检验部署影响，使用当前dense前向提取的同一批特征重新计算所有learned selector及R2补偿，保持权重、训练、测试输入与统计完全不变。完整逐输入结果见 `evaluation_live/`，完整统计见 `analysis_live_final/`。Gating−R2硬门控的KL差依次为−0.007634、−0.001074、0.001124，区间分别为[−0.015284,−0.000496]、[−0.003598,0.001505]、[0.000132,0.002264]；主要判断不变。此次敏感性检查未用于调参或选择结果。

## 7. 复杂度与有效范围

{markdown(cost.round(6))}

FLOPs只统计Linear乘加（2 FLOP/MAC），不包括GELU、softmax、Top-K、subset reductions；R2的pair结构使前向线性运算约40.02MFLOP，Gating约1.64MFLOP。参数匹配不等于计算量匹配。R2分别枚举220/924/220个子集，Gating直接Top-K。R2_joint_only保留相同结构/搜索开销。GPU时间为RTX PRO6000上batch1、FP32、50 warmup、200次GPU events的孤立router forward+一个K的选择，使用随机权重；不包含summary、backbone或补偿，不能作为整模型加速倍数。Taylor没有router参数，但原Taylor mask依赖标签/梯度，不能称为同成本的forward-only部署。

## 8. 回答研究问题

**Q1：普通Gating能否达到原R2？** 在本实验中可以在部分预算达到或超过：K=3更低KL且更高accuracy，K=6的KL差异未分辨，K=9原R2更低KL。不能声称全部预算等价，因为没有预设等效性界限。

**Q2：共享补偿下CFS selection是否有独立优势？** 无学习补偿的硬门控下没有一致优势；共享原R2补偿下K=6、9区间支持原R2更低KL，K=3未分辨。但该补偿器与R2共同训练，因此这不是对任意补偿器都成立的选择优势。

**Q3：完整pipeline差异多少来自补偿？** 上述分解明确：K=3、6的原R2硬选择无优势，补偿使最终系统KL更低。K=9两部分均有正贡献。贡献依赖具体选择与补偿组合，准确率优势未统一成立。

**Q4：pairwise functional structure是否提供额外信息？** 相同Joint-KL监督下，R2结构比加性Gating在K=3、6更低KL，配对差0.005914 [0.001459,0.011065]、0.002859 [0.000727,0.005201]，K=9未分辨。这支持结构化效用/非嵌套子集搜索在本设置下的价值；并不能将收益唯一归因于pairwise CFS标签，因为该对照未使用这些标签，且其前向运算与搜索成本更高。原pairwise监督相对于普通Gating的普遍独立优势证据不足，应弱化此类表述。

## 9. Reviewer WCQn response

We added a parameter-matched learned head-gating baseline (822,460 vs. 825,091 parameters), using the same pre-attention summaries, 4,096/64/448 train/tune/test split, cached joint-subset KL supervision, 60-epoch schedule, and three seeds. Under shared hard gating, its KL at K=3/6/9 is 0.11154/0.02699/0.00808, compared with 0.11901/0.02814/0.00700 for the original R2. Paired input-bootstrap differences (gating minus R2) are −0.00747 [−0.01512, −0.00033], −0.00114 [−0.00368, 0.00142], and 0.00108 [0.00008, 0.00221]. Thus R2 does not uniformly outperform learned gating in selection alone. Applying the same R2 compensator yields lower R2 KL at K=6 and K=9, but this compensator was trained with R2. Because original R2 also uses pairwise supervision, we additionally trained its structure using only the identical joint-KL labels: it outperforms additive gating at K=3 and K=6, with the K=9 difference unresolved. This supports a budget-dependent benefit of structured subset selection, rather than establishing an irreducible benefit of pairwise CFS labels. Intervals are unadjusted 95% paired input-bootstrap intervals conditional on the three seeds. We make no end-to-end speedup claim.

## 10. 交付与复现

新增文件：

| 文件 | 用途 |
| --- | --- |
| src/learned_head_gating.py | 加性Gating、Top-K、与原R2一致的Joint-KL目标 |
| scripts/prepare_learned_gating.py | 从原缓存导出数据、划分校验、portable R2权重、图像及hash |
| scripts/train_learned_gating.py | Gating与Joint-only R2训练；不读测试集 |
| scripts/evaluate_learned_gating.py | 统一Dense、hard/shared compensation、历史核查、逐输入输出 |
| scripts/audit_gating_live_features.py | 当前前向特征漂移与部署敏感性数据 |
| scripts/analyze_learned_gating.py | 输入Bootstrap、种子统计、CSV/JSON/LaTeX/PDF/PNG |
| scripts/benchmark_learned_gating.py | 单独router线性运算和GPU时间 |
| scripts/report_learned_gating.py | 从结果生成此报告 |
| scripts/run_learned_gating.sh | 六个固定配置的完整训练 |
| tests/test_learned_head_gating.py | 参数、Top-K、目标、划分、种子、补偿与Bootstrap检查 |
| docs/LEARNED_GATING_PROTOCOL.md | 测试前冻结的方案与监督/补偿边界 |

48个项目测试通过。先前attention代码与数据均未修改。首次评估发现原 `clear_interventions` 会关闭capture flag，已在dense forward前重新开启；失败日志保留，没有产生可用测试指标。小样本训练和同种子重复运行另存。P1可选Direct Subset Predictor未运行，避免增加未预注册的模型选择。

完整归档包含六个新checkpoint、原R2可移植checkpoint、训练日志、原标签/图像导出、每输入CSV/JSON、mask/scale/dense logits、Bootstrap与图表。ViT大权重不重复归档，使用上面的原本地文件及SHA验证。

在仓库根目录导出缓存（输出须是新目录）：

```powershell
python scripts/prepare_learned_gating.py --workspace D:/tries2 --output D:/tries2/outputs/learned_gating_inputs_new
python -m pytest -q -p no:cacheprovider
```

Linux复现，使用归档的 `learned_gating_inputs_20261115`、原 `base.pth`，在包含scripts/src的实验目录执行（新结果目录名称须未存在）：

```bash
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
for kind in gating joint_only_r2; do
  for seed in 17 23 41; do
    python scripts/train_learned_gating.py --data learned_gating_inputs_20261115 --output new_runs/${{kind}}_seed${{seed}} --kind "$kind" --seed "$seed"
  done
done
python scripts/evaluate_learned_gating.py --data learned_gating_inputs_20261115 --runs new_runs --checkpoint base.pth --output new_evaluation
python scripts/analyze_learned_gating.py --evaluation new_evaluation --output new_analysis --bootstrap 2000
python scripts/benchmark_learned_gating.py --output new_analysis
python scripts/audit_gating_live_features.py --data learned_gating_inputs_20261115 --checkpoint base.pth --output new_live_features
python scripts/evaluate_learned_gating.py --data learned_gating_inputs_20261115 --runs new_runs --checkpoint base.pth --output new_live_evaluation --feature-file new_live_features/live_features.npz
python scripts/analyze_learned_gating.py --evaluation new_live_evaluation --output new_live_analysis --bootstrap 2000
```

主图：`figure1_dynamic_routing.pdf/png`、`figure2_paired_difference.pdf/png`。`
'''
    # Avoid leaving a dangling markup character in the report footer.
    text=text.rstrip('`\n')+'\n'
    (analysis/'REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
