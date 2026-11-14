# CFS Matched-N / Fixed-K 最终结论

七模型于 2026-11-15 完成，未训练任何模型。四个 Qwen 规模统一使用新冻结的 32 个 FineWeb-Edu contexts；三个 ViT 使用同一组 512 张 ImageNet-1K 验证图像。Qwen 是新输入复现实验，不能声称逐输入复现原论文。

## Q1：固定候选数后的趋势

Qwen 在 N=2、4、8、11 下均值均随 1.5B→3B→7B→14B 单调增加；12 个相邻规模/配置对照的配对 95% CI 均大于零。

ViT 在 N=2、4、5 下均值也单调增加，但 Small→Base 的三个差异 CI 均包含零，Base→Large 的三个差异 CI 均大于零。因此保留系列总体均值方向，但应弱化“每次扩大模型均明确提高替代性”的说法。

| 模型 | Raw Oracle | Matched-2 | Matched-2 95% CI |
|---|---:|---:|---|
| Qwen-1.5B | 0.267861 | 0.160793 | [0.149812, 0.172324] |
| Qwen-3B | 0.327258 | 0.203103 | [0.187849, 0.218944] |
| Qwen-7B | 0.376968 | 0.225949 | [0.209176, 0.244129] |
| Qwen-14B | 0.440904 | 0.271380 | [0.246692, 0.298381] |
| ViT-Small | 0.408621 | 0.285045 | [0.273517, 0.296977] |
| ViT-Base | 0.497663 | 0.289034 | [0.277943, 0.300356] |
| ViT-Large | 0.603245 | 0.373882 | [0.360237, 0.387634] |

## Q2：固定子图规模后的趋势

Qwen 的 K=4、8、12 与 ViT 的 K=4、6 均保持 Cover/K、Rank/K 均值随规模下降。所有相邻 Rank 差异的配对 CI 均小于零。Cover 的以下三个相邻差异 CI 包含零，不能声称每一步下降都明确：

| 对照 | 统计量 | 差异（较大模型减较小模型） | 95% CI |
|---|---|---:|---|
| Qwen 3B→7B | Cover/8 | -0.023115 | [-0.049648, 0.001621] |
| Qwen 3B→7B | Cover/12 | -0.024920 | [-0.053737, 0.002985] |
| ViT Small→Base | Cover/6 | -0.015288 | [-0.032784, 0.002648] |

## Q3：保留与弱化的结论

保留：本次输入与模型系列内，固定候选数后的 Oracle 均值上升、固定节点数后的 Cover/Rank 均值下降。这些已测趋势不能仅用候选 head 数量增多解释。

弱化：不能将未控制候选数的变化幅度全部归为功能结构变化，不能声称每个相邻规模对照均有明确差异，也不能推出因果 scaling、任务效用提升或可执行联合压缩率。CI 是未校正多重比较的逐项区间。

有效 source 比例为 Qwen 四模型均 100%，ViT Small/Base/Large 分别约 99.64%/96.74%/74.44%。Fixed-K 保持原过滤规则，但没有控制有效 source 比例，ViT-L 图统计的变化仍需带此限制解释。

## 统计、验证和交付

- Matched-N 为精确组合期望；Fixed-K 每输入/层 500 次均匀子集抽样，seed 42；K=H 只计算一次。
- 2000 次输入级 bootstrap，保留同一输入内的头和层依赖；配对差异使用同一冻结输入。Oracle 保留原有效 source 权重及层聚合。
- 七模型同矩阵 Matched-2 与原实现最大绝对误差 4.44e-16；K=H、原始文件 SHA256 一致性验证通过。
- 最大子图 Monte Carlo SE 为 0.000596；250→500 次抽样的最大聚合均值变化为 0.000589。子图抽样误差单列，不能把子图当作独立样本。
- 代码全套 32 项测试通过；最终图表已人工查看。来源版本、环境和逐输入结果保存在完整本地归档。

仓库内完整结果位于 `results/matched_scaling_rebuttal_20261115/`：82 行均值、CI、输入 SD 和子图 MC SE 为 `all_results.csv/json/tex`；相邻差异为 `paired_contrasts.csv`；原论文汇总对照为 `historical_comparison.csv`；抽样稳定性为 `sampling_stability_summary.csv`。两幅图均提供 PDF/PNG，`EXPERIMENT_REPORT.md` 含全部数值表。方法及复现命令见仓库 `docs/MATCHED_SCALING_REBUTTAL.md`。

本地完整矩阵、bootstrap replicate 与冻结输入：`D:/tries2/outputs/matched_scaling_final_20261115/`。压缩归档为 `rebuttal_results_final_20261115.tar.gz`，SHA256：`f7860486464b810d273e2064b0758becc59c2b292f4af5e68e2c9af236c02625`。

## Reviewer nLkz response draft

We agree that Raw Oracle can be affected by the number of candidate heads. We therefore evaluated exact expected best-of-N recovery and fixed-size induced subgraphs, retaining the original CFS, validity, Cover, and Rank definitions. The rerun uses 32 newly frozen shared FineWeb-Edu contexts across four Qwen2.5 sizes and 512 shared ImageNet-1K validation images across three ViT sizes. We use 2,000 input-level paired bootstrap replicates and 500 subgraph draws per input/layer, except when K=H.

For Qwen, Matched-N increases at every measured size transition for N=2,4,8,11, with all adjacent pointwise 95% difference intervals above zero. For ViT, the mean ordering remains increasing for N=2,4,5, but Small-to-Base intervals include zero; Base-to-Large intervals are positive. Fixed-K Cover/K and Rank/K decrease in mean across both families. Three Cover contrasts include zero: Qwen 3B-to-7B at K=8/12 and ViT Small-to-Base at K=6. All adjacent Rank difference intervals are negative.

Thus the observed mean trends persist under candidate-count and subgraph-size controls, while some stepwise claims require qualification. These intervals are not multiplicity-adjusted. The controls do not isolate model scale from architecture, pretraining, or validity-rate differences, and do not establish causal scaling or executable joint compression.
