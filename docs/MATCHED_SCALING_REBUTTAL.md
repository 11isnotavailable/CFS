# Matched-N / Fixed-K 补充实验：方法与复现

本文件记录 2026-11-14 启动的补充实验，七模型已于 2026-11-15 完成。
完整结果、限制及 Reviewer nLkz 回复草稿见 [最终交付](../results/matched_scaling_rebuttal_20261115/README.md)。
用户已授权重新下载公开权重、重新采集缺失矩阵，并为四个 Qwen 规模统一冻结新的 FineWeb-Edu 输入。
不训练模型、不改原论文正文、不覆盖历史实验。

## 代码库与原始数据检查

- Qwen 收集：`scripts/collect_qwen_cfs.py`；ViT 收集：`scripts/collect_predictor_dataset.py`。
- 每个 shard 保存 `single_drop[input, layer, H]`、`substitution[input, layer, H, H]` 和 `best_alpha`。`substitution` 是各 alpha 中最小的输出 KL，不是已经归一化的 CFS。
- CFS 按原实现恢复为 `1 - substitution / max(single_drop, 1e-12)`，对角线不参与 Oracle；有效 source 为 `single_drop > 1e-5`。Oracle 不增加裁剪。
- Qwen 的八个预测位置先在 KL 中平均，再选最优 alpha；不把位置作为独立输入。矩阵保存的是这个聚合结果，不是逐 token 的矩阵。
- 现有正式 Qwen-14B 完整矩阵可直接离线复用；Qwen-1.5B/3B/7B 与 ImageNet-1K ViT 三模型的本地正式矩阵缺失。现存 ViT Imagenette 数据不能替代 ImageNet-1K。
- 历史 Qwen 的冻结文本未找回，因此新的四规模统一重采集属于独立的 rebuttal 复现实验，不能声称逐输入或逐数值复现原论文。

## 固定的方法

| 模型 | H | 层（从零计数） | 输入 | Alpha |
|---|---:|---|---:|---|
| Qwen2.5-1.5B base | 12 | 6,13,20 | 32 contexts | 16 个点，[0,3] |
| Qwen2.5-3B base | 16 | 8,17,26 | 同上 | 同上 |
| Qwen2.5-7B base | 28 | 6,13,20 | 同上 | 同上 |
| Qwen2.5-14B base | 40 | 11,23,35 | 同上 | 同上 |
| ViT-Small AugReg IN21K ft IN1K | 6 | 11 | 512 images | 31 个点，[0,3] |
| ViT-Base AugReg IN21K ft IN1K | 12 | 11 | 同上 | 同上 |
| ViT-Large AugReg IN21K ft IN1K | 16 | 23 | 同上 | 同上 |

Qwen：原收集器，BF16，长度 128，`random:8`，position seed 20260802，minimum position 7。
新的 FineWeb-Edu 10BT parquet 冻结文件保留 256 个 129-token 数组及 document ID，四模型共用前 32 个。
ViT：原收集器，FP16，`vit_{small,base,large}_patch16_224.augreg_in21k_ft_in1k`，timm 原 checkpoint 的评估预处理，按类别/文件名排序后的 50,000 张 ImageNet 验证图像使用 `linspace(0,49999,512)`。
从公开完整分辨率 parquet 复本恢复 JPEG 原始字节，保存来源版本、文件校验和、逐图 SHA256。
权重逐文件验证 LFS SHA256 或 Git blob SHA1，并记录来源 revision。

Matched-N 调用原 `analyze_scaling_cfs.expected_best_of_k`，以整数二项式系数之比计算精确期望。Qwen N=2,4,8,11；ViT N=2,4,5。

Fixed-K 对所有 H 个节点均匀抽样，每次内部无放回，跨次独立；seed 42，每输入/层/K 500 次。
Qwen K=4,8,12；ViT K=4,6。K=H 只计算一次。
重复抽到的相同子集可复用计算，但保留原抽样次数和权重。
调用原 `exact_minimum_cover`（严格 CFS>0.5，自覆盖，有效 source 必须覆盖，所有节点可当 representative）和 `spectral_effective_rank`（正值裁剪、有效 source 自亲和为 1、无效 source 行置零、奇异值熵有效秩）。
分母是全部 K 个节点，不能改成有效节点数。负 CFS 只在原 Rank 定义要求处裁剪。

## 统计与可比性

Oracle 在每层内按所有有效 source 等权，然后各选定层等权；不能在有效头数不等时直接平均逐输入 Oracle。
Cover/Rank 先对子集平均，再对输入和层平均。
Bootstrap 2000 次，以整个独立输入为单位连同其全部层/头重采样；每次重新计算 Oracle 的分子/有效 source 数之比。
各模型输入排序、bootstrap seed 和抽样次数保持一致；只有冻结输入 hash 一致才计算配对模型差异。
Fixed-K 单独报告条件于输入的 Monte Carlo 标准误，以及 100/250/500 次抽样的均值变化。
Bootstrap CI 条件于已采集矩阵和保存的子集均值；Monte Carlo 标准误单列，不把 500 子图当作独立输入。
CI 是逐项 percentile 95% CI，不是多重比较校正后的同时置信带。
输入标准差描述逐输入统计量波动；Oracle 的主估计仍遵循原 source 权重。

32 个 Qwen 输入只支持有限精度的样本内复现。Qwen 深度、宽度、GQA 等随规模同时变化；ViT-L 的有效 source 比例也可能变化。
固定 N/K 仅控制候选数/图节点数，不能证明因果 scaling、跨任务普遍规律、可执行联合压缩率或任务效用。

## 新增脚本与运行

- `prepare_rebuttal_remote.py`：下载、校验、冻结输入。
- `run_rebuttal_pipeline.py`：GPU 1 原收集器小样本检查、完整收集、分模型离线分析。
- `analyze_matched_scaling.py`：精确 Matched-N、Fixed-K、输入级 bootstrap、逐输入/层结果和子图。
- `summarize_matched_scaling.py`：配对差异、趋势、CSV/JSON/LaTeX 和 PDF/PNG 图。
- `validate_matched_scaling.py`：同矩阵原实现 Matched-2/Raw 一致性、K=H 和抽样稳定性。
- `report_matched_scaling.py`：根据实际表格生成 Markdown 报告，记录包含零的差异 CI；缺失模型不作完整趋势判断。
- `finish_rebuttal_pipeline.py`：等待已运行的 ViT 任务，先输出阶段报告，再串行执行 Qwen 与最终汇总。
- `rebuttal_status.py`：只读查看日志、分片进度和数据类别完整性。
- `tests/test_matched_scaling.py`：组合枚举、边界和聚合测试。

远端独立工作目录为 `/data/cfs_rebuttal_20261114`，独立环境为 `venv`。GPU 0 保留给原有实验。

```bash
cd /data/cfs_rebuttal_20261114
venv/bin/python scripts/prepare_rebuttal_remote.py qwen
venv/bin/python scripts/prepare_rebuttal_remote.py imagenet
venv/bin/python scripts/run_rebuttal_pipeline.py --family Qwen
venv/bin/python scripts/run_rebuttal_pipeline.py --family ViT
# 离线重分析使用新输出目录；--model 可重复，也可省略以运行整个 manifest。
venv/bin/python scripts/analyze_matched_scaling.py --manifest qwen_manifest.json --output analysis_repeat --model qwen15b --repeats 500 --bootstrap 2000 --seed 42
venv/bin/python scripts/summarize_matched_scaling.py --input analysis --output figures
venv/bin/python scripts/validate_matched_scaling.py --result analysis/qwen15b
venv/bin/python -m pytest -q tests/test_matched_scaling.py
```

输出 `analysis/<model>/` 包含逐输入/层分子、分母、值、Monte Carlo SE、seed/抽样次数，全部 bootstrap replicate，抽中的 head IDs、子图统计，来源 hash，以及汇总 CSV/JSON/LaTeX。
汇总表的 `repeats` 是请求的子图抽样上限（500）；逐输入表 `draws` 记录实际次数，精确 Oracle 为 0、完整图及 K=H 为 1。
`figures/` 包含完整表、相邻模型配对差异、单调性检查、抽样稳定性摘要及两幅论文用图。
不要将未完成模型的结果填入表中，也不要把新的四规模 Qwen 结果与历史 14B 拼成一条曲线。
