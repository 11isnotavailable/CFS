# MLP Group CFS：跨模块分析协议

范围：冻结 ViT-Base 的 MLP neuron groups，验证原 CFS 定义的可应用性。无训练、微调、预测器、路由、剪枝或推理加速主张。

## 代码库检查及冻结设置

原协议和函数来源：

- `scripts/run_experiment0.py`：`output_kl`（KL(p_dense || p_intervened)，temperature=1，float32 logits，微小负 KL 裁剪为 0）、`build_substitution_masks`、`summarize_substitutions`、`substitution_error_cube`、`current_distance_matrices`。
- `scripts/collect_predictor_dataset.py`：冻结 ImageNet 索引、原数据加载/预处理、真实标签交叉熵 gate Taylor。
- `scripts/analyze_baselines15.py`：固定 source，按候选代表 Taylor 大小选替代者，在相同 alpha 网格上评价所选 pair 的 CFS；在同一 source 的候选之间计算 Spearman。源组 Taylor 对删除 KL 的相关单独报告。
- `src/head_intervention.py`：在输出线性投影前施加门控，投影 bias 保留一次。

模型：`vit_base_patch16_224.augreg_in21k_ft_in1k`，复用已有 checkpoint；权重来源 revision `2ec9fb3d7bb664aac471ac44582c94d18de33780`。输入 224×224、bicubic、中心 crop_pct=0.9、mean/std 各 [0.5,0.5,0.5]，与已有 ViT 实验一致。

输入：从上一轮 ImageNet-1K 的固定 512 样本中均匀选取 64 个（索引在 metadata 中保存），所有层、G、分组 seed 共用。试验阶段使用这 64 个中的前 4 个，非重新选样。
层为实际零基索引 6、11（第 7、12 个 block）。MLP hidden width=3072，G=8/16 分别为每组 384/192 个神经元。
分组：连续分组；np.random.default_rng(seed) 均匀打乱神经元后等分，seed=42/123/2027。组索引完整保存，不按结果选种子。

## 干预定义及 alpha 差异

在 MLP 的 fc2 输入（GELU 和 eval dropout 后）对相应神经元组进行门控，所有其他组的缓存 activation 保持不变。fc1 bias 已包含于 activation；fc2 bias 不缩放且只使用一次。

与原 Attention 协议完全相同：source i 的 gain=0，representative j 的 gain=alpha，其余 gain=1；alpha 为 [0,3] 上的 31 个点。
因此 alpha=1 等于只删除 source，alpha=0 同时删除 source 和 representative。若写成额外补偿 beta，则 beta=alpha−1，beta=0 才是只删除 source；本实验不重参数化或移动原网格。

CFS=1−min_alpha(KL_replaced)/max(KL_dropped,1e−12)。有效 source 为 KL_dropped>1e−5，代表组可以是无效 source；Oracle 不增加裁剪。原始 replacement 对角 KL=0、alpha=1；用于分析的 CFS 对角为 NaN。原始逐输入 G×G 矩阵、完整 alpha KL 曲线和近零删除损失均保留。

## 缓存与数值检查

每个输入仅做一次 unchanged dense forward，缓存两层的 MLP post-activation 和 attention 后残差。每个分组配置复用缓存，只运行 fc2、当前 block 的后半段和后续 blocks，不重算前半 backbone。

冻结参数、eval、FP16 与原 ViT 一致。显存估计选批大小（上限 128），OOM 时折半；最后不足一批时用重复 mask 填充并丢弃填充值，确保 deletion/substitution/alpha=1 使用同一 GEMM shape。填充不计入统计。

首次试验捕获到 FP16 整网 batch=4 与单输入缓存路径的舍入差异；该失败试验独立保留，未进入正式结果。最终检查包括：FP32 组贡献和 bias 的重构、同 batch=1 的完整前向与缓存干预对照、所有组 gain=1 的 dense 等价、非零 bias 单元测试、alpha=1 的删除等价。跨批大小的 FP16 差异单独记录，不伪装成数学定义变化。
另外测量原 dense 输出与 padded batch 的无干预 KL，并统计有效删除 KL≤10×该值的 source 比例；该数值是诊断性敏感度标记，不是严格误差上界。

## 统计和传统指标比较

每种 layer/G/partition 单独分析，2000 次 bootstrap（seed=42）以整个输入为单位重采样，不把 pair 当作独立样本，不合并分组 seed。
Oracle/策略恢复保持有效 source 权重；pair 统计保持有效 source 的全部非对角候选权重。每次 bootstrap 重新计算分子/分母。

- Taylor 为真实标签 CE 对 group gate 的绝对一阶导数。固定 source，选择剩余组中 Taylor 最大者，再使用该 pair 原网格最优 alpha 评价；候选内 Spearman 比较 Taylor(j) 与 CFS(i→j)，不将无映射的 source 重要性与 pair 分数混算。
- Contribution cosine 使用 post-W2 的组贡献，按原实现 flatten 全部 token 与输出维度，取余弦相似度最高的代表组；不替代功能干预。
- Random baseline 为同 source 的所有其他代表组 CFS 的精确均值，也给予各 pair 相同 alpha 搜索。
- Oracle 对这些策略的差值为配对输入 bootstrap；Oracle 使用真实干预结果，优势不能解释为可部署预测性能或任务效用提升。
- 输入依赖：在固定 partition 的相同有向 pair 上统计跨输入 CFS 标准差；同一 source 的最佳 representative 在不同输入对之间的不一致概率。后者使用输入重采样 multiplicity 加权的 U-statistic，排除同一原始输入在 bootstrap 中被重复抽中而形成的自配对，避免人为增加“不切换”次数。输入 SD 对这类跨输入非线性量不适用。
- 有向性：只在两个 source 都有效的 unordered pair 上统计 |Sij−Sji| 和 S>0.5 的方向不一致比例。
- 分组稳健性：比较汇总量、CFS/删除 KL 分布和相对连续分组的配对差异。不同 partition 的 group 身份不同，不逐元素比较矩阵。
- 热图固定展示前四个预先选定输入，另提供各 partition 的平均热图；显示色域不修改保存的 CFS 数值。

## 复现

环境复用 `/data/cfs_rebuttal_20261114/venv`，原权重和数据只读。新实验目录 `/data/cfs_mlp_group_20261115`。

```bash
cd /data/cfs_rebuttal_20261114
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
venv/bin/python scripts/collect_mlp_group_cfs.py \
  --checkpoint models/vit_base_patch16_224.augreg_in21k_ft_in1k/collector_checkpoint.pt \
  --data-root data/imagenet1k_val --indices-metadata outputs/vit_base/metadata.json \
  --output /data/cfs_mlp_group_20261115/collection_repeat \
  --samples 64 --layers 6,11 --groups 8,16 --partitions contiguous,42,123,2027
venv/bin/python scripts/analyze_mlp_group_cfs.py \
  --input /data/cfs_mlp_group_20261115/collection_repeat \
  --output /data/cfs_mlp_group_20261115/analysis_repeat --bootstrap 2000 --seed 42
venv/bin/python scripts/report_mlp_group_cfs.py \
  --collection /data/cfs_mlp_group_20261115/collection_repeat \
  --analysis /data/cfs_mlp_group_20261115/analysis_repeat
venv/bin/python -m pytest -q tests/test_mlp_group_cfs.py
```

首次小测试把参数改为 `--samples 4 --layers 6 --groups 8 --partitions contiguous`，输出新目录。已有结果不覆盖。

## 最终交付

已完成 64 输入×2 层×2 个 G×4 种分组，共 1,024 个矩阵。正式统计使用 `analysis_final`，不使用初版跨输入切换 Bootstrap 的自配对结果。最终结果和英文 Reviewer UWwP 回复见 [实验报告](../results/mlp_group_cfs_20261115/EXPERIMENT_REPORT.md)。
新增文件为 `src/mlp_group_intervention.py`（缓存/门控）、`scripts/collect_mlp_group_cfs.py`（采集）、`scripts/analyze_mlp_group_cfs.py`（统计与图）、`scripts/report_mlp_group_cfs.py`（报告）和 `tests/test_mlp_group_cfs.py`（数学及统计检查）。原 Attention 收集与分析代码未修改。
