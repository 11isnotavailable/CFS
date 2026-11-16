# BERT-Medium 七模型受控实验：启动记录

状态：**正式训练运行中，尚无完整 2B CFS 结果。** 本记录只报告已完成核验、实测吞吐量及冻结协议，不把预估或软件测试当作实验结论。后续正式分析由 `scripts/analyze_bert_medium.py` 从全部七模型的完成文件生成。

## 已完成的核验

- 复核原 GPT-2 档案：FineWeb `sample-10BT`，revision `9bb295ddab0e05d785b879661af7260fed5140fc`；沿用原文档哈希训练/验证划分。原实验没有独立 test split。
- 已重新 WordPiece tokenize 并冻结 2,000,000,000 个非 padding 输入位置：15,625,000 个长度 128 的块，包含 CLS/SEP，共 1,968,750,000 个正文 token。所有模型读取同一顺序，动态 mask 使用独立公共随机流。
- 固定 2,048 个独立验证文档、256 个 CFS 文档及其中 64 个稳定性 probe。输入、mask、位置和来源哈希均已保留。
- B2/B6/B7 的实际参数量均为 **41,404,730**，同一运行环境下初始参数逐项相同；B2 只训练一次。
- 12 项针对性测试通过；稀疏 MLM loss/gradient 等价检查通过。14 步连续训练与 7+7 断点恢复权重逐位一致，最大误差 0，CUDA/Mask RNG 一致。
- 缓存最后一层的 CFS 与完整前向在 4 个输入上核验，删除/替代 KL 最大绝对差异小于 `5.75e-7`，最佳 alpha 一致。正式 checkpoint 首个输入继续做该核验。
- 使用隔离的合成数据完整执行了统计、CSV/LaTeX 输出和绘图流程；这些合成数据不进入科研结果。

详细证据见 [架构与 FLOPs](audit/architectures.csv)、[数据冻结清单](data/READY.json)、[代码与协议锁](PROTOCOL_LOCK.json)、[数值检查](smoke/CFS_CHECKS_PASSED.json)、[恢复检查](smoke/TRAIN_RESUME_CHECKS_PASSED.json)。

## 新架构实测

两张 RTX PRO 6000 Blackwell Workstation Edition；batch 256、seq 128、BF16、真实 WordPiece 缓存、完整 forward/backward/optimizer；每项 50 warmup + 200 计时更新。

| 模型 | 参数量 | tokens/s | 秒/update | GPU 利用率 | 峰值 allocated VRAM |
|---|---:|---:|---:|---:|---:|
| B1 W384/D8/H6 | 26,293,434 | 967,794 | 0.03386 | 93.92% | 6.09 GiB |
| B2 W512/D8/H8 | 41,404,730 | 725,496 | 0.04517 | 96.65% | 7.61 GiB |
| B3 W640/D8/H10 | 59,694,522 | 509,803 | 0.06428 | 97.04% | 9.23 GiB |
| B5 W512/D12/H8 | 54,014,266 | 514,786 | 0.06365 | 96.52% | 10.45 GiB |

另一次独立 I/O 确认运行得到 980,072 / 728,006 / 510,168 / 518,255 tokens/s。主机取 batch 等待与异步拷贝提交时间占 0.56%–0.94%；没有单独隔离 device copy 完成时间。正式训练有额外 gradient clipping、验证和存盘，不能用短测窗口保证完整运行速度。

七模型各 2B 的纯训练预估 **5.65 GPU-hours**；最优双卡约 **2.87 小时**，假设三卡约 **1.95 小时**。B4 为深度外推，B6/B7 暂用 B2 速度代理，尚非其实测。准备、验证、存盘、CFS 和统计不在该估算内。详见 [完整 Benchmark](benchmark/REPORT.md)。

原 GPT-2 七模型实际各约 3.0005B token，总训练完成日志为 **38.00 GPU-hours**。这只能作为历史资源量参照：模型尺寸、seq 1024/128、BPE/WordPiece、训练目标、GPU Server/Workstation 版本及软件版本均不同，不能将两者比值解释为 BERT 的受控加速倍数。

## 正式运行

断点恢复预检通过后已经启动：

- GPU 0：B2 → B7 → B5。
- GPU 1：B1 → B3 → B4 → B6。
- 每模型先完成统一 2B，再执行同模型 1B/64 输入和 2B/256 输入的最后层 CFS；完成后进入下一模型。
- 七模型完成后统一生成输入级 2000 次 bootstrap、配对差异、Matched-N、Fixed-K=4、1B/2B 稳定性检查和图表。
- **不自动延长到 3B；不按结果修改学习率、阈值或主 checkpoint。**

最早正式运行窗口中，B1/B2 约 90–93 万 / 69–70 万 tokens/s；初始验证 CE 下降到约 6.4，尚不能据此判断训练充分性或最终功能组织。

## 方法对齐与限制

沿用原 BERT MLM 的 FP16 干预协议、8 个固定 MASK 位置、16 点 `[0,3]` alpha grid、删除 KL `>1e-5` 有效源规则、coverage 阈值 `0.5`。原协议 alpha 是候选 head 的绝对倍率，**alpha=1 对应只删除 source**。保留原 BERT 先对每输入有效源求平均、再对输入平均的 Oracle 聚合。

Rank 沿用最终论文 `build_section5_delivery.py` 调用的 `spectral_effective_rank`：裁剪 affinity、自连接和无效行处理；不使用旧 BERT 专用脚本中被替换的 no-self rank。原函数复用与差异记录见 [冻结协议](../../docs/BERT_MEDIUM_CONTROLLED_PROTOCOL.md)。

Bootstrap 单位为独立文档；不同 head、子图不作为独立样本。固定子图抽样误差另外记录。第一轮仅一个共同训练种子，因此输入级 CI 不覆盖训练随机性。

## 当前能回答的问题

- **Q1：计算是否可行？** 新四模型实测和数据准备已完成，当前双卡成本可行，已启动正式训练。
- **Q2：2B 是否足够稳定？** 待七模型学习曲线、有效干预比例及同输入 1B/2B probe；暂不能判断。
- **Q3/Q4：Width、Depth、等参数 Head 分解的趋势？** 待完整结果及候选数量控制，暂不下结论。
- **Q5：是否形成第二架构的实证支持？** 方法移植与数值实现已核验；科学结论仍待正式矩阵，不能以 smoke test 代替。

当前不提供声称取得 scaling 发现的 Reviewer WCQn 英文结论。最终报告须以实测趋势和配对区间为依据，也需完整报告非单调或退化结果。

## 代码和复现

`src/bert_medium.py` 冻结架构/训练协议；`prepare_bert_medium_data.py` 构建带文档溯源的公共数据；`train_bert_medium.py` 负责精确预算、验证和恢复；`bert_medium_intervention.py` 与 `collect_bert_medium.py` 复用原干预并缓存最后层；`analyze_bert_medium.py` 生成统计和报告；`run_bert_medium.py` 校验代码/数据锁并管理双卡队列。Benchmark、架构审计、恢复/干预预检分别有独立脚本。

远端运行目录：`/data/cfs_bert_medium_20261116`。查看进度：

```bash
cd /data/cfs_bert_medium_20261116
/data/cfs_rebuttal_20261114/venv/bin/python scripts/status_bert_medium.py --root "$PWD"
```

已有队列运行时不要重复启动。训练 checkpoint 在 `runs/B*/model_{500000000,1000000000,1500000000,2000000000}.pt`，恢复状态在 `last.pt`；原始矩阵在 `cfs/B*/{1B,2B}/shard_*.npz`。这些大文件保留远端，不写入 Git。其余命令和依赖见冻结协议及 Benchmark 环境清单。
