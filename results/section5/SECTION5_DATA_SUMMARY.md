# Section 5 数据交付与协议索引

## 1. 交付状态

GPT 所需的 Section 5 数据主体已经齐全，可直接用于撰写正式稿。

| 模块 | 状态 | 说明 |
|---|---|---|
| ViT natural scaling | 完整 | Tiny/Small/Base/Large；主表采用原生 ImageNet-1K 协议 |
| DeiT natural scaling | 不存在完整 sweep | 只有 DeiT-S 的发现与验证实验，不应作为第四条 scaling family |
| BERT natural scaling | 完整 | Tiny/Small/Base/Large；预训练 MLM，未微调 |
| Qwen2.5 natural scaling | 完整 | 0.5B/1.5B/3B/7B/14B 的 loss、Raw、Matched-2、cover/head、rank/head、deletion KL 与 weighted recovery 完整 |
| Controlled depth | 完整 | D6/D12/D18 |
| Controlled width | 完整 | W192/W384/W768 |
| Controlled head granularity | 完整 | H3/H6/H12，参数量与解析 FLOPs 相同 |
| Training dynamics | 完整 | epoch 1--100，layers 3/7/11，含性能、CFS、graph 与 role stability |

因此，写作不需要等待新实验。Qwen2.5 0.5B--7B 的原始 `single_drop` 与 `substitution` shards 已从旧 3090 找回，并完成统一 self-affinity 口径的 rank/head 后处理；没有重新运行模型。

## 2. 推荐直接上传的文件

### 2.1 Natural scaling

- `natural_scaling_vit.csv`：ViT Tiny/Small/Base/Large，ImageNet-1K 主协议。
- `natural_scaling_vit_imagenette_replication.csv`：同一 ViT family 的 Imagenette 复现实验，仅作补充。
- `natural_scaling_bert.csv`：BERT Tiny/Small/Base/Large。
- `natural_scaling_qwen.csv`：Qwen2.5 0.5B--14B；0.5B 保留并标注 64-d head，其余为 128-d head。
- `qwen_random8_unified_rank.csv`：Qwen2.5 0.5B--14B 的逐层及三层平均 effective rank/head 后处理结果。
- `natural_scaling_deit_availability.csv`：明确记录仅有 DeiT-S，没有完整 scaling sweep。

### 2.2 Controlled scaling

- `controlled_scaling_depth.csv`
- `controlled_scaling_width.csv`
- `controlled_scaling_head_granularity.csv`

三张表均含模型结构、参数量、估算 GMACs、accuracy、deletion KL、Raw、Matched-2、cover/head、rank/head、weighted recovery 与 valid-source rate。

### 2.3 Training dynamics

- `training_dynamics_epoch_layer.csv`：100 epochs × 3 layers 的完整轨迹。
- `training_dynamics_epoch_macro.csv`：三个 layer 平均后的逐 epoch 轨迹。
- `training_dynamics_graph_similarity.csv`：epoch-to-final CFS matrix Spearman 与 top-1 representative agreement。
- `training_dynamics_head_role_stability.csv`：每个 source head 的角色排序与最终 epoch 的相似度。
- `training_dynamics_layer_summary.csv`：epochs 1/5/10/25/50/75/100 的紧凑 layer-wise 表，适合直接写文或制表。

## 3. Natural scaling 主结果

### 3.1 ViT：原生 ImageNet-1K

| Model | Params | Heads | Head dim | Accuracy | Raw | Matched-2 | Cover/head | Rank/head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 5.72M | 3 | 64 | 0.7695 | 0.3379 | 0.3379 | 0.7982 | 0.8998 |
| Small | 22.05M | 6 | 64 | 0.8340 | 0.4088 | 0.2852 | 0.7321 | 0.8602 |
| Base | 86.57M | 12 | 64 | 0.8535 | 0.4976 | 0.2889 | 0.5972 | 0.7276 |
| Large | 304.33M | 16 | 64 | 0.8691 | 0.6034 | 0.3737 | 0.3455 | 0.4647 |

这里的 accuracy 与 CFS 均在共享的 512-image deterministic validation subset 上统计，并非完整 50k validation accuracy。

### 3.2 BERT：预训练 MLM

| Model | Params | Layers | Heads | Head dim | MLM loss | Raw | Matched-2 | Cover/head | Unified rank/head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 4.42M | 2 | 2 | 64 | 4.5937 | 0.0128 | 0.0128 | 1.0000 | 0.9998 |
| Small | 28.80M | 4 | 8 | 64 | 3.0326 | 0.1318 | 0.0634 | 0.9922 | 0.9827 |
| Base | 109.51M | 12 | 12 | 64 | 3.0940 | 0.1365 | 0.0515 | 0.9948 | 0.9769 |
| Large | 335.17M | 24 | 16 | 64 | 3.1018 | 0.1799 | 0.0692 | 0.9414 | 0.9209 |

`natural_scaling_bert.csv` 同时保留 `rank_per_head_legacy_no_self`。原 BERT analyzer 没有给有效 source 添加单位自环，而 ViT capacity analyzer 添加了；上表的 `rank/head` 已从 BERT 原始矩阵按 ViT 定义重算，供跨 family 使用。原结果文档中的 legacy rank 仍可用于复现旧表，但不应与 ViT rank 直接比较。

### 3.3 Qwen2.5：开放域 causal LM

| Model | Params | Q/KV heads | Head dim | Loss | Raw | Matched-2 | Cover/head | Rank/head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 0.494B | 14/2 | 64 | 2.9939 | 0.3608 | 0.2139 | 0.8289 | 0.8128 |
| 1.5B | 1.544B | 12/2 | 128 | 2.7024 | 0.2725 | 0.1620 | 0.9549 | 0.9052 |
| 3B | 3.086B | 16/2 | 128 | 2.5819 | 0.3198 | 0.1953 | 0.8984 | 0.8266 |
| 7B | 7.616B | 28/4 | 128 | 2.4570 | 0.3712 | 0.2180 | 0.8240 | 0.7427 |
| 14B | 14.770B | 40/8 | 128 | 2.3134 | 0.4430 | 0.2745 | 0.6992 | 0.6419 |

不要删除 0.5B。正文应明确：这是自然模型族横截面，模型间同时改变 depth、width、query/KV heads 与 GQA；0.5B 的 query-head dimension 为 64，而 1.5B--14B 为 128，因此 0.5B 到 1.5B 的变化不应解释为单一规模变量的因果效应。

### 3.4 DeiT

本地和既有远端回收结果中只有 `deit_small_patch16_224`，没有 DeiT Tiny/Base/Large 的统一 natural scaling sweep。建议 Section 5 natural scaling 只写 ViT、BERT 与 Qwen2.5；DeiT-S 留在 Section 4 的 discovery/validation 证据中。

## 4. Controlled scaling 完整性

共同训练协议是 ImageNet-100、seed 17、75 epochs、random initialization；CFS 使用 final-epoch checkpoint，而 accuracy 采用既定 recognition evaluation。所有 CFS 数字来自 final attention layer 的共享 512-image test subset。

### Depth axis

`D6/D12/D18` 固定 width=384、heads=6、head_dim=64。CSV 已含全部所需字段。

### Width axis

`W192/W384/W768` 固定 depth=12、head_dim=64，heads 随 width 为 3/6/12。CSV 已含全部所需字段。

### Head-granularity axis

`H3/H6/H12` 固定 depth=12、width=384，仅把同一 attention space 分解成 3×128、6×64、12×32。三者：

- 参数量均为 21,704,164；
- 解析估算均为 4.5985 GMACs（224×224、patch 16）；
- 因此可用于说明整体 dense 计算规模基本不变。

CSV 中的 `estimated_gmacs_224` 是依据标准 ViT dense MAC 公式计算的理论估计，不是 wall-clock profiling，也不应标成实测 FLOPs。

## 5. Training dynamics 完整性

逐 epoch 数据覆盖：

- epochs 1--100；
- layers 3、7、11；
- accuracy、loss；
- Raw、Matched-1/2/4、weighted recovery；
- deletion KL、best-substitution KL；
- cover/head @ tau=0.25/0.50/0.75；
- spectral rank/head；
- CFS matrix 到 epoch 100 的 Spearman；
- top-1 best-representative agreement；
- 每个 source head 的 role-ranking stability。

因此 5.3 不需要补实验。大矩阵和 checkpoint 无需交付给写作者；当前 CSV 已覆盖其要求的所有 summary 统计。

## 6. 跨 family 协议说明

| Family | Dataset/task | Performance | CFS samples | Layers | Functional outcome D | Alpha | Valid source | Matched | Cover |
|---|---|---|---:|---|---|---:|---|---|---|
| ViT natural | ImageNet-1K classification | accuracy on shared 512-image subset | 512 | final block: 11/23 | KL over 1000-way class distribution | 31-point [0,3] | deletion KL > 1e-5 | expected best of 2 candidates | tau=0.5, exact enumeration |
| BERT natural | frozen MNLI validation text, pretrained MLM | MLM loss on 512 texts, 8 masked tokens/text | 32 | final layer | mean KL over 8 masked-token distributions | 16-point grid | deletion KL > 1e-5 | expected best of 2 | tau=0.5, exact MILP |
| Qwen2.5 natural | frozen FineWeb-Edu slice, causal LM, seq 128 | all-token loss on 256 contexts | 32 | approx. 25/50/75% depth | mean KL over 8 deterministic random-schedule next-token positions/context | 16-point grid | deletion KL > 1e-5 | expected best of 2 | tau=0.5, exact MILP |
| Controlled ViT | ImageNet-100 classification | test accuracy | 512 | final block | KL over 100-way class distribution | 16-point grid | deletion KL > 1e-5 | expected best of 2 | tau=0.5, exact enumeration |
| Training dynamics | ImageNet-100 classification | accuracy/loss | 512 fixed probes | layers 3/7/11 | KL over 100-way class distribution | 16-point grid | deletion KL > 1e-5 | expected best of 1/2/4 | tau=0.25/0.5/0.75 |

### 可比较性的边界

1. `S = 1 - D_sub / D_drop`、valid-source threshold、Matched-2 与 cover threshold 的数学意义跨 family 一致。
2. `D` 的输出空间随任务变化：分类 logits、masked-token distributions 与 next-token distributions 的 KL 数值尺度不同。因此 deletion KL 的绝对值不能直接跨任务比较；normalized recovery、cover/head 与统一定义的 rank/head 更适合比较趋势。
3. Natural scaling 是公开 checkpoint family 的横截面，不是单变量受控 scaling。只有三组 controlled ViT 实验支持特定结构轴的干预式比较。
4. Raw Oracle 随候选 head 数增长，必须与 Matched-2 同时报；Tiny/H3 只有两个候选，所以 Raw 与 Matched-2 必然相同。
5. Rank/head 的统一定义为：对正 CFS affinity 截断到 [0,1]，给有效 source 加单位 self-affinity，无效 source 行置零，对有向矩阵取奇异值熵有效秩并除以 H。

## 7. 仍需补什么

### 写作必需

无。现有数据足够写 Section 5，建议使用三条 natural family（ViT/BERT/Qwen）而不是虚构 DeiT sweep。

### 可选增强

当前 GPT 指定的 Section 5 汇总字段已经全部补齐。后续若需置信区间，可对现有 sample-level rank 进行 paired bootstrap；这属于统计增强，不需要重新运行模型。

## 8. 原始来源

- ViT ImageNet-1K：`outputs/imagenet1k_final/outputs/imagenet1k_scaling_analysis_test512_alpha31/` 与 `imagenet1k_functional_capacity_test512_alpha31/`。
- ViT Imagenette：`outputs/vit_scaling_analysis_tsbl/`、`FUNCTIONAL_CAPACITY_RESULTS.md`。
- BERT：`outputs/bert_mlm_natural_scaling/analysis/` 与四个 `*_final/` raw directories。
- Qwen 0.5B--7B：`outputs/qwen_scaling/`、`outputs/qwen_measurement_audit/`。
- Qwen 0.5B--7B unified rank：`outputs/section5_delivery/qwen_random8_unified_rank.csv`，由旧 3090 canonical random-8 raw shards 后处理得到。
- Qwen 14B：`outputs/qwen14b_final/`。
- Controlled ViT：`outputs/controlled_scaling_smallbase_v1/`。
- Training dynamics：`outputs/functional_evolution_visualizations/`。

本交付包可由 `scripts/build_section5_delivery.py` 从上述来源重新生成。
