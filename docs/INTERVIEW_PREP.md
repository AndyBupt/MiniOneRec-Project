# 生成式推荐项目面试准备

> **使用条件**：以下是“项目完成后”的待填模板，当前尚未训练，不能原样写进简历或口头声称已完成。所有 `[待填]` 都必须以 `EXPERIMENT_PLAN.md` 的真实日志替换；未完成项删去。原版能力和个人贡献以 `CODEBASE_ANALYSIS.md` 为准。

## 面试时先讲清的 10 个问题

1. **为什么要 Semantic ID？** 普通 item ID 无语义，LLM 需要在大 catalog 上输出可控 token；SID 把文本语义压成短离散序列，降低输出空间并可共享前缀。代价是量化误差、碰撞与 SID 维护。
2. **RQ-VAE 如何工作？** frozen encoder 给 item embedding，MLP 压缩到量化空间；逐层对残差选码字，码字和送解码器重构；优化重构误差和码本/commitment loss，straight-through 传梯度；导出 `<a_i><b_j><c_k>`。
3. **为什么用 LLM 生成 item？** 自回归模型可统一消费历史 SID、item 文本与任务指令，输出离散 SID；SFT 对齐 catalog，RL 再针对排序指标优化。它的代价是解码慢、训练昂贵，不能笼统说优于传统推荐。
4. **SFT 数据和 loss 是什么？** 历史按时间顺序写 prompt，target 是下一件商品的 SID；prompt token label 为 `-100`，只对 target SID 与 EOS 做 token CE。仓库另混入 SID↔title、SID 历史→title 任务，valid 仅 SID next-item。
5. **为什么 SFT 后还做 GRPO？** CE 只最大化真实 target 的 token likelihood，不直接优化 Top-K 候选的相对排序；GRPO 从同 prompt 生成多个候选，借组内相对奖励更新偏好，但效果必须用离线指标证实。
6. **奖励和 Advantage 如何算？** 原版有精确命中 0/1、rank-aware 错误候选惩罚；多奖励相加后每组 G 个 completion 标准化 `A=(r-mean)/(std+ε)`。同奖组的偏好梯度为零；KL 仍可能有梯度。
7. **RL 为什么可能改善推荐？** rollout 暴露模型实际会出的候选，奖励可直接区分候选好坏；若采样覆盖不到 GT 或奖励噪声大，就可能无益或退化。实验须报告 zero-var 组率和 Top-K 差值。
8. **主要技术挑战是什么？** 数据/奖励真值一一对应、SID 碰撞、约束生成正确性、单卡 policy+reference+rollout 显存峰值、版本兼容以及公平的评估协议。
9. **个人贡献与开源区别？** 原版已有 RQ-VAE、SFT、GRPO 风格训练、ranking reward、HEPO、constrained decoding。计划中的个人方法是“零方差组条件 SFT 锚定”，另有目标映射/指标修复；只在完成实现与对照后才能宣称实际贡献。
10. **如何证明有效？** E1 SFT、E2 同起点原版 GRPO、E3 只加条件 CE、E4 门控与 λ 消融；固定 split、SID、基座、step、G、β、解码；看 HR/NDCG、置信区间、zero-var 率、成本与失败案例。

## 简历项目描述：完成后从以下 3–4 条填实

- 基于 MiniOneRec 构建 `[数据集]` 生成式推荐实验链，复核 RQ-VAE 三层 SID（`[item 数]`、碰撞率 `[实测]`），完成 Qwen2.5-`[规模]` 的 SID SFT 与 Top-K 离线评估；明确采用 transductive catalog 与时间划分协议。
- 从相同 SFT checkpoint 实现/复现仓库 GRPO 风格训练，审计组内奖励标准化、ranking reward、KL 与约束解码；记录 `[G/β/步数]`、zero-var 组率 `[实测]`、invalid@10 `[实测]`。
- **若 E3 完成**：针对零方差组设计条件目标 CE 锚定，在保持原版 rollout 奖励和 KL 的前提下做 E2/E3 与门控消融；HR@10 从 `[E2 实测]` 到 `[E3 实测]`、NDCG@10 从 `[E2 实测]` 到 `[E3 实测]`，并报告 `[置信区间/种子]` 与 `[GPU 小时]`。若无可靠增益，写“完成对照并分析稀疏奖励失效条件”，不写提升。
- 修复/规避开源实现中相同历史多目标的奖励错配与 SID 碰撞评估歧义，统一 valid/test 协议；输出可复现实验配置、成本和错误案例。此条属于实验正确性贡献，不称算法创新。

不要写“线上部署”“提升点击率/GMV”“工业规模”“HEPO 由我提出”。若只跑成 1.5B 与部分 RL，简历精确写完成范围。

## 2 分钟项目介绍（完成后按括号替换）

“我做的是一个基于开源 MiniOneRec 的生成式推荐项目，目标是把 item 的文本特征转成 Semantic ID，再用 Qwen2.5 根据用户历史生成下一个 item 的 SID。主实验用了 `[数据集]`，按照 `[时间切分方式]` 划训练、验证和测试，同一个 SID catalog 贯穿 SFT、RL 和评估。

我先复核了 RQ-VAE：它把 item embedding 编成三层残差码，每层选一个码字，训练时兼顾重构误差和量化损失。SFT 阶段用历史 SID 预测下一个 SID，同时利用仓库已有的文本对齐任务。之后从同一个 SFT checkpoint 做原版 GRPO 风格训练：对一个 prompt 生成 G 个候选，用精确命中与排序奖励算组内 Advantage，并用 KL 约束更新。

源码审查时，我发现同一历史可能有多个 target，但原代码用字典只保留一个，奖励可能错配；已有 SID 也存在少量碰撞。因此我先统一了样本级 GT 和指标口径。在这个共同底座上，我的个人优化是 `[仅在实际完成时说：对零方差奖励组加条件目标 CE 锚定]`。E1–E4 固定模型、SID、数据与训练预算，最终 `[填真实 HR/NDCG、置信区间和额外成本，或如实说未提升]`。最重要的经验是，生成式推荐里的 reward 与候选空间、数据协议一起决定结果，不能只看 loss 下降。”

## 5 分钟深度讲解提纲

| 时间 | 要讲什么 | 面试官可能追问时展示的证据 |
|---|---|---|
| 0:00–0:40 | 问题定义：下一 item 预测、全 catalog 生成、时间划分、为什么用 SID | catalog item 数、SID 层数及与 item ID 的区别。 |
| 0:40–1:30 | RQ-VAE：文本 encoder→MLP→残差量化→重构；loss 与碰撞 | `rq/models/rqvae.py`、`rq/models/rq.py`、`rq/models/vq.py`；实际碰撞统计。 |
| 1:30–2:15 | SFT：样本、响应 CE、辅助任务、tokenizer 新增 SID token | `data.py:397-466`、`sft.py:150-205`；SFT validation HR/NDCG。 |
| 2:15–3:20 | RL：G 候选、原版奖励、Advantage、KL、约束解码；说明原版无 PPO clip | `rl.py:156-197`、`minionerec_trainer.py:965-1064`；reward/zero-var/KL 曲线。 |
| 3:20–4:15 | 自己做的事：先解决样本目标映射，再做条件 CE 锚定；与 HEPO/add_gt 的区别 | 单测/样本审计、E2/E3/E4 代码 diff、消融表。 |
| 4:15–5:00 | 结果与局限：HR/NDCG、CI、invalid、显存/成本；失败或无提升也解释 | 同协议结果表、paired bootstrap、典型失败案例。 |

## 20 个高频追问与答题线索

| # | 问题 | 答题线索 |
|---:|---|---|
| 1 | SID 相比原始 item ID 的收益和代价？ | 语义共享、短序列、受限输出；量化误差、碰撞、catalog 更新成本。 |
| 2 | 三层 RQ 与单层 VQ 的区别？ | 残差逐层编码，组合容量约 `K^L`，深层补细节；误差传播与码本利用率。 |
| 3 | RQ-VAE 的梯度如何穿过 argmin？ | straight-through；codebook 与 commitment loss 分别作用于码本/encoder。 |
| 4 | Sinkhorn 何时用、为何不能保证无碰撞？ | 平衡批内分配，导出 `use_sk=False` 可再冲突；组合容量与语义聚集。 |
| 5 | SID 碰撞怎样影响 HR？ | SID 相同的不同 item 会被当作同一个命中；分 SID 与 item 口径。 |
| 6 | SFT 为何只在响应 token 计算 CE？ | prompt 是条件，目标是生成下一件 SID；`-100` 屏蔽 prompt。 |
| 7 | 辅助 SID↔title 对齐是否构成数据泄漏？ | 全 catalog metadata 是 transductive；不能读取 valid/test 交互标签；冷启动结论不适用。 |
| 8 | `freeze_LLM=True` 与 LoRA/全参各有什么取舍？ | 可训练参数、容量、显存；新 token 的 tied head 梯度要核实。 |
| 9 | 为什么 SFT loss 降了 HR 可能不升？ | token CE 与 Top-K item 排序目标不一致；输出约束/碰撞/曝光偏差。 |
| 10 | GRPO 组内标准化如何算？ | 同 prompt G 个 reward 求均值、标准差；同奖组 A=0。 |
| 11 | 零方差组真的完全没梯度吗？ | 偏好项为 0，但 reference KL 在两策略不同后仍可能有梯度。 |
| 12 | `β` 太大或太小会怎样？ | 大则限制更新，小则分布漂移/不稳定；看 KL 与 HR 的共同曲线。 |
| 13 | 原仓库与标准 GRPO 的差异？ | 自定义 detached ratio，无 old-policy clip；可称 GRPO 风格，不混称严格 PPO。 |
| 14 | 为什么不直接用 HEPO 前缀奖励？ | 仓库已有；前缀相似不一定表示目标 item 正确，会偏向语义近邻。可列对照。 |
| 15 | `add_gt` 与条件 CE 锚定有什么不同？ | 前者改 rollout 候选分布，后者保留 sampled group，仅对无区分度组加监督 loss。 |
| 16 | 排序奖励为何可能有偏？ | 原版根据候选位次惩罚错误项，只在组内有命中时启用；与全量 NDCG 不完全一致。 |
| 17 | 受限解码如何保证候选合法？ | info 构前缀 trie/哈希表，逐 token mask；必须检查 tokenizer 前缀与 EOS。 |
| 18 | 为什么单卡 GRPO 比 SFT 更吃显存？ | policy+reference、G 路生成 KV、logprob forward、optimizer/activation 峰值。 |
| 19 | 如何证明 E3 提升不是算力或随机性？ | 同起点、同 rollout/steps 与配置，多 seed、paired CI，另报同 GPU 小时对照。 |
| 20 | 如果优化没有提升，结论是什么？ | 检查 zero-var 率、CE 与 RL 梯度竞争、约束/数据正确性；报告真实负结果与适用范围。 |

## 面试前必须填的证据清单

最终回答需能现场指向：固定 commit 与个人 diff、数据 hash/划分、SID 冲突统计、模型/环境版本、E1–E4 配置与指标、训练曲线、至少 3 个具体失败案例、单卡显存/时间/费用、统计区间。缺任何一项就缩小叙述范围。
