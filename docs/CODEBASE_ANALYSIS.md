# MiniOneRec 源码审查

> 更新（2026-10-22）：下文原版行号与缺陷描述保留为首次审查依据。GRPO主入口现已做行级标签/隔离与资源修复，当前实现和验证边界见 [GRPO_REPAIR.md](GRPO_REPAIR.md)；GPU训练尚未验证。

审查版本：`5f4f733cfda387f7f6f8f158db0d300eed9471c6`。路径均相对仓库根目录，行号对应此 commit。阅读范围包括 README、三个训练入口、数据处理、RQ 模型、约束生成、评估及 GPR/HEPO 分支。仓库自带 CSV、embedding、index 和 item metadata；没有模型权重。下述结论区分已实现代码与尚待实验验证的行为。

## 目录与真实执行路径

| 阶段 | 入口/关键函数 | 行为 |
|---|---|---|
| Amazon 数据 | `data/amazon18_data_process.py:218-339` 的 `generate_interaction_list_json2csv_style`、`convert_to_atomic_files_json2csv_style` | 按时间戳排序交互，再按全局时间顺序 8:1:1 划分；每条样本用此前历史，写 `*.inter`。 |
| Item embedding | `rq/text2emb/amazon_text2emb.py:21-50` 的 `generate_text`/`preprocess_text` | 拼 title+description；`load_qwen_model` 在 148–157 行加载文本 encoder。 |
| RQ-VAE | `rq/rqvae.py:14-90`、`rq/models/rqvae.py:10-85`、`rq/models/rq.py:39-56`、`rq/models/vq.py:63-99` | MLP encoder → 三层残差量化 → MLP decoder。量化用最近码字或 Sinkhorn；MSE 重构 + codebook/commitment loss。 |
| SID 导出 | `rq/generate_indices.py:44-149` | 载入指定 checkpoint，以 `<a_i><b_j><c_k>` 导出；对冲突组最多迭代 20 次再量化。脚本顶部有硬编码路径，执行前需参数化。 |
| 数据转换 | `convert_dataset.py:13-150` | 读取 item/index/split，SID token 串接为字符串，输出 CSV 与 `SID\ttitle\titem_id` info。train `keep_longest_only` 参数可选。 |
| SFT | `sft.py:90-268` 的 `train` | Qwen causal LM 加 SID token；训练三类任务，HF `Trainer` BF16、AdamW、响应 token CE。`sft.sh` 固定 8 进程、大 batch。 |
| RL | `rl.py:30-313` 的 `train`；`minionerec_trainer.py:121-1073` 的 `ReReTrainer` | 三种 RL prompt，多个完成项、组内奖励标准化、参考模型 KL、GRPO 风格 policy loss。`rl.sh` 为 8 进程 ZeRO2。 |
| Top-K | `evaluate.py:38-246` 的 `main`、`LogitProcessor.py:24-73`、`calc.py:12-81` | 基于合法 SID 前缀约束的 beam search，后处理并计算 HR/NDCG。`evaluate.sh` 固定拆 8 份并行。 |

## Semantic ID 和 RQ-VAE 细节

- 默认 `rq/rqvae.py:35-44` 为 3 个 256 大小码本、32 维量化空间；`rq/models/rq.py:44-54` 每层量化当前 residual，再减去码字并累加，产生三级离散索引。
- `rq/models/vq.py:70-95` 用 L2 距离选码字；正的 `sk_epsilon` 时进入 Sinkhorn 平衡分配；straight-through estimator 使梯度流到 encoder。单层损失是 `MSE(x_q, stopgrad(x)) + β MSE(stopgrad(x_q), x)`，总损失见 `rq/models/rqvae.py:74-85`。
- `rq/trainer.py:127-152` 计算整套 item 的 SID 碰撞率并在 205–215 行分别保存最低训练 loss 与最低碰撞率 checkpoint。`rq/generate_indices.py:106-141` 又尝试处理导出冲突。因此“发现/修复 SID 碰撞”“用 Sinkhorn 平衡码本”均已属原版功能。
- 仓库实际索引仍有碰撞：Industrial 3,686 个 item、3,670 个不同 SID、16 个额外碰撞项，31 个 item 处于碰撞组、最大一组 3 个；Office 3,459/3,444/15，30 个 item 在碰撞组。统计来自自带 `*.index.json`，尚未验证 item id 对齐与元数据完整性。SID 级推荐命中与 item 级推荐命中不可混称。
- README 还列出 `rq/rqkmeans_faiss.py`、`rq/rqkmeans_constrained.py`、`rq/rqkmeans_plus.py`；B 方向必须对这些现有方案做对照。

## 用户序列、SFT 与 Loss

- `data/amazon18_data_process.py:276-339` 是全局时间切分，而非用户级 leave-one-out。测试时历史可包含过去的训练/验证交互，符合时间前向使用，但不得包含当前或未来 target。`convert_dataset.py:65-149` 从 `.inter` 将历史和 target 转成 SID；可选 train 每用户仅保留最长序列，当前默认 `False`，须记录执行参数。
- `data.py:397-466` 的 `SidSFTDataset` 把按时间排列的 SID 历史写进 prompt，目标是下一个 item 的 SID；prompt 标签为 `-100`，只有响应（包括 EOS）参与 causal LM token CE。它不是全序列每位置 next-item loss。
- `sft.py:192-205` 实际拼接：`SidSFTDataset`、`SidItemFeatDataset`（`data.py:678-785` 的 SID↔title 对齐）、`FusionSeqRecDataset`（`data.py:1126-1320` 的 SID 历史→title，当前代码只用 title 分支）。验证集只用 SID 序列任务。`sft.py:134-160` 加新 SID token 并 resize embedding；`sft.py:162-190` 可冻结原 LLM，仅训练新 embedding 行（代码仅显式处理 input embedding，需验证 tied output head 的梯度）。
- `sft.py:227-268` 用 `Trainer`，BF16、AdamW、每 5% steps 评估/保存、early stop。脚本 `sft.sh` 的 `batch_size=1024` 与 `micro_batch_size=16` 在 8 卡下每设备 accumulation=8；单卡不能照搬。当前没有 LoRA 入口、显式 gradient checkpointing 或 FlashAttention 设置。
- `SidItemFeatDataset` 遍历整个 item catalog，而非仅 train 交互；这是**transductive catalog**设置，可用于已有商品推荐，但若要测冷启动或完全无泄漏的 inductive 设置必须重做。auxiliary 任务不得读取 valid/test 的用户交互标签。

## GRPO、奖励、KL 与约束生成

- `rl.py:84-111` 拼接 `SidDataset`、`RLTitle2SidDataset`、10k 条 `RLSeqTitle2SidDataset`。`minionerec_trainer.py:80-118` 的 `RepeatRandomSampler` 把同一样本连续复制 G 次；`rl.py:263-305` 配 `GRPOConfig` 后交自定义 `ReReTrainer`。
- `rl.py:156-197` 有 binary correctness 和名为 `ndcg_rule_reward` 的 rank-aware wrong-item 惩罚。`reward_type=ranking` 是两函数相加；当组内没有精确命中时，ranking 部分全 0。此逻辑不是“新增排序奖励”的空间。`rl.py:199-258` 还有 semantic cosine 与 SASRec CF reward。
- `minionerec_trainer.py:935-981` 对多奖励加权求和，按每组 G 个 completion 求均值/标准差，`A_i=(r_i-mean_g)/(std_g+1e-4)`。同奖组偏好梯度为 0；若 policy 与 reference 不一致，KL 仍能产生梯度。`compute_loss` 在 1046–1064 行用当前策略 log-prob 的 detached 比值 surrogate 与 `β·(exp(log p_ref-log p)- (log p_ref-log p)-1)` token KL；默认先按每条 completion token 平均再按样本平均。此实现没有标准 PPO/GRPO old-policy clip；应称“仓库的 GRPO 风格实现”，不能宣称完整标准 GRPO 裁剪。
- `minionerec_trainer.py:283-292` 在非 PEFT 下复制 reference 模型；`899-906` 算 reference log-prob。`rl.sh` 设置 `sync_ref_model=True`、`beta=1e-3`，并选 `beam_search=True`、`G=16`。`minionerec_trainer.py:479-495` 此分支实际上 `do_sample=True`，属于采样 beam，不是确定性 beam。`rl.py:70-71` 主动关 flash 与 memory-efficient SDP。
- `rl.py:136-138` 在建 trainer 前额外加载一次完整 LLM，而 `ReReTrainer` 于 `minionerec_trainer.py:271,283-292` 再加载 policy/reference。对 `ranking` reward，这个预加载模型没有参与训练，只为取得 device；若局部变量仍在作用域内，单卡会同时占据约三份模型权重。这是单卡移植必须先修的显存问题，属于共同工程修复，不是算法贡献。
- `minionerec_trainer.py:529-592` 从 info 构建 token 前缀表；`_prepare_inputs` 的 689–700 行创建 `ConstrainedLogitsProcessor`；`LogitProcessor.py:44-73` 将非法后继置 `-inf`，无合法后继时强制 EOS。它依赖固定的 `### Response:\n` tokenizer 前缀长度和状态计数，换 Qwen 版本必须验证。`evaluate.py:86-135,170-220` 也建前缀表并用 `do_sample=False`、`num_beams=50` 生成候选。README 已记录 Instruct 模型约束失败问题，`evaluate.py:196-204` 显式覆盖 `do_sample`。
- `minionerec_trainer.py:796-850` 已有 dynamic sampling，`858-875` 已有把 GT 加入生成组的 `add_gt`，`1056-1064` 已有 DAPO/GSPO 风格 loss 分支。`rl.py:58` 的 `mask_all_zero` 形参未传到 trainer，当前**无实际作用**；不能把它当现成实验开关。
- `rl_gpr.py:94-120` 的 HEPO 已给 SID 第一、二、三级前缀匹配 0.2、0.5、1.0 层级奖励；`rl_gpr.py:292-301` 在 `sasrec` 分支与 CF reward 相加。`sft_gpr.py:90-130` 的 `VAFT_Trainer.compute_loss` 用 `log1p(final_value)` 加权每条样本的序列 loss。因此本项目不把前缀奖励、CF 奖励、VAFT、HEPO 或多模态 token 伪装为个人贡献。

## 推荐指标及审计风险

- `evaluate.py:170-220` 返回排序候选；`calc.py:47-81` 在第一个 target SID 命中位置 `r` 上计算 `HR@K=1[r<K]`、`NDCG@K=1[r<K]/log2(r+2)`（代码用自然对数后归一化，等价）。`calc.py:62-65` 的 `CC` 是**遇到 target 前**的非法预测计数，且命中后停止循环；它不是完整 Top-K 无效 SID 比率。项目需另算 `invalid@K = 无效候选数/(查询数×K)`，重复 SID 率也另报。
- 仓库评分按 SID 字符串，不按原 item id，碰撞会使两个不同物品共享“命中”。主表须给 SID 口径，并补充 item 口径或至少报告碰撞涉及查询比例；item 口径遇碰撞必须定义确定映射或标作不可判别。
- `data.py:386-388` 及 `rl.py:120-131,160-197` 通过 `prompt→history→target` 字典取奖励真值。Industrial train 中有 1,550 种历史对应多个 target，valid 有 53 种；Office train 有 1,617 种，valid 有 87 种。后写入会覆盖前一条，且 `rl.py` 将 eval 映射并入训练映射，这会让同 prompt 的奖励可能引用另一条甚至验证样本 target。**这是必须先修的正确性问题**：奖励直接读取当前样本的 `completion`/target 列，按样本 id 分组；禁止将 eval 标签并入训练奖励查表。所有 E2–E4 使用相同修复；原版 reward 公式不改。
- 自带 Industrial train/valid/test 为 36,259/4,532/4,533 行，Office 为 38,924/4,866/4,866 行；完全相同 `(history SID, target SID)` 在 Industrial train∩test 有 39 例、Office 有 73 例。可能是不同用户的共同短历史，不可仅凭重合判定泄漏；需要用 user id、时间及原始 item id 逐条审计，并决定是否报告去重敏感性结果。
- `data.py` 多处用 `eval` 解析 CSV 字符串，包括 363、404、612、1236 行；受控数据可运行但应替换为 `ast.literal_eval`。评估 prompt (`data.py:623-625`) 与 SFT prompt (`416-417`) 的措辞不同，可能降低复现效果；应统一评估协议，并分别标记“原样复现”与“协议校正”，绝不在 E1–E4 混用。

## 文档与实现边界

本阶段没有下载权重、运行训练，也没有声称复现 README 图片数值。任何改动先保留 upstream commit 和 patch 清单：①共同正确性修复；②A 方法；③实验脚本。只有②在实测后可作为个人算法贡献，①是保障结论可信的工程工作。
