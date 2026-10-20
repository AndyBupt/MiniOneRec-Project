# 实验计划与评估协议

状态：预注册设计，尚无训练结果。数据、模型、时间和价格数字均不代表实验成绩。代码基准见 [源码分析](CODEBASE_ANALYSIS.md)，资源与时间见 [项目计划](PROJECT_PLAN.md)。

## 固定条件与数据审计门槛

主实验采用仓库自带 Industrial 数据和既有三层 SID；所有 E1–E4 固定同一 `*.index.json`、`*.item.json`、train/valid/test CSV、Qwen2.5 base 型号、tokenizer、prompt、历史截断长度、解码和指标代码。建议先用 1.5B base 跑通，profiling 允许时整套实验统一升至 3B。禁止 E2 用 1.5B、E3 用 3B 来宣称方法提升。任何 SFT/GRPO checkpoint 只在 valid 选型，test 仅在配置冻结后评一次；若必要的代码修复发生在中途，重跑所有受影响组。

数据审计必须输出：每个 split 的行数、用户数、item 数、时间范围；同用户历史仅含早于 target 的行为；train/valid/test target 时间单调边界；跨 split 完全相同 `(user_id, history_item_id, item_id)` 的数量；item metadata 与 SID 覆盖率；SID 冲突组；相同历史多 target。仓库转换代码的全局 8:1:1 时间切分见 `data/amazon18_data_process.py:276-339`。现有 CSV 无显式 timestamp 列，若无法从原始 `.inter` 或原数据证明每条历史严格早于 target，明确标注限制，并使用现有 split 不重新混排。text encoder 只接触 item metadata，不能接触 test interaction label；transductive catalog 设置写在报告中。

**奖励正确性共同补丁**：`rl.py` 目前用覆盖性的 `history2target` 字典，并把 eval 数据映射合并进来；每条训练样本应携带不可混淆的 target SID（以及可选 item id），`ReReTrainer` 将它复制到 G 个 rollout，reward 直接按行取。该补丁在 E2–E4 一致，记为协议修复，不算 A 方法。修复前的 upstream 数值只能列诊断栏。`mask_all_zero` 当前未接线，不能直接拿它做消融。

## 主实验矩阵

| ID | 初始化 | 训练/算法 | 核心目的 | 对照控制 |
|---|---|---|---|---|
| E1：SFT Baseline | Qwen2.5 base | 原仓库三任务 SFT；记录实际 LoRA/全参、epoch/token 数 | 提供 RL 共同起点和可用 Top-K | 所有 RL 组从同一 E1 checkpoint 出发。 |
| E2：原版 GRPO | E1 checkpoint | 原版 `ranking` reward + 原版组标准化/KL/采样设置，仅共同协议修复 | 测 RL 本身的收益及 0 方差问题 | 与 E3 相同训练 prompt 子集、G、β、学习率、步数、随机种子、解码。 |
| E3：条件 CE 锚定 | E1 checkpoint | E2 + 零方差组条件 CE，`λ` 在 valid 选 | 检验能否把无偏好信号的组用于稳定学习 | 额外 CE 的计算耗时单独报；训练预算和 rollout 次数相同。 |
| E4：消融 | E1 checkpoint | ① λ=0（应等价 E2）；② 固定 CE 不门控；③ 只对全零组门控；④ λ 小/中/大；可加 `rule` 对照 | 识别门控、CE 权重、reward 选择各自作用 | 一次只改一个因素；完整 test 只给 valid 选中的代表组。 |

若要比较 `add_gt` 或 HEPO，列为**原仓库已有功能的额外基线**，不写入 E3 个人贡献。`add_gt` 会改变 rollout 分布，不能作为完全等价的对照；它主要检验锚定项相对现有稀疏奖励缓解办法的价值。

## A 方法定义与实现验收

每个 prompt 从策略采样 G 个合法 SID，原版奖励 `r_{g,i}` 不变。令 `A_{g,i}=(r_{g,i}-mean(r_g))/(std(r_g)+1e-4)`；令 `z_g=1[std(r_g)<ε]`，推荐先 `ε=1e-6`，并把 all-zero 与同一正值组分别统计。设 `CE_g` 为当前样本真实目标 SID 在当前 policy 下的 token 平均负对数似然（prompt token 不计），则：

`L_total = L_repo_GRPO + λ · (Σ_g z_g CE_g)/(Σ_g z_g + 1e-8)`。

若没有选中组，该项定义为 0。实现要避免 `.item()` 后丢失计算图；目标 token 用 SFT 同样的 tokenizer 和 EOS 约定。`z_g` 为 detached 判定，不对 reward 求梯度。为避免条件 CE 权重随零组比例变化而不可解释，同时记录每 step 激活比例和 CE 项原始/加权大小；必要时把分母改为总组数的备选形式作为敏感性分析，并明确这两种目标不同。梯度累积下组必须完整落在同一采样批次，`batch_size % G == 0`。

验收检查：同 reward 组原版 advantage 全 0；非同 reward 组 E3 的附加项为 0；每条样本自己的 GT 正确传至 rollout；条件 CE 仅对目标响应 token 求损失；λ=0 数值等价 E2；合法 SID 前缀约束在 G=4、8 与不同 batch 下无效比例可测。短时间单元/小样本检查只针对这些真实风险。

## 指标、统计与显著性

主指标：`HR@5/10`、`NDCG@5/10`。每个用户查询给有序且去重的 Top-10，目标 SID 第一个命中位置为 r，`HR@K=1[r<K]`，`NDCG@K=1[r<K]/log2(r+2)`；对全体查询取平均。主口径使用合法 SID 命中，因为当前信息文件以 SID 解码；补充：无效 SID 比例（全 K 候选分母）、重复 SID 比例、碰撞目标查询比例、可判别 item 级命中。碰撞 SID 如果不能唯一映射 item，不能算 item 级正确；可报告严格下界（只认唯一 SID）和宽松上界（碰撞 SID 视可能正确），并保持各组同口径。不要用 `calc.py` 的 CC 直接代替无效比例。

训练诊断：reward 均值/标准差、all-zero 组率、零方差组率、exact-hit rollout 率、组内候选去重率、KL、CE 项、loss、梯度范数、每 step 时长、生成 tokens/s、peak allocated/reserved VRAM、总 GPU 小时。评估另报全量 test 解码耗时和峰值。候选列表保存匿名样本 id、GT、预测 SID 与 rank，便于错误案例复核。

先用固定 valid 子集调参数和做 profiling，最后 valid 全集选 checkpoint；对最终 E1–E3 同一 test 集做 paired bootstrap（按用户分组抽样，至少 1,000 次）给 HR/NDCG 差值 95% 区间。资源允许做 3 seed，至少做 2 seed 并呈现均值、范围；单 seed 结果必须明确不确定性。用户数未知前不能保证用户级 bootstrap 足够稳定。若变化小于噪声或成本显著上升，结论如实写“未见可靠增益”。

## 公平预算和训练约束

- E2/E3 固定 SFT 权重、RL 训练行、抽样顺序/seed、G、max prompt/completion、学习率、β、优化器、累积、训练步数。E3 多做一次 GT forward，**相同步数**是主对照，另报**相同 GPU 小时**的效率对照，避免只看效果不看成本。
- 原版 `rl.sh` 为 8 卡、G=16、train batch=64；单卡试 `G=4`、每设备 batch=4、accum=4–8、BF16、completion 16–32、prompt 256–512。先以 100–500 个样本 smoke，确认最长历史截断不会删除响应/标签；再扩大到固定 RL 子集。若单卡仍 OOM，优先 LoRA 与 gradient checkpointing，减 batch/G，必要时改 1.5B；不得在 E2/E3 之间单独变化。
- 用 PyTorch SDPA 起步，避开 `rl.py:70-71` 主动关闭高效注意力的默认设置；先做正确性再测速。自定义 trainer 与所选 TRL/Transformers 版本接口逐项测试，不能靠安装成功判断可运行。
- 文件记录 `git commit`、环境 lock、数据哈希、配置 YAML、seed、启动时间、结束时间、崩溃日志、checkpoint hash。评估脚本必须固定 `do_sample=False` 和同一 `num_beams≥10`，不能对不同组使用不同 decoding。

## 结果表模板（待实测）

| 组 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | invalid@10 | zero-var 组率 | GPU h | 峰值 GB | seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 | 待测 | 待测 | 待测 | 待测 | 待测 | 不适用 | 待测 | 待测 | 待定 |
| E2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待定 |
| E3 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待定 |
| E4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待定 |

## 失败时的解释顺序

先检查 SID collision、prompt/constraint 正确性、同历史不同 GT、valid/test 协议；再看 E2 的 rollout 命中与零方差组率；随后看 E3 的 CE/GRPO/KL 比例、训练步数及是否过拟合高频 item；最后分析模型容量、SID 质量与数据噪声。不得因 E3 未提升而修改 test 集、删除困难样本或只汇报最优 seed。
