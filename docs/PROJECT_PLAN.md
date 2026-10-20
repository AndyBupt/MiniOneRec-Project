# MiniOneRec 单卡生成式推荐项目计划

> 状态：规划完成，训练未启动。审查基准为 `5f4f733cfda387f7f6f8f158db0d300eed9471c6`（2026-10-10）。本文中的显存、耗时、费用均为预算假设，必须用租卡后的短时 profiling 校准。

## 项目目标与验收

在同一数据、SID、Qwen2.5 基座和评估协议下完成：SID 构建或复核 → SFT → 原版目标的 GRPO → 独立改进 → Top-K 评估。验收物是可复现实验配置、数据校验报告、E1–E4 指标及曲线、代码差异、失败案例和能准确区分原版与个人贡献的面试材料。没有实测结果前，简历不写提升数字。

主数据建议使用仓库自带 `Industrial_and_Scientific`，因为已有 3,686 个 item 的 SID、约 3.6 万条训练样本和固定 valid/test；`Office_Products`（3,459 个 item）作 P1 复核。仓库自带 embedding、索引和 CSV，不需另下载大型数据即可启动规划与数据审计。先用现有 SID 做训练复现，再用仓库 embedding 重训一次 RQ-VAE 以覆盖完整技术链；是否重新生成 text embedding 取决于时间及授权资源。现有 SID 并非无碰撞：Industrial 有 16 个额外碰撞项，Office 有 15 个，详见 [源码分析](CODEBASE_ANALYSIS.md)。

## 三个方向的决策

| 方向 | 原理与问题 | 与原仓库的边界 | 修改点 | 难度/工时（人时） | 实验和指标 | 面试深度 |
|---|---|---|---|---|---|---|
| A. 零信号感知的 GRPO | 精确命中奖励下，一个组的奖励方差为 0 时，其偏好梯度消失；对这类组加入仅作用于真实目标的条件 SFT 锚定项，保留有信号组的原版 GRPO 与 KL。可测 0 方差组占比、梯度/稳定性与 Top-K。 | 原版已有 binary+rank reward、`add_gt`、`dynamic_sampling`、HEPO 前缀奖励；本方案**不**把 GT 塞进 rollout，也不把前缀匹配或排序奖励当新方法。 | `rl.py` 配置与标签传递；`minionerec_trainer.py` `_prepare_inputs`/`compute_loss` 加按组门控的 CE；训练日志。 | 中，约 18–30 h（含验证） | E2 对比 E3；E4 分别关门控、固定权重、不同 λ；HR/NDCG、0 方差组、KL、耗时/显存。 | 高：奖励稀疏、on-policy 采样、目标偏差、KL 与监督锚定的权衡。 |
| B. SID 碰撞与码本利用率 | 对已有 RQ-VAE 的码本占用、碰撞、语义邻域一致性做诊断，尝试少量码本/层数设置；SID 唯一性影响 item 级指标。 | 原版已有 Sinkhorn、碰撞率 checkpoint、迭代冲突修复、constrained RQ-Kmeans 和 RQ-Kmeans+；不能把“解决碰撞”或“平衡码本”笼统算创新。 | `rq/rqvae.py`、`rq/generate_indices.py`、SID 诊断脚本、转换与评估。 | 中高，约 25–45 h，且重训 SFT 才能做端到端因果比较 | 码本 perplexity/使用率、碰撞率、重构误差、邻域保持率、item 级 HR/NDCG。 | 高：残差量化、码本塌缩、容量与语义保真度；但 2–4 周风险较高。 |
| C. SFT/RL 训练策略 | 控制 SFT 中 SID 预测与 item 对齐任务的采样比例，再比较 LoRA/全参及 GRPO 的 G、β、温度；解决任务比例和成本问题。 | 原版已有三种 SFT 任务、冻结 LLM、新词训练、VAFT、DAPO/GSPO 开关。不能将任务混合或 LoRA 本身算新方法。 | `sft.py` 的数据采样、训练参数；`rl.py` 配置；评估日志。 | 中，约 18–35 h | 固定 token/step 预算下比较 HR/NDCG、训练稳定性、时间/显存；消融辅助任务比例。 | 中高：多任务迁移、预算公平性、模型容量；但个人优化叙事可能偏工程调参。 |

**推荐 A**。其切口直接来自源码中按组标准化的结构与数据稀疏性，可以保持同一个 SFT checkpoint、SID 和评估集合，只改变 RL 阶段，因果对照清楚。实施前必须先修复/隔离奖励目标映射和 SID 碰撞带来的评估歧义；这些是实验正确性工作，不单独包装为优化贡献。如果短时 profiling 显示原版 GRPO 无法在单卡完成，则用 1.5B LoRA、G=4、短 completion、少量训练样本完成同一目标；若仍不可行，保留 SFT 与离线奖励/候选分析，不把它称作“完整 GRPO”。

### A 的预注册实现规则

- 原版奖励保持 `rule` 或仓库 `ranking` 公式不变。候选建议先用 `ranking`（与 `rl.sh` 一致），并记录每组标准差、命中率。
- 对组 `g`，令 `z_g=1[std(r_g)<ε]`；仅在 `z_g=1` 时对该 prompt 的真实目标计算 token 平均 CE。建议 `L=L_GRPO+λ·mean_g(z_g·CE_g)`；λ 从 0.05、0.1、0.2 进行验证集选择，不能用 test 调参。可单独记录“全零组”与“全相同正奖励组”。
- CE 的 GT 来自当前样本的 `completion` 列，不能通过 `history2target` 字典反查。多个相同历史对应不同 target 时，每条样本仍保留其自身标签。必要的数据传递修复应用到 E2/E3/E4 的共同底座。
- 该辅助 CE 会引入监督偏置，提升不保证发生；报告 0 方差组率、Top-K 与训练成本三者的真实变化。

## 单卡 96GB 资源规划

RTX PRO 6000 Blackwell Workstation 版是 96GB GDDR7；租用页面须核对具体型号、驱动、显存和价格。[NVIDIA 规格](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000/)；[AutoDL 首页价格展示](https://autodl.com/home) 曾显示 PRO 6000 96GB 约 ¥6.98/小时，实际下单价、地区、磁盘费需重新确认。

| 模式 | 参数状态与静态显存粗估 | 96GB 判断 | 起步设置 |
|---|---|---|---|
| 1.5B SFT 全参 | BF16 参数+梯度+FP32 Adam 主权重/状态约 16 B/参数，约 25GB；另需激活、logits、CUDA 工作区 | 较稳 | micro-batch 1–2、accum 16–32、BF16、checkpointing、长度 512 起 |
| 3B SFT 全参 | 同算法约 49GB 静态，加激活可能达到 65–85GB | 可试，必须 profile | micro-batch 1、accum 16–32、长度 512；OOM 切 LoRA |
| 1.5B/3B GRPO 全参 | 上述优化器内存 + BF16 reference 约 3/6GB + G 路 rollout/KV/logits；生成峰值依赖 prompt 长度和 G | 1.5B 可试；3B 风险高 | 不用官方 G=16/批量 64；先 G=4、per-device batch=4、accum=4–8、completion 16–32 |
| 1.5B/3B GRPO LoRA | 基座及 reference 约 2 份 BF16 权重（若 PEFT 正确复用可更低），adapter 状态很小；激活和 rollout 主导 | 首选 3B 的 RL 路线 | 先 1.5B 跑通，再决定 3B；G=4 或 8 |
| 7B 扩展 | 全参 Adam 静态约 112GB，超过卡容量；LoRA 双模型权重约 28GB，另有激活/KV | 仅 LoRA 试验，P2 | 压短长度、G=4、4-bit 量化需单独验证，不作为主线 |

估算只计算模型量级，具体取决于 Qwen 词表、优化器实现、attention backend、批量与 prompt 分布。[Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) 为 1.54B，[Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) 为 3.09B。优先使用 1.5B base 保证 E1–E4，3B 是在通过 profiling 后升级的同一方案；不要在实验组之间混用模型尺寸。

### 环境门槛与短时 profiling

仓库 `requirements.txt` 钉死 PyTorch 2.6.0，且混入 CUDA 11.8/12.4 包；不能原样用于 Blackwell。PyTorch 官方从 2.7 起提供 Blackwell/CUDA 12.8 wheel；CUDA 12.8 对 Linux 驱动有最低要求（按具体版本确认），详见 [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/) 与 [NVIDIA CUDA 12.8 发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)。FlashAttention-2 官方 CUDA 支持列表未列 Blackwell；先用 PyTorch SDPA，只有实测兼容与收益才选装 FlashAttention，并注意 `rl.py:70-71` 目前主动关闭 flash/memory-efficient SDP。[FlashAttention 官方仓库](https://github.com/Dao-AILab/flash-attention)。

单卡前置修复还包括移除 `rl.py:136-138` 对 `ranking` reward 无用的整模型预加载；否则 trainer 再建 policy/reference 时会驻留第三份权重，使上表静态显存估算失真。E2–E4 都使用同一修复。

租卡后先运行 30–60 分钟诊断，不做完整训练：`nvidia-smi` 记录型号/驱动；`torch.cuda.get_device_capability()`、`torch.cuda.get_arch_list()` 验证 wheel；BF16 GEMM+反向、SDPA 前反向、Qwen tokenizer/SID 单 token、100 个 SFT step、20 个 GRPO step、50 个 valid 查询的 Top-10 解码。记录 `max_memory_allocated/reserved`、step 秒数、生成 tokens/s、无效 SID 比例。冻结镜像、Python、torch、transformers、trl、accelerate、bitsandbytes/peft 版本及 commit。先按仓库已锁的 transformers/trl API 验证自定义 trainer，再解决版本冲突；不要假设新版本自动兼容。

以 40–100 GPU 小时作为**暂定预算带**，按 ¥6.98/h 是约 ¥279–698，未计存储与停机闲置；其中环境/数据 3–8h、SFT 8–25h、原版 GRPO 10–30h、改进和消融 15–35h、评估 4–10h。它不是训练时间承诺。真实预算 `总小时=各阶段样本数×步/样本实测值 + 评估/重跑`，在 profiling 后更新，并由用户决定是否下载权重与租卡。

## 执行日程与优先级

| 时间 | 优先级 | 可核验结果 |
|---|---|---|
| D1–2 | P0 | 固定代码 commit、数据版本及 train/valid/test 清单；核查历史时间顺序、SID/Item 一对一性、重复历史多目标、重叠样本；确定评估主口径。 |
| D3–4 | P0 | 建 Blackwell 兼容环境并做上述 smoke/profile；确定 1.5B/3B、全参或 LoRA、最大长度、批量和租卡预算。此步涉及下载/租卡，须先获用户确认。 |
| D5–7 | P0 | 用现有 SID 跑 SFT E1，记录验证集 HR/NDCG 与耗时；复核 RQ-VAE 源码，能在已有 embedding 上小规模重训并报告码本/碰撞。 |
| D8–10 | P0 | 修复所有组共用的目标映射/评估协议；用相同 SFT checkpoint 启动 E2 原版奖励目标 GRPO，记录 0 方差组、KL 和 rollout 样本。 |
| D11–14 | P0 | 完成 E1/E2 的统一 valid/test 离线评估与训练记录；若 RL 卡住，收敛到 1.5B LoRA、G=4、短序列与固定训练子集。 |
| D15–18 | P0 | 实现条件 CE 锚定 E3；加入组级统计和单元/小数据检查；从 E1 同一 checkpoint 初始化。 |
| D19–22 | P0 | E3 与 E4 消融：λ=0、条件门控、固定 CE、不同 λ；同等 RL 样本/步数/候选数与测试协议。 |
| D23–25 | P1 | 固定选定超参，至少 2–3 个随机种子或可负担的重复；Office 作为外部数据复核；分析失败案例及训练成本。 |
| D26–28 | P0 | 整理脚本、环境锁、数据审计与图表；更新真实结果、面试材料和简历措辞。 |

P2：7B LoRA、重新生成 text embedding、第二种 SID 方法、更多数据集。P0 未达标时不启动 P2。任何无法跑完的阶段都明确标为“未完成/未验证”，不推断效果。

## 当前资源决定点

下一阶段需要用户确认：租用单卡的实际价格上限、允许下载约 3–7GB/模型的 Qwen 权重及是否使用现有 Amazon 数据作为主数据。确认前仅做本地代码/数据审计及规划。
