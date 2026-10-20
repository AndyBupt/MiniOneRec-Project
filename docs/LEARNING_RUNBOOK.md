# MiniOneRec 阅读与首跑路线

本文件用于先把数据流和调用关系看懂，再做单卡小样本检查。当前只准备脚本；尚未下载模型、租 GPU 或启动训练。

## 现在先改什么

| 顺序 | 文件 | 怎么处理 | 原因 |
|---|---|---|---|
| 1 | `scripts/inspect_data.py`、`scripts/sft_smoke_single_gpu.sh` | 已新增只读数据检查和独立的单卡 SFT 小样本入口；不要直接改动官方 `sft.sh` | 官方 `sft.sh` 固定 8 卡与大 batch。先验证环境、数据、前向/反向、保存。 |
| 2 | `data.py` 中 `SidSFTDataset` 与 `EvalSidDataset` | 正式 E1 前统一推荐任务的指令和用户历史模板；先记录原始模板及两者差异 | SFT 与评估当前用不同措辞，同一实验协议必须固定。 |
| 3 | `rl.py` 与 `minionerec_trainer.py` | 正式 E2 前让奖励函数从每条训练样本取得 target，删除依赖 `history2target` 的跨样本映射；移除 `rl.py` 中对 `rule`/`ranking` 无用的第三份 LLM 预加载；保留相同修复用于 E2–E4 | 相同历史可能对应多个目标，现有 dict 会覆盖；训练中还合并了 valid 映射。单卡显存也不应被无用模型占据。 |
| 4 | `rl.py`、`rl.sh`、`config/zero2_opt.yaml` | 在上述正确性修复和显存 profiling 后，新建单卡 RL 入口并定 `G=4/8`、小 micro batch、累积步数、BF16；不要照搬 `rl.sh` 的 8 卡配置 | GRPO 需要 policy/reference 与生成缓存，实际可行性取决于序列长和候选数。 |
| 5 | `evaluate.py`、`calc.py` | 用 valid 做调参与 smoke；在 `calc.py` 旁新增逐条 `invalid@K` 统计并核查 NDCG；最后只评一次 test | 当前 `CC` 只在命中前计无效 SID，不是完整无效率；存在 SID 碰撞。 |
| 6 | `rq/generate_indices.py` | 仅在跑通预置 SID 主线后，把硬编码 checkpoint、路径与类别改为参数 | 先理解量化链路，之后才做重新训练与导出。 |

RL 标签修复的具体落点：`data.py:354-394` 的 `SidDataset.pre` 已把目标放在 `completion`，但训练数据构造时要在 `rl.py` 给每条样本增设明确的 `target_sid=completion`；`minionerec_trainer.py` 的 `_set_signature_columns_if_needed` 要保留 `target_sid`，`_prepare_inputs` 要直接使用各输入行的 `target_sid`；reward 函数从 trainer 已传递的额外列读取目标。这样一条历史出现多次时，每条训练样本仍有自己的标签。还要核查其他两个 RL 子任务的 `completion` 是否都是 SID，并从训练 reward 路径移除 valid 映射。这个正确性修复先于任何 GRPO 正式实验，且 E2–E4 共用。

## 按这个顺序看源码与运行

### 0. 本机，约 30 分钟：先看一条样本

运行：`python3 scripts/inspect_data.py`。

接着打开 `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv` 的前两行、`data/Amazon/index/Industrial_and_Scientific.index.json`、`data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt`。自己说清：一个 item 的原始 ID、SID、用户历史 SID、下一物品 SID 如何对应；为什么 SID 碰撞会影响 item 级指标。

### 1. 本机，约半天：理解 SID 从哪里来

依次读 `rq/rqvae.py` 的训练入口、`rq/models/rqvae.py` 的 `forward/compute_loss/get_indices`、`rq/models/rq.py` 的残差量化、`rq/models/vq.py` 的最近邻码字选择、`rq/trainer.py` 的训练与保存、`rq/generate_indices.py` 的导出。此阶段使用现成索引，不需重训 RQ-VAE。

自测：画出 `item embedding → encoder → 逐层残差量化 → decoder → <a_i><b_j><c_k>`；能说清重建损失、量化损失和碰撞。

### 2. 本机，约半天：理解一条 SFT 样本

先读 `data.py` 的 `CSVBaseDataset`、`SidSFTDataset.get_history/pre`，手工写出一条样本的完整 prompt、目标 token 和 label mask（prompt 为 `-100`）。再看 `SidItemFeatDataset` 与 `FusionSeqRecDataset`，确认三个任务如何拼接。最后看 `sft.py` 的 `TokenExtender`、`train` 中 tokenizer 扩词、模型 resize、batch、Trainer 与保存。

自测：回答模型输入是什么、真正参与 CE loss 的 token 是哪些、为什么新增 SID token、三个训练子任务的作用各是什么。

### 3. 取得 GPU 和已确认的本地模型权重后：先跑 SFT 小样本

先建立适配 Blackwell 的环境，检查 `torch.cuda.is_available()`、BF16、一次前向/反向；不要直接按 `requirements.txt` 混装 CUDA wheel。然后设置 `MODEL_PATH` 为**本地** Qwen2.5-1.5B 模型目录，运行 `bash scripts/sft_smoke_single_gpu.sh`。这个脚本每个 SFT 子任务抽 128 条、验证集抽 128 条、1 epoch、batch 1、累积到 16，目的是检查训练闭环。脚本只接受本地 `config.json`，不会自行下载权重。记录耗时、峰值显存、loss、保存文件；这不是 E1 正式基线。

### 4. GPU，小规模 valid：理解约束生成与指标

先读 `data.py` 的 `EvalSidDataset`，比较它与 SFT prompt；再读 `evaluate.py` 的 SID 候选前缀树、beam search 与结果 JSON，`LogitProcessor.py` 的合法 token 约束，`calc.py` 的 HR/NDCG。统一 prompt 后，在 valid 中固定 32 条、Top-10、batch 1 做生成检查。正式 E1 固定完整 valid 协议；最后冻结模型和配置后再用 test。

自测：对一条生成的 Top-10，手算 HR@5/10 与 NDCG@5/10，并检查所有候选 SID 是否位于索引中。

### 5. SFT 基线可靠后：理解 GRPO 再跑小样本

按 `rl.py` 的数据构造、reward、`GRPOConfig`、`ReReTrainer` 入口读；再按 `minionerec_trainer.py` 的采样、`_prepare_inputs`、reward 汇总、组内 advantage 与 `compute_loss` 读。特别检查相同 prompt/历史对应不同 target 时的行为。完成上表 P0 修复后，先用 `G=4`、短输出、少量训练样本做单卡 profiling；若内存不足，再调整有效 batch、生成数、长度或采用 LoRA/QLoRA。正式 E2 与 E3 使用相同的共同修复、数据、seed、评估协议。

自测：给定 4 个 reward，手算组均值、标准差、advantage；解释全 0 reward 组为何没有有效 GRPO 梯度，并区分源码中现有的 HEPO、`add_gt`、`dynamic_sampling` 与未来个人改进。

## 每一步留下什么

把环境、命令、commit、数据版本、模型权重、耗时、峰值显存、异常及结果写入 `docs/PROGRESS.md`。未实测的吞吐、显存和 HR/NDCG 保持为空。下载权重、租卡或正式训练前按 `docs/PROJECT_PLAN.md` 的资源门槛确认。
