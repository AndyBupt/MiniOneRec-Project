# 项目进度与实验日志

## 当前状态（2026-10-21）

| 项目 | 状态 | 证据/下一步 |
|---|---|---|
| 仓库获取 | 完成 | 从 `https://github.com/AkaliKong/MiniOneRec` clone；基准 commit `5f4f733cfda387f7f6f8f158db0d300eed9471c6`。clone 前工作目录为空，未覆盖已有文件。 |
| 源码审查 | 完成首轮 | README、SFT/RL/HEPO、数据、RQ-VAE、约束解码与指标已审；细节见 `CODEBASE_ANALYSIS.md`。 |
| 自带数据静态审计 | 完成初步 | Industrial 36,259/4,532/4,533 train/valid/test 行；3,686 item 对 3,670 唯一 SID。Office 38,924/4,866/4,866；3,459 对 3,444。未做时间戳/用户级全量核查。 |
| 资源可行性 | A800 SFT 已实测 | A800-SXM4-80GB；全量 SFT 训练加保存 11,263.16 秒，CUDA peak allocated/reserved 14.42/16.35 GiB。该结果不代表 GRPO 的显存需求。 |
| 模型权重/大型数据下载 | 已用于 SFT | 基座路径 `/root/autodl-tmp/models/Qwen2.5`；版本 `8faed761d45a263340a0528343f099c05c9a4323`。用户已提供训练正常结束与最终 2.9GiB 权重文件的证据；最终模型重载仍待验证。 |
| 训练/评估 | E1 训练完成，推荐评估待做 | 用户日志确认 3 epochs、3744 steps、COMPLETED；train loss 1.0992、best valid loss 2.8872、best checkpoint 3572。HR/NDCG 仍为空。详见 `experiments/E1_SFT_20261020.md`。 |
| 本地学习入口 | 已准备 | 新增 `docs/LEARNING_RUNBOOK.md`、只读 `scripts/inspect_data.py` 与独立 `scripts/sft_smoke_single_gpu.sh`；数据检查已在本地执行，shell/Python 语法已检查；SFT 脚本尚未在 GPU 上运行。 |
| 训练产物本地备份 | 已下载并静态核验 | `/Users/andy/研二/rec模型权重/e1_sft_backup.tar` 为 2.908 GiB，含最终模型/tokenizer/运行配置/日志。数据哈希与本地一致，560 个 SID token 完整，模型头部和词表大小匹配；尚未做 GPU 重载和推荐评估。 |
| E1 推荐评估 | 完整 valid 已完成 | 4532 条 SID HR@5/10=0.038173/0.055384，NDCG@5/10=0.035078/0.040476；invalid/missing/duplicate 均 0。评估 548.58 秒，A800 CUDA allocated/reserved 3.00/3.10 GiB。238 条目标 SID 碰撞；详见 `experiments/E1_VALID_20261021.md`。 |
| 简单参照与诊断 | 预测明细已审计 | 训练集热门 SID HR@5/10=0.046778/0.056708；最近交互 SID+热门补齐为 0.075243/0.082966。当前 SFT 尚未超过参照；4532 条样本与 CSV 对齐且指标复算一致。全体 Top-10 仅覆盖 28 个 SID，Top-1 仅 4 个。详见预测审计 JSON 和实验记录。 |

## 已发现问题（按优先级）

| 优先级 | 问题 | 影响 | 处理记录 |
|---|---|---|---|
| P0 | `rl.py:120-131` 合并 train/eval 的 `history2target`；`data.py:386-388` 把相同历史压为单个目标 | 奖励标签可能错配或引用 valid target | 在 E2–E4 共同补丁中改成样本自身 target，不将 eval label 用于训练 reward。 |
| P0 | `calc.py:62-65` 的 CC 不是完整无效 SID 比例；SID 有碰撞 | 指标可能误读 | 新增完整 invalid@K、碰撞口径及 SID/item 区分。 |
| P0 | `requirements.txt` 锁 PyTorch 2.6/cu118/cu124 混合，Blackwell 需要兼容 wheel | 租卡后可能无法跑 | 用官方 PyTorch CUDA 12.8+ 版本建环境，先 smoke。 |
| P0 | `rl.py:136-138` 预加载 LLM，trainer 又创建 policy/reference | `ranking` reward 下第三份权重占满单卡显存 | 在共同工程补丁中移除无用预加载；所有 E2–E4 同一处理。 |
| P0 | SFT 与 eval 的序列 prompt 措辞不同（原版 `data.py:416` vs `623`） | 复现/公平对照风险 | 2026-10-20 已对齐；用真实 Dataset 方法验证普通/重复交互样本的 train/eval prompt 一致，且原 SFT 输入及 loss mask 不变。 |
| P1 | `rl.py:58` 的 `mask_all_zero` 仅形参未接线；`rl.py:70-71` 关闭高效 SDP | 开关失效/性能浪费 | 不把该参数列为有效方法；profile 后做正确性和性能修复。 |
| P1 | `rq/generate_indices.py:44-49` 路径硬编码 | 无法直接重训 SID 导出 | 后续参数化，保留原版逻辑；单独标记工程修复。 |

## 实验记录模板

每次运行复制一段，并保留原始日志与配置：

```
日期/运行 ID：
阶段与实验组：
代码 commit / patch：
数据文件 hash / split 版本 / SID hash：
模型与 tokenizer / 权重来源：
Python、PyTorch、CUDA、Transformers、TRL、驱动：
配置（batch、accum、G、β、λ、LR、长度、seed）：
训练样本数 / step / wall time / GPU hours / 峰值 allocated 与 reserved：
valid HR@5/10、NDCG@5/10、invalid@10：
test 指标（最终冻结后填写）：
zero-var/all-zero 组率、KL、奖励分布：
异常、修复、是否需要重跑对照：
结论及证据路径：
```

## 下一行动与资源门槛

用户已在克隆的 A800 实例完成 full-valid 评估并下载指标。助手读取本地结果、记录真实 loss/推荐指标/耗时/显存并在 CPU 上补算两个参照，未直接连接服务器。完整 predictions.json 已做对齐、集中度和重复交互分桶诊断；下一步核查全部 SID 的解码可达性及基座到 SFT 的参数更新，再决定是否调整 SFT。Test、GRPO 修复及 E2–E4 尚未开展；不宣称现有 SFT 已取得算法收益。


## 2026-10-21 SFT 诊断更新

修正后的只读诊断已在用户服务器运行：3670 个目录 SID 全路径可达，100 项真实 processor CPU 掩码检查通过。三处主干权重抽样均有更新，最终 norm 无变化，尚未确诊数值精度问题。下一步用现有模型在 train 抽样 256 条生成推荐以检查拟合情况；不将 train 指标作为泛化结果，不启动完整重训或 GRPO。


## 2026-10-21 train-256 拟合诊断

用户已运行并提供 256 条 train 抽样结果：逐条对齐和指标复算通过，HR@5/10=14/256、16/256，NDCG@5/10=0.041444/0.044048。所有 Top-10 合计仅 24 个 SID，Top-1 仅 4 个，最常见占 67.58%。候选集中并非只出现在 valid；下一步设计固定小样本的拟合与参数精度对照，尚未启动。详见 `experiments/E1_TRAIN256_20261021.md`。


## 2026-10-21 tiny-fit 脚本交付

已准备 `scripts/tiny_sft_diagnostic.py` 与 `scripts/tiny_sft_diagnostic_single_gpu.sh`，从同一 E1 独立运行 BF16/FP32 参数组；128 条无冲突历史抽样实际对应 125 个目标 SID，仅 next-SID 任务。含初始生成评估、目标 mask/prompt 审计、teacher-forced 分层指标、第10步资源门槛、独立 BF16 导出和共同生成评估。实际数据抽样、语法与3项 CPU 单测已通过；GPU 训练待用户服务器执行，未产生新效果结论。完整说明见 `TINY_SFT_DIAGNOSTIC.md`。


## 2026-10-21 诊断与调优过程复盘

新增 `SFT_TROUBLESHOOTING_LOG.md`，按证据串联首次 SFT、valid 评估、简单参照、SID 碰撞与预测集中、解码/参数检查、train-256 和 tiny-fit 对照，区分已确认事实与待验证假设。用户反馈 128 条短程对照正在运行；尚未收到 profile/comparison，未记录调优收益。


## 2026-10-21 tiny-fit 对照结果

已收到用户 `comparison.json` 并保存原始副本。两组报告的起点、样本哈希和训练超参一致；同一128条train样本的初始/BF16/FP32 HR@10为5/128、34/128、44/128，候选不同SID数24/117/133。最终目标NLL为BF16 2.1485、FP32 1.9970；FP32训练149.07秒、allocated28.83GiB，BF16为131.78秒、14.41GiB。支持参数精度配置影响小样本拟合，不代表完整valid收益或唯一根因。完整三任务FP32参数SFT对照尚未启动；详见 `experiments/TINY_SFT128_20261021.md`，总复盘已更新。


## 2026-10-21 完整FP32参数SFT对照脚本

已交付 `scripts/sft_fp32_full_single_gpu.sh`：原Base重新初始化、原三任务与原超参，扩SID后再提升参数到FP32；第10步资源profile、模型单独保存、自动完整valid评估及不含权重的review_bundle。增加eval对FP32分片权重的哈希支持。至少35GiB空闲磁盘；此轮不保存optimizer，不能精确断点续训。语法、profile资源门槛及单文件/分片哈希CPU测试通过，实际GPU训练尚未启动。详见 `SFT_FP32_FULL_RUN.md`。
