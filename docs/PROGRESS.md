# 项目进度与实验日志

## 当前状态（2026-10-13）

| 项目 | 状态 | 证据/下一步 |
|---|---|---|
| 仓库获取 | 完成 | 从 `https://github.com/AkaliKong/MiniOneRec` clone；基准 commit `5f4f733cfda387f7f6f8f158db0d300eed9471c6`。clone 前工作目录为空，未覆盖已有文件。 |
| 源码审查 | 完成首轮 | README、SFT/RL/HEPO、数据、RQ-VAE、约束解码与指标已审；细节见 `CODEBASE_ANALYSIS.md`。 |
| 自带数据静态审计 | 完成初步 | Industrial 36,259/4,532/4,533 train/valid/test 行；3,686 item 对 3,670 唯一 SID。Office 38,924/4,866/4,866；3,459 对 3,444。未做时间戳/用户级全量核查。 |
| 资源可行性 | 纸面估算 | 单卡 96GB 的 1.5B 主线可行性较高，3B 需 profile；本机没有租用 GPU，尚无实测吞吐/显存。 |
| 模型权重/大型数据下载 | 未开始 | 等待用户确认资源需求。仓库 clone 已带约 91MB `data/`，无额外数据下载。 |
| 训练/评估 | 未开始 | E1–E4 结果均为空；不得在简历中声称已经取得指标。 |
| 本地学习入口 | 已准备 | 新增 `docs/LEARNING_RUNBOOK.md`、只读 `scripts/inspect_data.py` 与独立 `scripts/sft_smoke_single_gpu.sh`；数据检查已在本地执行，shell/Python 语法已检查；SFT 脚本尚未在 GPU 上运行。 |

## 已发现问题（按优先级）

| 优先级 | 问题 | 影响 | 处理记录 |
|---|---|---|---|
| P0 | `rl.py:120-131` 合并 train/eval 的 `history2target`；`data.py:386-388` 把相同历史压为单个目标 | 奖励标签可能错配或引用 valid target | 在 E2–E4 共同补丁中改成样本自身 target，不将 eval label 用于训练 reward。 |
| P0 | `calc.py:62-65` 的 CC 不是完整无效 SID 比例；SID 有碰撞 | 指标可能误读 | 新增完整 invalid@K、碰撞口径及 SID/item 区分。 |
| P0 | `requirements.txt` 锁 PyTorch 2.6/cu118/cu124 混合，Blackwell 需要兼容 wheel | 租卡后可能无法跑 | 用官方 PyTorch CUDA 12.8+ 版本建环境，先 smoke。 |
| P0 | `rl.py:136-138` 预加载 LLM，trainer 又创建 policy/reference | `ranking` reward 下第三份权重占满单卡显存 | 在共同工程补丁中移除无用预加载；所有 E2–E4 同一处理。 |
| P0 | SFT 与 eval 的序列 prompt 措辞不同（`data.py:416` vs `623`） | 复现/公平对照风险 | 固定统一评估 prompt，分别记录 upstream 与校正协议。 |
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

下一阶段在当前本地仓库可继续做用户/时间/交互审计和生成精确可运行脚本。下载 Qwen 1.5B/3B 权重、额外大型数据或租用 RTX PRO 6000 之前，先按 `PROJECT_PLAN.md` 的预算与 profiling 设计向用户确认。当前没有花费租卡费用。
