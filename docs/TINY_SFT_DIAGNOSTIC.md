# 128 条样本拟合与参数精度对照

## 实验目的

从已完成的 E1 SFT checkpoint 出发，检查固定小批训练样本能否被进一步学会，以及参数存储/更新精度是否影响学习。128 指历史—目标样本，不是 128 个物品。此次固定抽样实际包含 125 个不同目标 SID。

这不是正式效果优化、完整训练或泛化评估。两组均不包含 SID↔标题及历史→标题任务；不读取 valid/test 标签；不修改原始 E1。

## 自动执行的流程

1. 读取原 train CSV，以历史 SID 序列为 key；全 train 中同一历史对应多个 SID 标签的历史全部排除。其余每个历史保留第一条代表，seed=42 抽 128 条。
2. 保存 `samples.csv` 与 `samples_manifest.json`，包含原 train 行号、哈希和选择规则。实际本地检查：30,016 个不同历史中 1,550 个有冲突；28,466 个符合条件。
3. 先用原 E1 在这 128 条生成 Top-10，保存 `initial/metrics.json`。不能直接用此前 train-256 充当这次的初始指标。
4. BF16 参数组与 FP32 参数组分别从同一个 E1 权重独立开始，优化器从零初始化，不串行接续模型。两组均全参数训练，仅使用 next-SID 任务，BF16 autocast，AdamW，LR=2e-5，constant LR，无 warmup，100 updates，micro batch2、累积16、有效 batch32，seed42，无 early stopping。
5. 使用实际 `SidSFTDataset` 与 `EvalSidDataset`，逐条验证训练/推理 prompt 相同；prompt 的 label=-100；三个 SID token、换行与 EOS 均有正确监督，未因截断丢失。发现异常立即停止。
6. 两组分别计算训练前后目标 token NLL、teacher-forced a/b/c top1/top10 准确率；检查部分主干、完整最终 norm 和本次目标 SID embedding 行的参数变化。teacher-forced 的 b/c 使用真实前缀，不等于完整生成准确率。
7. 两组均导出 BF16 checkpoint，然后通过现有 `evaluate.py` 以相同 BF16 推理、beam10、原始 SID 排名计算 HR/NDCG/invalid、候选覆盖。FP32 组的内存训练指标与 BF16 导出后的生成指标需区分。
8. 汇总 `comparison.json`。没有保存 optimizer checkpoint，两个导出不能用于精确断点续训。

## 资源与停止门槛

- 目标环境：现有单卡 A800 80GB、现有 conda base；不安装新包、不下载模型。
- 额外磁盘：两个 BF16 模型导出约 6 GiB，加日志；要求开始前至少 10 GiB 空闲。
- FP32 参数 + FP32 Adam 状态与梯度约 24 GiB（1.5B×16 bytes 的粗估），另需激活与临时张量。不是峰值保证。
- 每组第 10 个 update 保存 `profile_step10.json`：实测训练耗时、峰值 allocated/reserved、实际 optimizer 状态 dtype。
- 如 reserved 超 60 GiB，或按前 10 步估计该组训练超过 1,800 秒，脚本报错停止。OOM 也会停止，不自动重试、改配置或覆盖结果。
- 预估总耗时约 10–30 分钟，待实际 profile 修正；并非时限承诺。第 10 步门槛是早期估算，不能保证其后耗时/显存上界。

## 用户运行

```bash
cd /root/autodl-tmp/MiniOneRec-Project
git pull --ff-only
```

更新成功后运行：

```bash
CHECKPOINT_PATH="$PWD/output/sft_e1_Industrial_and_Scientific_20261020T101253Z/final_checkpoint" \
bash scripts/tiny_sft_diagnostic_single_gpu.sh
```

默认新建带时间戳的目录，终端会显示 `Diagnostic output: ...`。运行时保持终端连接。不要同时运行其他 GPU 任务。最后应出现 `COMPLETED: .../comparison.json`；报错即停止并回传错误与已生成的日志，不直接启动正式训练。

完成后只需下载 `comparison.json`，其中包含三组推荐指标和两组训练汇总；如有异常，再下载 `*_training.log`、`profile_step10.json`。无需下载两个 `final_checkpoint`，也不要重新打包所有模型。样本文件、环境、代码/输入哈希、每组预测和日志均留在输出目录。

## 结果解读

若 FP32 组在相同预算下更好地降低目标 NLL、提高生成命中及参数变化，支持进一步检查精度/更新的假设，但仍不能直接断言完整 SFT 的问题已解决。若两组都能拟合，重点回到完整数据与混合任务配置；若两组都未拟合，100 步未必足够，也应继续查梯度、目标与学习率，不能立即归因模型容量不足。

只重复 128 条容易产生记忆，不将指标写作验证集收益或个人方法提升。不改动 E1/E2–E4 的正式实验协议。

## 验证状态

本地已执行实际 train 抽样、Python/shell 语法检查，以及冲突历史排除、重复历史去重、固定种子、loss mask 与 prompt 对齐拒绝逻辑的 CPU 单测。当前本地无 CUDA/Transformers 训练环境，真实 GPU 训练、teacher-forced 张量计算和模型导出等待用户服务器执行，尚无本实验训练结果。
