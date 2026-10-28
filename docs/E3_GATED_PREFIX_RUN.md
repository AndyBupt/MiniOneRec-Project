# E3：无命中组的前缀补充信号与 Advantage 缩放

## 定位

基于E2约86.5%全零奖励组的观察，做可解释的工程优化与消融。前缀分层奖励不是原创：rl_gpr.py已有HEPO 0.2/0.5/1。本方案的差异是只对sid_next任务的无完整命中组补充信号，并在组内标准化之后控制这类组的Advantage强度。不是论文创新声明，不把既有HEPO包装为个人贡献。

## 奖励与更新

保留rule+ranking。整组存在完整命中：奖励、Advantage均保持E2定义。其他任务也保持E2定义。只有task_type=sid_next且组内无完整命中时：第一层前缀匹配0.2，前两层匹配0.5，无匹配或非法格式0。SID必须完整匹配<a_数字><b_数字><c_数字>格式，不对残缺SID发奖。

先对总reward按原公式zscore，再对此类组的Advantage乘0.1；KL项不缩放。同组前缀奖励都相同时仍无奖励梯度，所以新增nonzero_advantage_group_fraction，不只看非零reward。加上prefix_eligible_group_fraction；全零、零方差日志保留。0.1为预先选定待验证超参，不是最优值；只对一个种子结果作有限结论。

启用范围暂不包含title_history_next_sid和标题描述对齐任务，避免改变所有辅助任务；后续如扩展须独立消融。前缀相同不保证偏好相同，可能增加热门偏置或降低完整命中，必须如实报告。

## 实验

同一FP32 E1起点，数据/seed42、G8、batch8、累积8、1epoch、LR1e-6、beta0.001、固定reference、BF16计算/FP32参数与E2相同。

| GRPO_VARIANT | 配置 | 目的 |
|---|---|---|
| e2 | rule+ranking | 原有基线；默认行为不启用新奖励 |
| e3 | 门控前缀 + post-zscore乘0.1 | 主实验 |
| e4_no_scale | 同奖励 + post-zscore乘1 | 单独检查缩放作用 |
| e4_no_prefix | 与e2相同 | 去掉补充信号，同时取消其缩放；可复用E2，不冒充额外独立实验 |

尚无普通HEPO实测对照，不声称优于HEPO。若需比较门控本身，另外实现正确SID格式的普通HEPO对照并固定任务/预算；当前消融仅分离“补充信号整体”和“缩放”的作用。

启动脚本仍为scripts/grpo_full_single_gpu.sh，通过环境变量GRPO_VARIANT=e3选择；输出独立时间戳目录。脚本要求65GiB空闲，已有E2检查点占空间，先查看df，不自动删除任何旧模型。训练后完整valid评估，不使用test。

## 共同诊断

增加grpo_diagnostics.py：每个进程最多32条异常事件，实际有效token KL>=10000或最大值非有限时记录step、任务、sample ID、token ID、reference/current logp和完成序列；不记录原始标题/用户文本，不修改KL或loss。异常日志为kl_events_rank0.jsonl；上限达到后不再保存新事件。新增kl_token_max日志。CPU测试未覆盖真实GPU异常路径，该路径待GPU验证。

训练资源写入training_resources.json，含Trainer.train耗时（含周期验证/保存）与峰值显存，最终导出不计。E2原始运行没有训练显存profile，不凭空补数。共同诊断不改变reward/loss公式，但有少量同步开销，不能把总时间差全部归于reward。

## 验收与下一步

本地13项测试通过（旧7+新6），语法检查通过；扩展scripts/check_grpo_runtime.py检查真实ML环境前缀callback、torch标准化与缩放。GPU执行尚未开始，不声称新方案有效。

先在服务器运行runtime检查，PASS后启动e3。对比E2/E3的HR/NDCG、有效Advantage组比例、KL分位/尖峰、覆盖与Top1集中度、训练资源。若有共同实现缺陷先修复并重建E2/E3公平对照；若新reward无收益保留失败结论。之后运行e4_no_scale，再决定是否需要权重敏感性/多seed；不因为单次validation较好就提前反复查看test。
