# SFT 集中推荐诊断

## 为什么现在做

E1 full-valid 的 4532 个 Top-10 列表合计只包含 28 个 SID；Top-1 仅有 4 个 SID。样本对齐与指标复算已通过。先排除解码和参数更新问题，再选择新的训练配置。

## 检查一：目录中的全部 SID 是否可达

`scripts/diagnose_sft.py` 只读取已保存的 tokenizer/config，抽取并执行当前 `evaluate.py` 的实际 SID 约束表构造代码，逐一检查目录中每个 SID 的所有前缀到 EOS 路径。另用真实 `ConstrainedLogitsProcessor` 在 CPU 上，以 10 beams、两个 batch 的人工输入抽查各步的可用 token 掩码。

重点看 `decoder.passed`、`blocked_paths`。全部通过说明被检查的约束路径正常；不说明有限宽度 beam search 一定能搜索到所有目标，也不验证模型概率或生成框架的整个运行过程。

## 检查二：SFT 主干参数是否变化

脚本比较原始基座与最终模型中第 0/10/27 层的部分注意力/MLP 权重，以及最终归一化权重；报告存储 dtype、不变比例、平均/最大绝对变化与相对 RMS 变化。矩阵只读少量行，不加载整个模型，也不读取优化器状态。

部分或大量参数相同可能来自小梯度、更新舍入等原因，不能只凭一个不变比例认定故障。当前 BF16 权重、AdamW、LR=2e-5 的组合值得核查，但数值更新问题尚未证实。若需要进一步定位，再单独设计短程受控实验。

必须使用此次 SFT 真正对应的基座；其他模型版本的差异不能解释为此次训练的更新。脚本记录代码哈希，未重新计算两个大权重文件的完整哈希。

## 服务器运行

在 conda base、项目根目录中执行；不需要 GPU，也不自动联网下载。

```bash
python scripts/diagnose_sft.py \
  --base /root/autodl-tmp/models/Qwen2.5 \
  --checkpoint /root/autodl-tmp/MiniOneRec-Project/output/sft_e1_Industrial_and_Scientific_20261020T101253Z/final_checkpoint \
  --output output/diagnose_sft_20261021.json
```

如输出文件已存在，另选一个文件名。某项检查异常会保存在该项 `error` 字段，其他项继续执行；文件保存成功不代表检查通过。先把完整 JSON 发回分析，不直接根据单个数值重训。

## 运行之后

1. 解码检查失败：定位受影响路径并修复，使用原 checkpoint 重评，不先重训。
2. 参数变化异常：核实基座来源、checkpoint、加载与冻结配置，必要时做短程精度/学习率对照。
3. 两项未发现问题：评估训练任务的 SID Loss、训练集小样本推荐拟合及每层 SID 准确率，再制定一次只改变一个主要因素的 SFT 对照。
4. SFT 可解释且评估稳定后，修复原版 GRPO 的目标映射问题，再开始 E2。

## 本地验证范围

本地通过 Python 语法检查，并用含分支的人工 token 序列验证了从真实 `evaluate.py` 抽取约束表的逻辑。本地无 PyTorch/Transformers/完整基座权重，真实 tokenizer、tensor mask 与权重比较等待服务器执行，未宣称通过。
