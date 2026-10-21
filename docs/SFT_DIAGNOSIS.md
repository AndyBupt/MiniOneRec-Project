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


## 第一次服务器诊断与修复

原始结果保留为 `experiments/E1_SFT_DIAGNOSIS_20261021_v1.json`。

- 解码项报 `IndexError ... got set`，源于诊断脚本漏抽取 `evaluate.py` 的集合转列表循环。不是原评估代码缺陷，不能将这次报错视为解码失败。已补抽取该实际循环，并新增返回类型回归测试。完整 CPU tensor-mask 检查仍待服务器重跑。
- 第 0/10/27 层抽样主干权重相同元素比例分别为 84.16%/80.97%/79.43%，相对 RMS 变化约 0.717%/1.101%/1.472%；均有变化。最终 norm 全部 1536 个元素无变化。
- 两侧保存权重均 BF16。这些结果证明抽查主干发生变化，但无法证明所有模块正常学习，也无法单独判定舍入问题；还未检查新增 SID embedding、优化器状态或每步更新。
- 不修改训练/评估逻辑，不启动重训。先用新输出文件名重跑修正后的只读诊断。


## 第二次服务器诊断结果

证据：`experiments/E1_SFT_DIAGNOSIS_20261021_v2.json`。

- 解码检查通过：3670 个唯一目录 SID，blocked_paths=0；所有生成后缀均为 5 token（含换行与 EOS），实际 processor 的 100 项 CPU 掩码检查通过。
- 这排除了所检查的目录约束路径被错误封锁，不排除有限 beam 的搜索损失或生成框架其他环节的问题。
- 抽样权重结果与 v1 一致：主干确实变化，最终 norm 未变；仍不能确诊 BF16 舍入或模型容量不足。
- 下一项轻量检查：固定 seed=42，在原始 train CSV 抽 256 条，用现有 checkpoint、相同 prompt/约束/beam10 生成 Top-10。只用于训练集拟合诊断，不能作为泛化指标或与 valid 作显著性结论。保留 raw SID ranks、碰撞口径；不训练、不改权重。
- 若训练集也集中且低分，优先核查任务拟合、loss mask 与数值更新；若训练集明显较好而 valid 较弱，再重点查泛化/数据分布。单次 256 抽样只能提供线索。
