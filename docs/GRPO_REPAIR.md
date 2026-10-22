# GRPO 共同工程修复与验证记录

日期：2026-10-22。当前状态：本地代码修复、CPU审计与7项回归测试完成；真实ML环境集成和GPU训练尚待验证。没有启动GRPO训练，没有GRPO效果指标。

## 1. SFT 是否到此结束

当前FP32参数SFT已满足进入RL阶段的基线要求：完整训练、重载、约束Top-10评估和逐条审计均完成；完整valid HR@10=12.3345%、NDCG@10=0.090979，超过两个简单参照，候选覆盖由28增至645个SID。

这属于阶段性达标，不是SFT充分调优或整个项目完成。单seed、SID碰撞和剩余热门倾向仍在。现在保留这版checkpoint、数据与评估配置，暂不继续改变SFT超参；“固定SFT基线”不是在后续RL中冻结模型参数。

P0优先完成原版GRPO与个人Reward对照。多seed、任务配比、不同学习率/模型规模等SFT扩展降为P1/P2，不同时扩展多条实验线。新模型本地备份下载尚未收到完整性校验证据，不应释放服务器原权重。

## 2. 已修复的正确性问题

### 样本目标被历史字典覆盖

原流程：prompt→history→全局history2target→reward。相同历史有不同目标时，后写入目标覆盖前者。

现在`SidDataset`、`RLTitle2SidDataset`、`RLSeqTitle2SidDataset`各自返回统一字段：

```text
prompt, completion, target_sid, history_sids, task_type, sample_id
```

`target_sid`来自样本自身，`completion`保留兼容并要求与target一致。序列任务保存实际历史SID；没有交互历史的标题/描述对齐任务保存空历史列表。原映射属性暂保留兼容，但`rl.py`和Trainer不再通过它们读取目标。

### valid标签合入训练reward

`rl.py`移除全局训练/验证映射收集与合并。rule/ranking/semantic奖励直接接收行级`target_sid`，SASRec读取行级`history_sids`。valid单独构造Dataset；没有把valid target传给训练奖励。

注意：valid仍用于正常验证，这与valid标签参与训练reward不同。并未完成整个数据预处理的时间戳/全局泄漏审查。

### Trainer丢列与重复标签对齐

Trainer signature列保留全部RL元数据。自定义sampler已经将整行连续重复G次，reward参数不再额外repeat标签。每个local batch必须由完整G组组成，并检查每组prompt、target、sample_id一致，避免同prompt不同目标的两条样本混组。

缺失target、completion/target矛盾、跨样本混组直接报错，不回退到旧字典。

### 状态式解码processor复用

原训练内评估与训练候选生成复用同一个有`count`状态的processor；评估可能推进训练processor计数，而且test_beam可能与G不同。

现在分别创建processor，test使用test_beam，训练使用对应generation/beam配置。两者计数独立。这里尚未实际运行GPU生成，不能说所有生成路径均验证通过。

### add_gt定位

注入GT时，按每G条一个样本组定位，不再按不同目标SID的数量反推组长。它是原有可选方法的组边界修复；正式原版对照将明确记录是否启用，不能当作新贡献。

## 3. 资源、日志与入口调整

- `rl.py`删除额外预加载LLM；policy/reference仍由Trainer负责。取得reward设备不再加载第三份完整权重。
- SASRec改成按需导入，rule/ranking不需要其额外依赖；semantic保留必要的物品向量加载。
- SASRec需要真实交互历史，只允许`task_mode=sid`；混合标题/描述任务不伪造历史。SASRec遇到非法SID报错，不再用随机物品替代。
- 默认`task_mode=mixed`保留原三类RL任务；`sid`只作为明确记录的单任务选项，不能混称原三任务复现。
- 新增`sample`、`max_steps`用于后续小规模检查；默认仍为全量、epoch控制。尚未交付正式单卡训练配置，不照搬八卡`rl.sh`。
- 显式传递seed/data_seed和Dataset seed；子集抽样的seed策略更明确，不声称与原版缺省seed逐位复现。
- 关闭外部WandB report，保留本地Trainer日志；新增zero-variance/all-zero reward group比例监控，不改变奖励或Advantage。
- `mask_all_zero`原来没有接线；现在传True会明确报错，避免误以为生效。

## 4. 保留的算法定义

`grpo_utils.py`将原rule/ranking定义提取为纯CPU函数：

- rule：完整SID精确命中1，否则0，保留原strip规则。
- ranking：保留原按候选位置计算的归一化负惩罚，命中项0；组内没有命中时整组0。
- ranking模式仍将rule与ranking奖励组合，不新增层级奖励或修改其权重。
- Advantage仍为组内中心化、除以原实现的sample std加1e-4；KL、policy loss、DAPO/GSPO分支公式均未修改。
- 当前自定义Trainer不是带old-policy clip的标准完整GRPO实现；仍按仓库的GRPO风格实现准确描述。

这些是原版和后续Reward改进共同使用的工程修复，不作为个人算法创新。

## 5. 本地验证结果

执行：

```bash
python3 -m unittest discover -s tests -p 'test_grpo_labels.py'
python3 scripts/audit_grpo_labels.py --output output/grpo_label_audit_local.json
```

7项回归覆盖：重复历史不同目标、实际三类Dataset方法的统一列、标签不重复扩展、错组拒绝、原ranking惩罚和全错组、实际Trainer列保留、实际prepare前段不读取旧字典、独立test/train processor（其中相关断言组合在7个test中）。

全量CPU审计结果保留在`experiments/GRPO_LABEL_AUDIT_20261022.json`：

| 项目 | 结果 |
|---|---:|
| 审计train / valid行数 | 36,259 / 4,532 |
| 原train字典后写覆盖造成的标签不符 | 4,836条train行 |
| 再合入valid后查回目标变化 | 1,570条train行 |
| 修复后行级target与原行标签不符 | 0 |
| 故意污染旧映射后，用行自身正确答案计算rule reward | train/valid全为1 |

审计通过AST抽取真实`data.py`方法，使用CSV和轻量行容器绕开pandas/torch构造；它不验证完整Dataset初始化、Arrow转换、GPU张量、反向或实际生成。静态旧映射数字只针对SID序列任务，不冒充已运行的混合GRPO统计。

三个SFT Dataset与EvalSidDataset的AST和修改前一致；`sft.py`、已有权重及评估协议本轮未修改。

## 6. 下一项CPU集成检查（尚未运行）

`scripts/check_grpo_runtime.py`面向已有ML依赖的服务器环境，无需GPU、不下载或加载模型、不训练：

```bash
python scripts/check_grpo_runtime.py --output output/grpo_runtime_check.json
```

检查真实TRL/Transformers导入、Arrow列保留、实际RepeatRandomSampler、Reward行级对齐、实际Trainer归一化表达式的CPU张量结果、CPU模式GRPOConfig构造。结果`PASS`也不代表GPU训练通过。

当前本地Python没有torch/transformers/trl，未执行此集成检查。最近服务器SFT环境记录也未列出trl，因此后续先核对依赖，不批量重装或升级已验证的torch/numpy环境。仓库原锁定TRL0.24.0；已核对其[官方导出](https://github.com/huggingface/trl/blob/v0.24.0/trl/__init__.py)、[模型工具](https://github.com/huggingface/trl/blob/v0.24.0/trl/models/__init__.py)和[配置定义](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py)，真实导入仍待实机验证。

## 7. 修复范围与剩余风险

- 后续主入口是`rl.py`。`rl_gpr.py`的reward签名和标签源也做了兼容迁移，避免共享Trainer的新字段破坏回调，但它没有完成资源/算法全链路验证。
- `rl_gpr.py`既有HEPO解析使用方括号格式，与本项目角括号SID不一致；SASRec/HEPO组合、额外模型加载等尚不作为可用实验入口，不能据此宣称HEPO已跑通或开发完成。
- vLLM路径与多卡尚未验证，当前目标是后续单卡、完整local group检查。
- 原SDP关闭策略、采样beam、temperature路径等保留，后续GPU检查要确认实际候选生成行为，不能假设与离线确定性beam评估相同。
- 尚未选定单卡G、batch、长度、LR、beta和训练预算；实际依赖检查、短程profile、Reward/Advantage/KL监控通过后再启动正式E2。
- 没有GRPO指标或Reward优化收益；新SFT仍是当前已验证的基线。
