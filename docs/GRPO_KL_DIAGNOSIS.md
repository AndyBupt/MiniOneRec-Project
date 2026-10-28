# GRPO KL 诊断（2026-10-28）

## 已知证据

E2有6597次更新；2362步KL约2.8895e8，8步KL>100。去掉最大尖峰，逐更新KL均值约9.01355，整体中位数0.856。生成长度一直为5。参数为FP32、BF16混合计算；原实现通过selective_log_softmax计算实际token概率，KL用exp(delta)-delta-1。大正delta会指数放大，但现有标量日志没有token、task或delta，不能倒推出唯一原因。

注意：组内优势均值约0，所以policy loss标量接近0不代表梯度为0；平均loss被KL异常主导，不能与SFT loss比较。裁剪前grad_norm尖峰不等于实际更新大小。约束beam sampling并非从原始完整softmax策略独立采样；该轨迹上的KL估计不等于精确全分布KL，不能直接将标量解释成策略距离。

## 冻结检查点诊断

新增scripts/diagnose_grpo_kl.py：固定seed42抽取128个valid查询的已保存Top10，补newline/EOS，按512token prompt上限构造批次，依次加载原SFT reference与E2 policy（均FP32参数）。同一token比较reference FP32 forward、policy FP32 forward、policy BF16 autocast forward；显式FP32 log_softmax计算概率与KL，输出最极端30个token及分位数。只读取本地模型，不生成、不更新、不使用test。参考/策略vocab必须完全一致。

此工具诊断最终模型的保存候选，不能复现已丢失的2362步微批、训练模式或原始选择性logsoftmax运算；eval模式和显式FP32 softmax也与训练有差异。它用于确定后续取证方向，不能单独宣判原因。GPU运行待用户执行；本地只验证语法、汇总非有限值逻辑和7项现有标签测试。

## 如何决定下一步

- 若FP32与BF16都存在大概率比，优先检查token位置（SID层/newline/EOS）、reference与policy偏移、训练任务与解码分布，而非直接归咎低精度。
- 若只有BF16出现明显异常，再使用原路径/相同token验证低精度logits及logsoftmax误差。不要直接clamp KL掩盖问题。
- 若最终模型未复现，需在后续受控诊断中记录异常事件的step、task、sample_id、mask、ref/current per-token logp及loss精度；旧日志不足以还原过去。
- 若发现必须修复的共同计算问题，E2和E3采用同一修复，重建可比baseline；不把修复收益混入Reward创新。

E3候选仍围绕86.5%全零奖励组。前缀奖励并非原创，需审查已有HEPO并比较覆盖/语义误奖励风险；当前不改变reward、不启动E3。最终test保留至方法/超参冻结。


## 2026-10-28 冻结模型诊断结果

已收到GPU结果：128个valid查询、1280条候选、6400个token；完成56.29秒，所有completion长度5。FP32 KL项median/p95/max=2.260/276.290/8657.213；混合计算=2.262/264.986/8059.404，无非有限值。policy两精度logp差绝对值median0.02817、p950.14918、max1.49152。两精度均存在较大KL项，因此本批最终模型候选不支持“只是BF16误差”的解释，不能排除历史训练某步的额外精度问题。

例：valid4398、首token<a_73>，reference FP32 logp=-17.8060，policy FP32=-26.8733，差约9.0673，exp(delta)-delta-1约8657；BF16 policy=-26.8019，KL约8059。大值在FP32仍存在。另有第三层SID token<c_0>出现KL约4543；不可把问题归因于EOS单一位置。

重要限制：这些是保存的Top10 beam候选上的逐token项，非训练组均值，不能与训练2362步的2.89e8直接作同单位对照，也不能估计真实全分布KL。左padding的前向位置处理沿用原Trainer直接forward方式，尚未验证padding不变性或与generate的position_ids一致性；该差异也应纳入后续诊断，不能直接把全部差值称为学习造成的策略漂移。

后续：保留既有E2结论，先检查同一query单条/左pad批次前向一致性和generate position_ids协议，再为训练异常增加逐token取证。未证实前不clamp、不切换全FP32重训、不将BF16当成根因。

## 位置编号检查脚本

scripts/check_grpo_positions.py固定12条查询（包含上次异常行4398），每条取已保存Top1，比较无padding、人工左pad17/97 token的直接forward、mask累计得到的显式position_ids，以及真实generate首步原始logits。分别检查SFT/E2和FP32/BF16，均eval模式、无梯度。人工padding用于隔离位置处理，不声称复刻训练混合batch；generate只做一个token的贪心诊断，不评估推荐、不应用约束处理器。只对首token比较generate；多步cache不在本次覆盖范围。

不因缺少position_ids就预判为bug：RoPE对统一位置平移可能不敏感，必须检查实际差异，并区分精度/内核计算差异。若隐式/显式/生成计算均接近，不修改共同训练实现，转向异常训练微批的概率比取证；若显式position明显消除偏差，进一步复查真实混合batch再修复，E2/E3共同采用同一实现。

本地检查：语法、mask累计位置示例、7项现有标签回归通过。GPU诊断尚待执行。脚本不会改写权重或已存在的结果文件。

## 位置检查实际结果（2026-10-28）

GPU完成12条查询、两模型、两精度、左pad17/97，耗时14.91秒。FP32下reference/policy隐式padding最大logp差分别8.31e-5/5.34e-5；generate首步与隐式forward最大差6.68e-5/3.43e-5，未见明显位置错位。BF16下隐式padding最大差0.716/0.185，显式位置为0.644/0.276；显式位置并不一致消除差异，不能将差异归咎缺失position_ids。BF16输入形状/路径数值敏感性需要记录，不能称完全无误差。

当前样本与首步检查不支持位置错位是主要原因，不修改Trainer位置实现，不因此重训E2。不能据12条数据排除所有混合batch/多步cache问题，不能解释历史尖峰唯一原因。结合上一轮FP32也有较大KL项，下一轮应记录异常轨迹上的逐token概率比，避免直接clamp掩盖问题。

Reward设计门槛：已有HEPO分层0.2/0.5/1，不能作为个人原创。可研究只在无完整命中组/推荐任务启用的前缀补充信号。group zscore会基本抵消整体奖励缩放（除eps影响）：all-miss组乘小lambda并不能保证更新小；同一组共享前缀也可能仍为零方差。必须统计非零Advantage组比例，而不只统计非零Reward。若引入group gate或Advantage缩放，明确作为新机制并单独消融。E3仍未实现或运行。
