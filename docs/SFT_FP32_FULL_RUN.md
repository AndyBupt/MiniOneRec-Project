# 完整 SFT 参数精度对照：服务器操作

状态：脚本已准备，尚未在 GPU 执行；不代表已经产生验证集提升。

## 目标与公平性

从原始 `/root/autodl-tmp/models/Qwen2.5` Base 出发，重跑完整三任务。不要使用旧 SFT 或128条诊断模型作为这次起点。沿用原 train/valid、SID、seed42、微批2、累积32、batch64、LR2e-5、3epoch、warmup3%、长度512、默认linear scheduler、patience3、最低valid loss选模。

主要训练改动：FP32参数＋BF16混合精度。仍先按原代码以BF16加载Base并扩词表，再转FP32，尽量保持原初始化数值；新增SID embedding初始值哈希写入run_config。优化器仍为原AdamW；实际状态dtype在第10步记录。参数与优化器数值行为不是独立分离的因素。

工程改动：为节省磁盘，本次中间checkpoint使用`save_only_model=True`，不保存optimizer/scheduler状态；不会改变持续运行时的更新，但**不能精确断点续训**。保留原E1，不自动清理任何旧文件。中途失败先检查原因；不要盲目指定resume。

`sft.py`和旧完整SFT启动器的默认参数精度仍为BF16，新增profile默认关闭；本次专用脚本显式开启FP32与profile。

## 资源

- 当前单卡A800 80GB，现有conda base和依赖，不安装或升级包。
- 必须至少35 GiB空闲磁盘。FP32保存权重约6GB，可能拆成多个safetensors文件；best/last检查点、保存过程及root/final导出可能同时占空间。
- 只保存模型减少占用，检查磁盘不等于预留磁盘；不要并行下载或训练。
- 小样本已测FP32 peak allocated约28.83GiB，完整三任务的峰值由实际profile确认。
- 预估训练约3.5–5小时，另加数据准备、验证、保存及推荐评估。费用按用户实例实际单价×实际计费时长计算，当前没有真实单价。
- 第10步输出`profile_step10.json`，若reserved超过64GiB或推算训练超过18000秒则停止；该早期估算不包含所有后续验证/保存，不能保证最终时长或显存上界。

## 操作

### 1. 更新与检查

```bash
cd /root/autodl-tmp/MiniOneRec-Project
git pull --ff-only
df -h /root/autodl-tmp
nvidia-smi
```

确认上一项诊断已结束，GPU没有其他训练，磁盘Avail至少35GiB。更新或检查失败先停下，不自动删除E1、Base和原始记录。

### 2. 后台启动

```bash
mkdir -p logs
run_tag=$(date -u +%Y%m%dT%H%M%SZ)
nohup env MODEL_PATH=/root/autodl-tmp/models/Qwen2.5 \
  CUDA_VISIBLE_DEVICES=0 RUN_ID="$run_tag" \
  bash scripts/sft_fp32_full_single_gpu.sh \
  > "logs/sft_fp32_full_${run_tag}.log" 2>&1 &
echo "Log: logs/sft_fp32_full_${run_tag}.log"
tail -f "logs/sft_fp32_full_${run_tag}.log"
```

Ctrl+C只退出tail监控，nohup训练继续。断开SSH不会主动终止nohup；关机/释放实例会停止训练。本次未保存完整优化器状态，不建议中途关机。

训练和完整valid评估自动接续：

1. 检查硬件、磁盘和原始tokenizer（防止误用已扩SID的checkpoint）。如果原E1的Base权重哈希记录存在，核对一致。
2. 新目录`output/sft_e1_fp32_Industrial_and_Scientific_<UTC-run-id>/`保存完整SFT。
3. 第10步保存实测profile；可先下载这个小JSON查看，不需要等待结束。
4. 最低valid loss模型导出`final_checkpoint`。FP32权重可能分片；评估启动器已支持单文件和分片权重。
5. 自动对4532条valid运行原协议BF16推理、beam10，写入`valid_eval/`；不跑test。
6. 自动生成只含记录和预测的`review_bundle.tar.gz`，**不含权重或optimizer**。

### 3. 判断结束与下载

必须看到`ALL_COMPLETED: .../valid_eval/metrics.json`，不是只看到训练100%或普通`COMPLETED`。

成功后只下载该训练目录的`review_bundle.tar.gz`交回分析，不要右键下载整个训练目录。该包包含run_config/run_metrics/profile、版本/输入哈希和完整valid预测，可核对公平条件、复算指标和检查候选覆盖。

若训练完成但自动评估失败，保留模型与日志，修复评估后重评，不重新训练。若资源门槛或磁盘检查失败，把报错和profile发回；不自动提升门槛或删除旧checkpoint。

## 结果如何判断

与旧E1的完整valid HR@5/10=0.038173/0.055384、NDCG@5/10=0.035078/0.040476比较，补查SID碰撞子集、候选集中度、invalid、训练耗时与显存。同一seed一次重跑的差异不能直接称为统计显著；确认后再决定后续复验和GRPO基线。

## 本地验证范围

Python/shell语法检查通过；resource profile记录/停止与实际optimizer dtype计数逻辑的CPU模拟测试通过；评估哈希对单文件/多分片权重的测试通过。未在本地运行真实GPU加载、训练、FP32分片导出或完整valid，不宣称这些已实测。
