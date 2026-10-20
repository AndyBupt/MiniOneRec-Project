# First full-data SFT run (E1 candidate)

Prepared 2026-10-20. This is a full-data run, not the 128-sample smoke run. No formal training results have been observed yet. Commands below run on the user's AutoDL server, not on the local Mac.

## Environment confirmed by user

- NVIDIA A800-SXM4-80GB; driver 580.126.09.
- Conda base, Python 3.12.3, PyTorch 2.5.1+cu124, CUDA runtime 12.4.
- Transformers 4.57.1, Datasets 4.2.0, NumPy 1.26.4, PyArrow 21.0.0.
- Project import and BF16 forward/backward check passed.
- Base model: Qwen/Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`, downloaded to `/root/autodl-tmp/models/Qwen2.5`. User confirmed the 2.9 GiB weight file exists; actual training/model load has not been confirmed.

## Fixed initial configuration

Industrial_and_Scientific; all three upstream SFT datasets (`sample=-1`); full-parameter BF16 training; max 3 epochs with the original patience-3 early stopping; micro batch 2; accumulation 32; effective batch 64; LR 2e-5; linear scheduler (Trainer default); warmup 3%; max length 512; seed/data seed 42. Full valid next-SID dataset supplies validation loss; no test interactions are used for training or checkpoint selection. Auxiliary item catalog alignment retains the upstream transductive setting.

The initial configuration is not a tuned optimum. Compared with the upstream eight-GPU launcher, model, batch, LR, epoch budget and warmup are explicitly changed and recorded. All later E2/E3 experiments must start from the same selected SFT checkpoint. The training run alone does not establish HR/NDCG or item-level accuracy.

## Common engineering changes

- EvalSidDataset prompt now matches SidSFTDataset; SFT sample inputs and label masks remain identical to upstream.
- TrainingArguments receive the explicit seed and data seed. External reporting is disabled.
- Warmup steps/ratio can be specified; default arguments preserve the original 20-step warmup.
- Every successful run writes configuration, trainer metrics, elapsed time including saving, and CUDA allocator peak allocated/reserved memory. Allocator peaks do not include every byte reported by nvidia-smi.
- The launcher captures code commit/patch, package versions, data/model hashes and refuses to reuse an existing output directory. No reward optimization or GRPO correctness fixes are included in this change.

## Launch on server

```bash
cd /root/autodl-tmp/MiniOneRec-Project
git pull --ff-only
mkdir -p logs
run_tag=$(date -u +%Y%m%dT%H%M%SZ)
nohup env MODEL_PATH=/root/autodl-tmp/models/Qwen2.5 \
  CUDA_VISIBLE_DEVICES=0 RUN_ID="$run_tag" \
  bash scripts/sft_full_single_gpu.sh \
  > "logs/sft_full_${run_tag}.log" 2>&1 &
echo "Log: logs/sft_full_${run_tag}.log"
tail -f "logs/sft_full_${run_tag}.log"
```

Ctrl+C exits `tail` while the nohup training continues. The log starts with the unique output directory, and must end with `COMPLETED` after a successful run. If import/model loading fails, training has not started; preserve the traceback and repair the actual cause before relaunching with a fresh run ID.

Expected output directory: `output/sft_e1_Industrial_and_Scientific_<UTC-run-id>/`. It contains `run_config.json`, `run_metrics.json`, environment/code/data/model records, intermediate checkpoints, and `final_checkpoint/` with both model and tokenizer. The exported model uses the lowest valid-loss checkpoint selected by Trainer, which is not necessarily the highest-HR checkpoint.

## After completion

Read actual train sample count and truncation warnings, check finite train/valid losses, record runtime and allocator memory peaks. Then implement the fixed Top-10 valid recommendation protocol, full invalid-SID counting and collision reporting. HR/NDCG remain pending until that evaluation succeeds. Test is reserved for evaluation after the configuration is frozen. GRPO still requires the sample-target mapping fix before it is safe to run formal E2–E4.
