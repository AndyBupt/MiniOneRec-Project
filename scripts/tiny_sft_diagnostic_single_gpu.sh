#!/usr/bin/env bash
# Short train-only precision comparison on one A800 80GB; source checkpoint stays intact.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the existing E1 final_checkpoint directory.}"
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "Use one process and one GPU."
  exit 1
fi
for filename in model.safetensors config.json tokenizer.json tokenizer_config.json; do
  if [[ ! -f "$CHECKPOINT_PATH/$filename" ]]; then
    echo "Missing source checkpoint file: $CHECKPOINT_PATH/$filename"
    exit 1
  fi
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
output_dir="${OUTPUT_DIR:-$repo_root/output/tiny_sft_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$output_dir" ]]; then
  echo "Choose a new OUTPUT_DIR; existing results will not be overwritten: $output_dir"
  exit 1
fi
python - <<'PY'
import shutil
import torch
if shutil.disk_usage('.').free < 10 * 2**30:
    raise RuntimeError('Need at least 10 GiB free disk for two separate model exports and logs')
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('Need a BF16-capable CUDA GPU')
print('GPU:', torch.cuda.get_device_name(0))
print('GPU memory GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
python scripts/tiny_sft_diagnostic.py prepare \
  --train data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output "$output_dir" --count 128 --seed 42
echo "Diagnostic output: $output_dir"
python -m pip freeze > "$output_dir/environment.txt"
git rev-parse HEAD > "$output_dir/code_commit.txt"
git diff HEAD -- data.py evaluate.py LogitProcessor.py scripts/tiny_sft_diagnostic.py \
  scripts/tiny_sft_diagnostic_single_gpu.sh scripts/score_recommendations.py > "$output_dir/code_patch.diff"
sha256sum "$CHECKPOINT_PATH/model.safetensors" "$CHECKPOINT_PATH/tokenizer.json" \
  data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  scripts/tiny_sft_diagnostic.py scripts/tiny_sft_diagnostic_single_gpu.sh \
  data.py evaluate.py LogitProcessor.py scripts/score_recommendations.py > "$output_dir/input_sha256.txt"

evaluate_arm() {
  local arm_name="$1"
  local model_path="$2"
  mkdir -p "$output_dir/$arm_name"
  python -u evaluate.py \
    --base_model "$model_path" \
    --info_file data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --category Industrial_and_Scientific \
    --test_data_path "$output_dir/samples.csv" \
    --result_json_data "$output_dir/$arm_name/predictions.json" \
    --batch_size 1 --num_beams 10 --max_new_tokens 16 \
    --max_prompt_length 512 --length_penalty 0.0 --sample -1 --seed 42 \
    2>&1 | tee "$output_dir/$arm_name/evaluation.log"
  python scripts/score_recommendations.py \
    --predictions "$output_dir/$arm_name/predictions.json" \
    --catalog data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --output "$output_dir/$arm_name/metrics.json"
}

evaluate_arm initial "$CHECKPOINT_PATH"
for precision in bf16 fp32; do
  python -u scripts/tiny_sft_diagnostic.py fit \
    --checkpoint "$CHECKPOINT_PATH" --samples "$output_dir/samples.csv" \
    --output "$output_dir/$precision" --precision "$precision" \
    --steps "${STEPS:-100}" --lr 2e-5 --seed 42 \
    --batch-size 32 --micro-batch-size 2 \
    --max-memory-gib 60 --max-seconds 1800 \
    2>&1 | tee "$output_dir/${precision}_training.log"
  evaluate_arm "$precision" "$output_dir/$precision/final_checkpoint"
done
python scripts/tiny_sft_diagnostic.py summarize --output "$output_dir"
echo "COMPLETED: $output_dir/comparison.json"
