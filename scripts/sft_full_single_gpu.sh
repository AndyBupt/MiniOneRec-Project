#!/usr/bin/env bash
# First full-data E1 run on one A800 80GB; do not launch with torchrun.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen2.5-1.5B model directory.}"
for model_file in config.json tokenizer_config.json tokenizer.json model.safetensors; do
  if [[ ! -f "$MODEL_PATH/$model_file" ]]; then
    echo "Missing model file: $MODEL_PATH/$model_file" >&2
    exit 1
  fi
done
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "This script requires one process and one GPU." >&2
  exit 1
fi

category="${CATEGORY:-Industrial_and_Scientific}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-$repo_root/output/sft_e1_${category}_${run_id}}"
if [[ -e "$output_dir" ]]; then
  echo "Output directory already exists; choose a new OUTPUT_DIR: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m pip freeze > "$output_dir/environment.txt"
git rev-parse HEAD > "$output_dir/code_commit.txt"
git diff HEAD -- sft.py data.py scripts/sft_full_single_gpu.sh > "$output_dir/code_patch.diff"
sha256sum \
  "data/Amazon/train/${category}_5_2016-10-2018-11.csv" \
  "data/Amazon/valid/${category}_5_2016-10-2018-11.csv" \
  "data/Amazon/index/${category}.index.json" \
  "data/Amazon/index/${category}.item.json" \
  > "$output_dir/data_sha256.txt"
sha256sum "$MODEL_PATH/model.safetensors" > "$output_dir/model_sha256.txt"
echo "Run output: $output_dir"

python -u sft.py \
  --base_model "$MODEL_PATH" \
  --train_file "data/Amazon/train/${category}_5_2016-10-2018-11.csv" \
  --eval_file "data/Amazon/valid/${category}_5_2016-10-2018-11.csv" \
  --sid_index_path "data/Amazon/index/${category}.index.json" \
  --item_meta_path "data/Amazon/index/${category}.item.json" \
  --output_dir "$output_dir" \
  --category "$category" \
  --sample -1 \
  --num_epochs 3 \
  --batch_size 64 \
  --micro_batch_size 2 \
  --learning_rate 2e-5 \
  --warmup_steps 0 \
  --warmup_ratio 0.03 \
  --cutoff_len 512 \
  --seed 42 \
  --freeze_LLM False \
  --parameter_dtype "${PARAMETER_DTYPE:-bfloat16}" \
  --save_only_model "${SAVE_ONLY_MODEL:-False}" \
  --profile_steps "${PROFILE_STEPS:-0}"

echo "COMPLETED: $output_dir/final_checkpoint"
