#!/usr/bin/env bash
# Fixed valid-only Top-10 protocol. EVAL_SAMPLE=32 for checks, -1 for full valid.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the trained final_checkpoint directory.}"
for filename in model.safetensors config.json tokenizer.json tokenizer_config.json; do
  if [[ ! -f "$CHECKPOINT_PATH/$filename" ]]; then
    echo "Missing checkpoint file: $CHECKPOINT_PATH/$filename" >&2
    exit 1
  fi
done
category="${CATEGORY:-Industrial_and_Scientific}"
sample="${EVAL_SAMPLE:-32}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-$repo_root/output/eval_valid_${category}_${sample}_${run_id}}"
if [[ -e "$output_dir" ]]; then
  echo "Choose a new output directory: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
python -m pip freeze > "$output_dir/environment.txt"
git rev-parse HEAD > "$output_dir/code_commit.txt"
git diff HEAD -- evaluate.py scripts/score_recommendations.py scripts/eval_valid_single_gpu.sh > "$output_dir/code_patch.diff"
sha256sum \
  "data/Amazon/valid/${category}_5_2016-10-2018-11.csv" \
  "data/Amazon/info/${category}_5_2016-10-2018-11.txt" \
  "$CHECKPOINT_PATH/model.safetensors" \
  "$CHECKPOINT_PATH/tokenizer.json" > "$output_dir/input_sha256.txt"
echo "Evaluation output: $output_dir"
python -u evaluate.py \
  --base_model "$CHECKPOINT_PATH" \
  --info_file "data/Amazon/info/${category}_5_2016-10-2018-11.txt" \
  --category "$category" \
  --test_data_path "data/Amazon/valid/${category}_5_2016-10-2018-11.csv" \
  --result_json_data "$output_dir/predictions.json" \
  --batch_size 1 \
  --num_beams 10 \
  --max_new_tokens 16 \
  --max_prompt_length 512 \
  --length_penalty 0.0 \
  --sample "$sample" \
  --seed 42
python scripts/score_recommendations.py \
  --predictions "$output_dir/predictions.json" \
  --catalog "data/Amazon/info/${category}_5_2016-10-2018-11.txt" \
  --output "$output_dir/metrics.json"
echo "COMPLETED: $output_dir/metrics.json"
