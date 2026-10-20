#!/usr/bin/env bash
# Small SFT process check only. It does not establish a recommendation baseline.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${MODEL_PATH:?Set MODEL_PATH to an existing local model directory after approving weight download.}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "MODEL_PATH must be a local model directory containing config.json" >&2
  exit 1
fi

category="${CATEGORY:-Industrial_and_Scientific}"
output_dir="${OUTPUT_DIR:-$repo_root/output/sft_smoke_$category}"
mkdir -p "$output_dir"

python sft.py \
  --base_model "$MODEL_PATH" \
  --train_file "data/Amazon/train/${category}_5_2016-10-2018-11.csv" \
  --eval_file "data/Amazon/valid/${category}_5_2016-10-2018-11.csv" \
  --sid_index_path "data/Amazon/index/${category}.index.json" \
  --item_meta_path "data/Amazon/index/${category}.item.json" \
  --output_dir "$output_dir" \
  --category "$category" \
  --sample 128 \
  --num_epochs 1 \
  --batch_size 16 \
  --micro_batch_size 1 \
  --learning_rate 1e-5 \
  --cutoff_len 512 \
  --seed 42 \
  --freeze_LLM False
