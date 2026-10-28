#!/usr/bin/env bash
# Full E2: upstream mixed tasks and ranking reward, adapted to one 80GB GPU.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the SFT final_checkpoint}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
category=Industrial_and_Scientific
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
variant="${GRPO_VARIANT:-e2}"
case "$variant" in
  e2) reward_type=ranking; prefix_scale=0.1 ;;
  e3) reward_type=ranking_prefix; prefix_scale=0.1 ;;
  e4_no_scale) reward_type=ranking_prefix; prefix_scale=1.0 ;;
  e4_no_prefix) reward_type=ranking; prefix_scale=0.1 ;;
  *) echo "Unknown GRPO_VARIANT: $variant" >&2; exit 1 ;;
esac
output_dir="$PWD/output/grpo_${variant}_${category}_${run_id}"
test ! -e "$output_dir"
python - "$CHECKPOINT_PATH" <<'PY'
import json, shutil, sys
from pathlib import Path
import torch
p = Path(sys.argv[1])
assert (p / 'tokenizer.json').is_file()
idx = p / 'model.safetensors.index.json'
names = set(json.loads(idx.read_text())['weight_map'].values()) if idx.exists() else {'model.safetensors'}
assert all((p / name).is_file() for name in names)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.cuda.get_device_properties(0).total_memory / 2**30 >= 70, 'Requires an 80GB-class GPU'
assert shutil.disk_usage('output').free / 2**30 >= 65, 'Need 65GiB free for optimizer checkpoints and exports'
print('GPU:', torch.cuda.get_device_name(0), '\nSFT:', p)
PY
mkdir -p "$output_dir"
git rev-parse HEAD > "$output_dir/code_commit.txt"
git diff HEAD > "$output_dir/code_patch.diff"
python -m pip freeze > "$output_dir/environment.txt"
cp "$0" "$output_dir/launcher.sh"
echo "Training output: $output_dir"
args=(
  --model_path "$CHECKPOINT_PATH" --output_dir "$output_dir"
  --train_file "data/Amazon/train/${category}_5_2016-10-2018-11.csv"
  --eval_file "data/Amazon/valid/${category}_5_2016-10-2018-11.csv"
  --info_file "data/Amazon/info/${category}_5_2016-10-2018-11.txt"
  --sid_index_path "data/Amazon/index/${category}.index.json"
  --item_meta_path "data/Amazon/index/${category}.item.json"
  --category "$category" --task_mode mixed --sample -1
  --train_batch_size 8 --eval_batch_size 8
  --num_generations 8 --gradient_accumulation_steps 8
  --num_train_epochs 1 --max_steps -1
  --learning_rate 1e-6 --beta 0.001 --temperature 1.0
  --reward_type "$reward_type" --prefix_advantage_scale "$prefix_scale" --beam_search True
  --parameter_dtype float32 --optim adamw_torch
  --max_prompt_length 512 --max_completion_length 16
  --gradient_checkpointing True
  --eval_step 0.25 --save_steps 0.25 --save_total_limit 2
  --sync_ref_model False --test_during_training False
  --sample_train False --add_gt False --dynamic_sampling False
  --mask_all_zero False --dapo False --gspo False --seed 42
)
printf '%q ' python -u rl.py "${args[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"
python -u rl.py "${args[@]}" 2>&1 | tee "$output_dir/train.log"
CHECKPOINT_PATH="$output_dir/final_checkpoint" EVAL_SAMPLE=-1 \
  OUTPUT_DIR="$output_dir/valid_eval" bash scripts/eval_valid_single_gpu.sh
echo "COMPLETED: $output_dir"
