#!/usr/bin/env bash
# Full-data E1 precision control, followed by full-valid recommendation evaluation.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${MODEL_PATH:?Set MODEL_PATH to the original Qwen2.5-1.5B Base directory, not an SFT checkpoint.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export CATEGORY=Industrial_and_Scientific
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
training_dir="${OUTPUT_DIR:-$repo_root/output/sft_e1_fp32_Industrial_and_Scientific_${run_id}}"
if [[ -e "$training_dir" ]]; then
  echo "Choose a fresh OUTPUT_DIR: $training_dir"
  exit 1
fi
python - "$MODEL_PATH" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
if shutil.disk_usage('.').free < 35 * 2**30:
    raise RuntimeError('Need at least 35 GiB free on the project disk for FP32 model-only checkpoints and exports. Do not delete old runs automatically.')
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('Need a BF16-capable CUDA GPU')
if torch.cuda.get_device_properties(0).total_memory < 70 * 2**30:
    raise RuntimeError('This full-data configuration targets the current A800 80GB; profile other GPUs separately')
base = Path(sys.argv[1])
if not (base / 'model.safetensors').exists():
    raise FileNotFoundError('Original 1.5B Base model.safetensors not found')
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
if any(token.startswith(('<a_', '<b_', '<c_')) for token in tokenizer.get_vocab()):
    raise ValueError('Base tokenizer already contains SID tokens; do not use the old SFT or tiny-fit model here')
baseline = Path('output/sft_e1_Industrial_and_Scientific_20261020T101253Z/model_sha256.txt')
if baseline.exists():
    expected = baseline.read_text().split()[0]
    digest = hashlib.sha256()
    with (base / 'model.safetensors').open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('Original Base weight SHA256 differs from the saved E1 record')
    print('Original Base weight hash matches E1')
else:
    print('E1 base hash record not found; retain new input hashes and verify model provenance separately')
print('GPU:', torch.cuda.get_device_name(0))
print('Preflight passed; at least 35 GiB free disk')
PY
echo "Full precision-control output: $training_dir"
export PARAMETER_DTYPE=float32 SAVE_ONLY_MODEL=True PROFILE_STEPS=10
RUN_ID="$run_id" OUTPUT_DIR="$training_dir" bash scripts/sft_full_single_gpu.sh
CHECKPOINT_PATH="$training_dir/final_checkpoint" EVAL_SAMPLE=-1 \
  OUTPUT_DIR="$training_dir/valid_eval" bash scripts/eval_valid_single_gpu.sh
python - "$training_dir" <<'PY'
import sys
import tarfile
from pathlib import Path
root = Path(sys.argv[1])
names = ['run_config.json', 'run_metrics.json', 'profile_step10.json',
         'environment.txt', 'code_commit.txt', 'code_patch.diff', 'data_sha256.txt', 'model_sha256.txt',
         'valid_eval/metrics.json', 'valid_eval/predictions.json', 'valid_eval/predictions.json.run.json',
         'valid_eval/environment.txt', 'valid_eval/code_commit.txt', 'valid_eval/code_patch.diff',
         'valid_eval/input_sha256.txt']
with tarfile.open(root / 'review_bundle.tar.gz', 'w:gz') as archive:
    for name in names:
        archive.add(root / name, arcname=name, recursive=False)
print('Download this records-only bundle:', root / 'review_bundle.tar.gz')
PY
echo "ALL_COMPLETED: $training_dir/valid_eval/metrics.json"
