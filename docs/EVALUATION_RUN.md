# E1 valid recommendation evaluation

Prepared 2026-10-21 for the user's RTX 4090 24GB instance, Python 3.12.3 / PyTorch 2.5.1+cu124. No GPU evaluation has been run by the assistant. Unit and shell checks are not evidence that model inference has succeeded.

## Restore and environment

Clone/sync this repository on the server and install `requirements-sft-eval.txt` from PyPI. It preserves the preinstalled torch build and pins PyArrow 21.0.0 alongside NumPy 1.26.4. The original full requirements file is not used for this phase.

Upload `/Users/andy/研二/rec模型权重/e1_sft_backup.tar` to `/root/autodl-tmp/e1_sft_backup.tar`. From the project root restore with:

```bash
tar --skip-old-files -xf /root/autodl-tmp/e1_sft_backup.tar -C /root/autodl-tmp/MiniOneRec-Project
```

This restores `logs/` and `output/...` under the repository. A prior backup exists on the Mac; do not release the old instance's optimizer checkpoints if training resume is still required.

## Protocol

- Data: Industrial valid only, full size 4,532; 32-row deterministic pandas sample with seed 42 for the initial check. Test is not selected by this launcher.
- Checkpoint: `output/sft_e1_Industrial_and_Scientific_20261020T101253Z/final_checkpoint` from best valid-loss step 3572.
- Same next-SID prompt wording as SFT; left-truncate prompts to at most 512 tokens. Unlike upstream EvalSidDataset's test branch, which ignores its max_len argument, this evaluator enforces the cap explicitly. This is a fixed inference prompt cap, not a claim that every train/inference truncation boundary is identical (SFT truncates prompt plus answer together).
- Original constrained decoder and catalog, deterministic beam search, beam width/return count 10, batch 1, max new tokens 16, length penalty 0. Original launcher's beam width 50 is not used; all later comparison runs must share this protocol.
- Model/tokenizer vocabulary, expected weight tying and response prefix shape are checked before generation; failures stop the run instead of producing silently invalid results.
- No history-item filtering, candidate deduplication or rank compaction. Primary scores are raw-ranked SID exact-match HR@5/10 and NDCG@5/10.
- Invalid slots include both out-of-catalog strings and missing slots, divided by N*K; missing slots and duplicate slots also reported separately. All positions are inspected even after an early target hit.
- Report fraction of queries whose target SID collides and a separate subset with unique target SID. These are not full item-level metrics, since ambiguous SID-to-item resolution is not implemented.
- Output includes predictions with source row index/user/target item ID, score JSON, evaluation settings/runtime/CUDA allocator peaks, environment/code/input hashes.

## Start with 32 queries

```bash
cd /root/autodl-tmp/MiniOneRec-Project
CHECKPOINT_PATH="$PWD/output/sft_e1_Industrial_and_Scientific_20261020T101253Z/final_checkpoint" \
EVAL_SAMPLE=32 bash scripts/eval_valid_single_gpu.sh
```

Check the `Model reload passed` message, finite metrics, 10 legal predictions per query and sensible examples. The 32-query scores verify the pipeline; they are not the full E1 baseline.

## Full valid after successful initial check

```bash
CHECKPOINT_PATH="$PWD/output/sft_e1_Industrial_and_Scientific_20261020T101253Z/final_checkpoint" \
EVAL_SAMPLE=-1 bash scripts/eval_valid_single_gpu.sh
```

Use nohup with a timestamped log for unattended execution. Each launcher creates a new unique output directory and refuses to overwrite an existing one. Preserve predictions, metrics and runtime for all experiments. Choose the final test protocol/checkpoint without looking at test labels.
