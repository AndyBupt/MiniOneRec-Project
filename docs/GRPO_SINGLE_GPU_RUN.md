# E2: single H800 80GB full GRPO run

Prepared 2026-10-27. GPU training is pending; CPU integration and seven label tests passed. This is an adapted upstream baseline, not a bit-for-bit reproduction of the eight-GPU launch.

## Launch

Run `scripts/grpo_full_single_gpu.sh` with CHECKPOINT_PATH set to the FP32 E1 final_checkpoint. It creates a new timestamped output directory, logs commit/diff/environment/command, trains on all mixed RL tasks for one epoch, retains two optimizer checkpoints, exports final weights/tokenizer and evaluates the complete valid split using the established Top-10 protocol. It does not use test labels or alter SFT weights.

## Fixed configuration

- Policy and reference load with FP32 parameters; BF16 autocast; AdamW with FP32 states. Gradient checkpointing, non-reentrant. No bitsandbytes/vLLM/DeepSpeed required.
- Mixed tasks: all history SID to next SID, catalog title/description to SID, up to 10,000 title-history to next SID rows, matching the existing mixed-task construction. Seed 42.
- Eight generations per prompt; local completion batch 8 (one prompt group per microbatch), accumulation 8: 64 completion sequences / eight prompt draws per optimizer update on one GPU. Repeated draws across the epoch are not necessarily distinct examples.
- One full epoch; no sample or max-step cutoff. Learning rate 1e-6, cosine decay, warmup 3%, grad norm 0.3. Beta 0.001, frozen SFT reference (sync disabled).
- Upstream ranking combines exact-match reward and existing rank-dependent penalties; all-miss groups remain zero reward. No new reward optimization claimed.
- Upstream `beam_search=True` currently sets both num_beams=8 and do_sample=True: stochastic beam sampling, not deterministic evaluation beam search. Temperature 1.0. No ground-truth insertion, dynamic sampling, DAPO or GSPO.
- Prompt cap 512, completion cap 16 (three SID tokens plus terminators). Monitor completion_length and sampled outputs for truncation. Evaluate/save each quarter epoch; retain two optimizer checkpoints. Need at least 65GiB free before launch.
- Full valid Top-10 HR/NDCG after training; intermediate RL eval loss/reward is not Top-10 recommendation quality.

## Interpretation and limits

This entry uses the repository's custom GRPO-style surrogate, including detached current-policy log probabilities, rather than claiming a modern TRL clipped multi-update GRPO implementation. CPU preflight does not establish GPU backward/save compatibility or peak memory. The user chose to start the full run directly, so the first real optimizer updates are the initial GPU integration check; preserve failures and do not silently alter reward/config mid-run.

At start, inspect logged dataset size and total steps. Mixed-task data is approximately 50k rows (actual size depends on catalog construction); sampler repeats each row eight times, giving roughly 6k optimizer updates at accumulation eight. Runtime cannot be inferred from SFT: use the actual progress after 20–50 optimizer updates, adding intermediate evaluation/checkpoint time. No runtime, cost or quality results have been measured for this run.

Differences from upstream launch include group size 16→8, one GPU, one epoch, learning rate 1e-6, fixed reference, explicit precision/lengths, ordinary AdamW and checkpointing. Hold these fixed for subsequent reward-method E3 comparisons. Report E2 as the repaired single-GPU baseline. Formal SFT E1 full-valid HR@10 is 0.1233451, NDCG@10 0.0909790.

## Monitoring

Use train.log, nvidia-smi and remaining disk space. Track KL, reward, reward_std, zero_variance_group_fraction, all_zero_reward_group_fraction, completion_length and grad_norm. A near-zero scalar policy loss alone does not mean no gradient because group advantages are centered. Stop on nonfinite loss/gradients or runtime errors and diagnose; do not launch repeated copies. Checkpoints contain optimizer state, but automated resume is not implemented in this launcher.
