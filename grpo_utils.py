"""Sample-local labels and unchanged upstream rule/ranking rewards (CPU only).

These helpers deliberately have no access to train/valid history-to-target maps.
"""

import math
import re


RL_COLUMNS = ["prompt", "completion", "target_sid", "history_sids", "task_type", "sample_id"]


def make_rl_record(prompt, target_sid, *, history_sids=(), task_type, sample_id):
    if not isinstance(prompt, str) or not isinstance(target_sid, str) or not target_sid.strip():
        raise ValueError("RL rows require a prompt and a nonempty sample-local target SID")
    if isinstance(history_sids, str):
        raise ValueError("history_sids must be a sequence, not an encoded string")
    return {"prompt": prompt, "completion": target_sid, "target_sid": target_sid,
            "history_sids": list(history_sids), "task_type": task_type, "sample_id": str(sample_id)}


def sample_targets(inputs):
    targets = []
    for row in inputs:
        target = row.get("target_sid")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Missing sample-local target_sid; use the repaired RL datasets, not a history map")
        if "completion" in row and row["completion"] != target:
            raise ValueError("completion and sample-local target_sid disagree")
        targets.append(target)
    return targets


def validate_generation_groups(inputs, num_generations):
    if not inputs or num_generations < 2 or len(inputs) % num_generations:
        raise ValueError("Each local batch must contain complete groups of >=2 generations")
    targets = sample_targets(inputs)
    for start in range(0, len(inputs), num_generations):
        group = inputs[start:start + num_generations]
        if (len({row["prompt"] for row in group}) != 1
                or len(set(targets[start:start + num_generations])) != 1
                or len({row.get("sample_id") for row in group}) != 1):
            raise ValueError("Generation group mixes samples/targets; check sampler and local batch size")
    return targets


def reward_kwargs_from_samples(inputs):
    if not inputs:
        raise ValueError("RL reward batch must be nonempty")
    sample_targets(inputs)  # Fail closed if Trainer removed the label column.
    keys = set(inputs[0])
    if any(set(row) != keys for row in inputs):
        raise ValueError("RL samples have inconsistent metadata columns")
    # The sampler already repeats the whole row G times: do not repeat labels again.
    return {key: [row[key] for row in inputs] for key in sorted(keys - {"prompt", "completion"})}


def _check_lengths(prompts, completions, target_sid):
    if not (len(prompts) == len(completions) == len(target_sid)):
        raise ValueError("One target_sid is required per generated completion")


def rule_reward(prompts, completions, target_sid, **kwargs):
    """Original exact-match reward; only its label source changes."""
    _check_lengths(prompts, completions, target_sid)
    return [float(completion.strip('\n" ') == target.strip('\n" '))
            for completion, target in zip(completions, target_sid)]


def make_ranking_reward(num_generations):
    """Preserve original normalized negative rank penalties and all-miss zeroing."""
    if num_generations < 2:
        raise ValueError("num_generations must be >=2")
    raw = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
    penalties = [-value / sum(raw) for value in raw]

    def ndcg_rule_reward(prompts, completions, target_sid, sample_id=None, **kwargs):
        _check_lengths(prompts, completions, target_sid)
        if sample_id is not None and len(sample_id) != len(prompts):
            raise ValueError("sample_id length mismatch")
        rows = [{"prompt": p, "target_sid": t, "sample_id": sample_id[i] if sample_id else None}
                for i, (p, t) in enumerate(zip(prompts, target_sid))]
        validate_generation_groups(rows, num_generations)
        rewards = []
        for start in range(0, len(completions), num_generations):
            hits = [completions[i].strip('\n"') == target_sid[i].strip('\n"')
                    for i in range(start, start + num_generations)]
            rewards.extend([0.0 if hit else penalties[j] for j, hit in enumerate(hits)]
                           if any(hits) else [0.0] * num_generations)
        return rewards

    return ndcg_rule_reward


def gated_prefix_signals(prompts, completions, target_sid, task_type, num_generations,
                         sample_id=None, advantage_scale=0.1):
    """E3: SID-next all-miss groups only; supplementary signal, not a new SID algorithm.

    Return per-completion bonuses, post-normalization scales and eligible flags.
    Full-match groups and other tasks retain their original ranking objective.
    """
    _check_lengths(prompts, completions, target_sid)
    if not 0 <= advantage_scale <= 1 or not math.isfinite(advantage_scale):
        raise ValueError("prefix advantage scale must be finite and in [0, 1]")
    if len(task_type) != len(prompts) or (sample_id is not None and len(sample_id) != len(prompts)):
        raise ValueError("One task/sample ID is required per completion")
    rows = [{"prompt": p, "target_sid": t, "sample_id": sample_id[i] if sample_id else None}
            for i, (p, t) in enumerate(zip(prompts, target_sid))]
    validate_generation_groups(rows, num_generations)
    normalize = lambda s: s.strip('\n" ')
    pattern = re.compile(r"<a_(\d+)><b_(\d+)><c_(\d+)>")
    bonuses, scales, eligible = [], [], []
    for start in range(0, len(prompts), num_generations):
        end = start + num_generations
        tasks = task_type[start:end]
        if len(set(tasks)) != 1:
            raise ValueError("Generation group mixes task types")
        target = normalize(target_sid[start])
        parts = pattern.fullmatch(target)
        if parts is None:
            raise ValueError(f"Invalid three-level target SID: {target}")
        hit = any(normalize(c) == target for c in completions[start:end])
        active = tasks[0] == "sid_next" and not hit
        for completion in completions[start:end]:
            candidate = pattern.fullmatch(normalize(completion))
            bonus = 0.0
            if active and candidate is not None:
                if candidate[1] == parts[1]:
                    bonus = 0.2
                    if candidate[2] == parts[2]:
                        bonus = 0.5
            bonuses.append(bonus)
            scales.append(advantage_scale if active else 1.0)
            eligible.append(active)
    return bonuses, scales, eligible


def make_gated_prefix_reward(num_generations):
    def gated_prefix_reward(prompts, completions, target_sid, task_type, sample_id=None, **kwargs):
        return gated_prefix_signals(prompts, completions, target_sid, task_type,
                                    num_generations, sample_id)[0]
    return gated_prefix_reward
