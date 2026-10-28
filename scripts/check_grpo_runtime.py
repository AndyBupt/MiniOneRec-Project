"""CPU-only integration preflight in the eventual ML environment; no model loading/training."""

import argparse
import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check():
    import torch
    import transformers
    import trl
    import rl  # Verify the actual entry point's optional/import dependencies.
    from datasets import Dataset
    from transformers import Trainer
    from trl import GRPOConfig
    from minionerec_trainer import ReReTrainer, RepeatRandomSampler
    from grpo_utils import (make_rl_record, validate_generation_groups, reward_kwargs_from_samples,
                            rule_reward, make_ranking_reward)

    x, y = "<a_1><b_2><c_3>", "<a_4><b_5><c_6>"
    dataset = Dataset.from_list([
        make_rl_record("same prompt", x, task_type="sid_next", sample_id="train:0"),
        make_rl_record("same prompt", y, task_type="sid_next", sample_id="train:1"),
    ])
    stub = ReReTrainer.__new__(ReReTrainer)
    stub.args = SimpleNamespace(remove_unused_columns=True)
    stub._signature_columns = None
    kept = Trainer._remove_unused_columns(stub, dataset, description="CPU label preflight")
    if set(kept.column_names) != set(dataset.column_names):
        raise ValueError("Trainer dropped sample metadata")
    rows = [kept[i] for i in RepeatRandomSampler(kept, 4, seed=42)]
    validate_generation_groups(rows, 4)
    completions = [row["target_sid"] if i % 4 == 0 else "WRONG" for i, row in enumerate(rows)]
    kwargs = reward_kwargs_from_samples(rows)
    rewards = rule_reward([row["prompt"] for row in rows], completions, **kwargs)
    if rewards != [1.0, 0.0, 0.0, 0.0] * 2:
        raise ValueError("Arrow/sampler/reward label alignment failed")
    ranking = make_ranking_reward(4)([row["prompt"] for row in rows], completions, **kwargs)
    if not (ranking[0] == ranking[4] == 0.0 and all(ranking[i] < 0 for i in (1, 2, 3, 5, 6, 7))):
        raise ValueError("Ranking reward differs from the expected original penalties")

    # Execute the actual trainer's unchanged group-normalization assignments on CPU tensors.
    tree = ast.parse((ROOT / "minionerec_trainer.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReReTrainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_inputs")
    wanted = {"mean_grouped_rewards", "std_grouped_rewards", "advantages"}
    assignments = []
    for node in method.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names & wanted:
            assignments.append(node)
        # Later code slices advantages for this process. This fixture checks the
        # group normalization only, before distributed slicing.
        if "advantages" in names:
            break
    if len(assignments) != 5:
        raise ValueError("Trainer normalization changed; review the CPU fixture")
    scope = {"rewards": torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.]),
             "self": SimpleNamespace(num_generations=4)}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "trainer:normalization", "exec"), scope)
    advantages = scope["advantages"]
    if (not torch.isfinite(advantages).all() or not torch.equal(advantages[4:], torch.zeros(4))
            or not 1.49 < advantages[0].item() < 1.51 or not -0.51 < advantages[1].item() < -0.49):
        raise ValueError("Group advantage or constant-reward behavior failed")
    with TemporaryDirectory() as folder:
        GRPOConfig(output_dir=folder, use_cpu=True, bf16=False, fp16=False,
                   per_device_train_batch_size=4, per_device_eval_batch_size=4,
                   num_generations=4, report_to=[], use_vllm=False)
    from grpo_utils import gated_prefix_signals, make_gated_prefix_reward
    partials = ["<a_1><b_2><c_4>", "<a_1><b_9><c_4>", "<a_9><b_2><c_4>", "WRONG"]
    bonus, scales, _ = gated_prefix_signals(["p"]*4, partials, [x]*4, ["sid_next"]*4, 4)
    if bonus != [.5, .2, 0., 0.]:
        raise ValueError("Gated prefix callback alignment failed")
    raw = torch.tensor(bonus)
    normalized = (raw-raw.mean())/(raw.std()+1e-4)
    scaled = normalized * torch.tensor(scales)
    if not torch.allclose(scaled, normalized*.1):
        raise ValueError("Prefix post-normalization scaling failed")
    from collections import defaultdict
    from grpo_diagnostics import record_kl_event
    with TemporaryDirectory() as folder:
        recorder = SimpleNamespace(args=SimpleNamespace(output_dir=folder),
                                   accelerator=SimpleNamespace(process_index=0),
                                   state=SimpleNamespace(global_step=12),
                                   model=SimpleNamespace(training=True), _metrics=defaultdict(list))
        diagnostic_inputs = {"completion_mask": torch.tensor([[1, 0]]),
                             "completion_ids": torch.tensor([[3, 0]])}
        # A masked padding spike must not trigger a record.
        record_kl_event(recorder, diagnostic_inputs, torch.zeros(1, 2),
                        torch.zeros(1, 2), torch.tensor([[0., 1e12]]))
        if list(Path(folder).glob("*.jsonl")):
            raise ValueError("KL recorder included masked padding")
        record_kl_event(recorder, diagnostic_inputs, torch.tensor([[-20., 0.]]),
                        torch.tensor([[-1., 0.]]), torch.tensor([[20000., 0.]]))
        event = json.loads((Path(folder)/"kl_events_rank0.jsonl").read_text())
        if event["tokens"][0]["token_id"] != 3:
            raise ValueError("KL recorder token alignment failed")
    return {"status": "PASS", "torch": torch.__version__, "transformers": transformers.__version__,
            "trl": trl.__version__, "arrow_sampler_reward_alignment": True,
            "trainer_column_retention": True, "actual_cpu_advantages": advantages.tolist(),
            "cpu_grpo_config": True,
            "gated_prefix_reward_and_scale": True,
            "kl_event_recorder_cpu": True,
            "scope": "Imports, Arrow rows, actual sampler, label columns, rewards and CPU advantage math. No model loaded, GPU generation or training performed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output filename")
    try:
        report = check()
    except Exception as error:
        report = {"status": "ERROR", "error": f"{type(error).__name__}: {error}",
                  "scope": "CPU preflight only; no model download, loading or training"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
