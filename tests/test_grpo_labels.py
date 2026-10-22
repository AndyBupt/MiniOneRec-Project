"""Regression tests for repeated histories, cross-split labels, grouping and rewards."""

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grpo_utils import (RL_COLUMNS, make_rl_record, sample_targets, validate_generation_groups,
                        reward_kwargs_from_samples, rule_reward, make_ranking_reward)

SPEC = importlib.util.spec_from_file_location("audit_grpo_labels", ROOT / "scripts/audit_grpo_labels.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

X, Y = "<a_1><b_2><c_3>", "<a_4><b_5><c_6>"


class GrpoLabelTest(unittest.TestCase):
    def sid_row(self, target):
        return {"history_item_sid": repr([X]), "item_sid": target}

    def test_actual_sid_dataset_same_history_different_targets(self):
        dataset, rows = AUDIT.materialize_sid([self.sid_row(X), self.sid_row(Y)])
        self.assertEqual(rows[0]["prompt"], rows[1]["prompt"])
        self.assertEqual(dataset.history2target[X], Y + "\n")  # Legacy map really did overwrite.
        dataset.history2target[X] = "VALIDATION_OR_POISON_LABEL"
        self.assertEqual(sample_targets(rows), [X + "\n", Y + "\n"])
        self.assertEqual(rule_reward([r["prompt"] for r in rows], [X, Y], **reward_kwargs_from_samples(rows)), [1.0, 1.0])
        self.assertEqual(rule_reward([r["prompt"] for r in rows], [Y, X], **reward_kwargs_from_samples(rows)), [0.0, 0.0])

    def test_actual_auxiliary_datasets_have_same_columns(self):
        item = AUDIT.dataset_methods("RLTitle2SidDataset")()
        item.data = [{"task": "title2sid", "input": "same title", "output": X},
                     {"task": "description2sid", "input": "same title", "output": Y}]
        item.prompt2history, item.history2target = {}, {}
        a, b = item.pre(0), item.pre(1)
        seq = AUDIT.dataset_methods("RLSeqTitle2SidDataset")()
        seq.data = AUDIT.Rows([dict(self.sid_row(Y), history_item_title=repr(["a title"]))])
        seq.prompt2history, seq.history2target, seq.dedup = {}, {}, False
        c = seq.pre(0)
        for row in (a, b, c):
            self.assertEqual(set(row), set(RL_COLUMNS))
            self.assertEqual(row["target_sid"], row["completion"])
        self.assertEqual(c["history_sids"], [X])
        self.assertEqual(a["history_sids"], [])

    def test_sampler_metadata_not_repeated_twice(self):
        a = make_rl_record("same prompt", X, task_type="sid_next", sample_id="a")
        b = make_rl_record("same prompt", Y, task_type="sid_next", sample_id="b")
        repeated = [a] * 4 + [b] * 4
        self.assertEqual(validate_generation_groups(repeated, 4), [X] * 4 + [Y] * 4)
        self.assertEqual(reward_kwargs_from_samples(repeated)["target_sid"], [X] * 4 + [Y] * 4)
        with self.assertRaises(ValueError):
            validate_generation_groups([a, b, a, b], 4)
        with self.assertRaises(ValueError):
            validate_generation_groups([a] * 3, 4)
        with self.assertRaises(ValueError):
            reward_kwargs_from_samples([{"prompt": "p", "completion": X}])

    def test_original_ranking_penalties_and_no_hit_behavior(self):
        reward = make_ranking_reward(2)
        penalty_second = -(1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertEqual(reward(["p"] * 2, [X, X], [X] * 2), [0.0, 0.0])
        self.assertEqual(reward(["p"] * 2, [Y, Y], [X] * 2), [0.0, 0.0])
        values = reward(["p"] * 2, [X, Y], [X] * 2)
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], penalty_second)
        self.assertEqual(rule_reward(["p"], ['"' + X + '"\n'], [X]), [1.0])
        with self.assertRaises(ValueError):
            reward(["p"] * 2, [X, Y], [X, Y])

    def test_real_trainer_retains_target_columns(self):
        tree = ast.parse((ROOT / "minionerec_trainer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReReTrainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_set_signature_columns_if_needed")
        scope = {"RL_COLUMNS": RL_COLUMNS}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer:columns", "exec"), scope)
        obj = SimpleNamespace(_signature_columns=None)
        scope[method.name](obj)
        self.assertIn("target_sid", obj._signature_columns)
        self.assertIn("sample_id", obj._signature_columns)

    def test_trainer_prepare_prefix_never_reads_legacy_maps(self):
        tree = ast.parse((ROOT / "minionerec_trainer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReReTrainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inputs")
        stop = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "prompts_text" for t in n.targets))
        method.body = method.body[:stop] + [ast.Return(value=ast.Name(id="targets", ctx=ast.Load()))]
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        ast.fix_missing_locations(method)
        scope = {"validate_generation_groups": validate_generation_groups}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer:prepare_prefix", "exec"), scope)
        obj = SimpleNamespace(accelerator=SimpleNamespace(device="cpu"), num_generations=2)
        # No history dictionaries exist on obj: access would fail this test.
        row = make_rl_record("p", X, task_type="sid_next", sample_id="a")
        self.assertEqual(scope[method.name](obj, [row, row]), [X, X])

    def test_test_generation_does_not_advance_training_processor(self):
        tree = ast.parse((ROOT / "minionerec_trainer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ReReTrainer")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inputs")
        wanted = {"ccc", "test_ccc", "self.logits_processor", "self.test_lp_list"}
        assignments = [n for n in method.body if isinstance(n, ast.Assign)
                       and any(ast.unparse(t) in wanted for t in n.targets)]

        class Processor:
            def __init__(self, **kwargs):
                self.num_beams, self.count = kwargs["num_beams"], 0

        scope = {"ConstrainedLogitsProcessor": Processor, "LogitsProcessorList": list,
                 "TemperatureLogitsWarper": lambda **kwargs: object()}
        obj = SimpleNamespace(num_generations=4, test_beam=20, beam_search=True, base_model="qwen",
                              temperature=1.0, prefix_allowed_tokens_fn=lambda *args: [],
                              processing_class=SimpleNamespace(eos_token_id=99))
        scope["self"] = obj
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "trainer:processors", "exec"), scope)
        train, test = obj.logits_processor[-1], obj.test_lp_list[-1]
        self.assertEqual(train.num_beams, 4)
        self.assertEqual(test.num_beams, 20)
        test.count += 5
        self.assertEqual(train.count, 0)


if __name__ == "__main__":
    unittest.main()
