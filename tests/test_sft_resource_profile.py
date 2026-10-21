"""Test profile recording and resource stop behavior without importing GPU dependencies."""

import ast
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProfileTest(unittest.TestCase):
    def callback(self, folder, reserved=30):
        tree = ast.parse((ROOT / "sft.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SFTResourceProfile")
        cuda = SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 28 * 2**30,
                               max_memory_reserved=lambda: reserved * 2**30)
        ticks = iter([100.0, 110.0])
        scope = {"transformers": SimpleNamespace(TrainerCallback=object),
                 "torch": SimpleNamespace(cuda=cuda, is_tensor=lambda value: hasattr(value, "dtype")),
                 "time": SimpleNamespace(perf_counter=lambda: next(ticks)),
                 "json": json, "os": os, "Counter": Counter}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "sft.py", "exec"), scope)
        return scope["SFTResourceProfile"](folder, 10, 64, 1000)

    def kwargs(self):
        tensor = SimpleNamespace(dtype="torch.float32")
        return {"model": SimpleNamespace(parameters=lambda: [tensor, tensor]),
                "optimizer": SimpleNamespace(state={0: {"step": tensor, "exp_avg": tensor, "exp_avg_sq": tensor}})}

    def test_profile_and_optimizer_dtype(self):
        with TemporaryDirectory() as folder:
            callback = self.callback(folder)
            callback.on_train_begin(None, None, None)
            callback.on_step_end(None, SimpleNamespace(global_step=9, max_steps=100), None, **self.kwargs())
            self.assertFalse((Path(folder) / "profile_step10.json").exists())
            callback.on_step_end(None, SimpleNamespace(global_step=10, max_steps=100), None, **self.kwargs())
            report = json.loads((Path(folder) / "profile_step10.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["estimated_training_seconds"], 100)
            self.assertEqual(report["optimizer_state_dtypes"], {"torch.float32": 2})

    def test_memory_gate_records_failure_before_stopping(self):
        with TemporaryDirectory() as folder:
            callback = self.callback(folder, reserved=70)
            callback.on_train_begin(None, None, None)
            with self.assertRaises(RuntimeError):
                callback.on_step_end(None, SimpleNamespace(global_step=10, max_steps=100), None, **self.kwargs())
            self.assertFalse(json.loads((Path(folder) / "profile_step10.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
