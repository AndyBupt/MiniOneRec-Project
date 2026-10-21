"""Regression: extracted evaluator callback must return tensor-compatible lists."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diagnose_sft", ROOT / "scripts/diagnose_sft.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConstraintExtractionTest(unittest.TestCase):
    def test_branching_paths_return_list_indices(self):
        tokenizer = type("Tokenizer", (), {"eos_token_id": 99})()
        callback = MODULE.source_constraints(
            ROOT, [[10, 11, 12, 1, 2, 3, 4], [10, 11, 12, 1, 5, 6, 4]], tokenizer, 3)
        for prefix, allowed in [([10, 11, 12], {1}), ([1], {2, 5}), ([1, 2, 3, 4], {99})]:
            result = callback(0, prefix)
            self.assertIsInstance(result, list)
            self.assertEqual(set(result), allowed)
        self.assertEqual(callback(0, [9]), [])


if __name__ == "__main__":
    unittest.main()
