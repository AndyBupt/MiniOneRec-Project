"""CPU tests for subset eligibility and loss-mask/prompt diagnostic safeguards."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tiny", ROOT / "scripts/tiny_sft_diagnostic.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TinyDiagnosticTest(unittest.TestCase):
    def test_conflicting_labels_and_repeated_histories(self):
        rows = [
            {"history_item_sid": "['h1']", "item_sid": "s1"},
            {"history_item_sid": "['h1']", "item_sid": "s2"},
            {"history_item_sid": "['h2']", "item_sid": "s3"},
            {"history_item_sid": "['h2']", "item_sid": "s3"},
            {"history_item_sid": "['h3']", "item_sid": "s4"},
            {"history_item_sid": "[]", "item_sid": "s5"},
        ]
        selected, manifest = MODULE.select_rows(rows, 2, 42)
        self.assertEqual({row["item_sid"] for row in selected}, {"s3", "s4"})
        self.assertEqual(set(manifest["source_row_indices"]), {2, 4})
        self.assertEqual(manifest["conflicting_histories_excluded"], 1)
        self.assertEqual(MODULE.select_rows(rows, 2, 42), (selected, manifest))
        with self.assertRaises(ValueError):
            MODULE.select_rows(rows, 3, 42)

    def test_masks_and_prompt_alignment(self):
        expected = [10, 11, 12, 13, 99]
        good = {"input_ids": [1, 2] + expected,
                "labels": [-100, -100] + expected, "attention_mask": [1] * 7}
        MODULE.validate_example(good, [1, 2], expected)
        with self.assertRaises(ValueError):
            MODULE.validate_example(good, [1, 3], expected)
        for labels in ([1, -100] + expected, [-100] * 7, [-100] * 3 + expected[1:]):
            with self.assertRaises(ValueError):
                MODULE.validate_example(dict(good, labels=labels), [1, 2], expected)

    def test_sid_token_order(self):
        self.assertEqual(MODULE.sid_tokens("<a_210><b_156><c_39>"),
                         ["<a_210>", "<b_156>", "<c_39>"])
        with self.assertRaises(ValueError):
            MODULE.sid_tokens("<a_210><c_39><b_156>")


if __name__ == "__main__":
    unittest.main()
