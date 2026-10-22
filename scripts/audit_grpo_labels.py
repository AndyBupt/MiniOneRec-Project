"""Read-only CPU label audit using actual RL Dataset methods, without ML imports.

AST extraction bypasses pandas/torch constructors. It checks label/prompt methods,
not GPU generation, Arrow conversion, or the full Trainer runtime.
"""

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grpo_utils import make_rl_record, rule_reward, reward_kwargs_from_samples, validate_generation_groups


class Rows:
    def __init__(self, values):
        self.values = values
        self.iloc = self

    def __getitem__(self, index):
        return dict(self.values[index])


def dataset_methods(class_name):
    """Compile actual get_history/pre/prompt methods without importing data.py."""
    tree = ast.parse((ROOT / "data.py").read_text())
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    names = {"pre", "get_history", "generate_prompt", "generate_formatted_prompt"}
    methods = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if not any(node.name == "generate_prompt" for node in methods):
        base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseDataset")
        methods.append(next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "generate_prompt"))
    module = ast.Module(body=[ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"ast": ast, "make_rl_record": make_rl_record}
    exec(compile(module, "data.py:actual_rl_methods", "exec"), scope)
    return scope[class_name]


def materialize_sid(rows):
    obj = dataset_methods("SidDataset")()
    obj.data, obj.prompt2history, obj.history2target = Rows(rows), {}, {}
    return obj, [obj.pre(i) for i in range(len(rows))]


def main():
    stem = "Industrial_and_Scientific_5_2016-10-2018-11.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/Amazon/train" / stem)
    parser.add_argument("--valid", type=Path, default=ROOT / "data/Amazon/valid" / stem)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.train.open(newline="") as handle:
        train = list(csv.DictReader(handle))
    with args.valid.open(newline="") as handle:
        valid = list(csv.DictReader(handle))
    old_train_map, train_records = materialize_sid(train)
    old_valid_map, valid_records = materialize_sid(valid)
    keys = ["::".join(ast.literal_eval(row["history_item_sid"])) for row in train]
    mismatch_before = sum(old_train_map.history2target[key] != row["item_sid"] + "\n" for key, row in zip(keys, train))
    merged = {**old_train_map.history2target, **old_valid_map.history2target}
    changed_after_merge = sum(merged[key] != old_train_map.history2target[key] for key in keys)
    corrected_mismatch = sum(record["target_sid"] != row["item_sid"] + "\n" for record, row in zip(train_records, train))
    # Deliberately poison legacy maps: repaired rewards use only row-local metadata.
    old_train_map.history2target.update(old_valid_map.history2target)
    old_train_map.history2target.update({key: "WRONG_LABEL" for key in keys})
    for records in (train_records, valid_records):
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            rewards = rule_reward([r["prompt"] for r in batch], [r["completion"] for r in batch],
                                  **reward_kwargs_from_samples(batch))
            if rewards != [1.0] * len(batch):
                raise ValueError("Row-local reward uses an incorrect label")
    for row in train_records[:32]:
        validate_generation_groups([row] * 4, 4)
    if corrected_mismatch:
        raise ValueError("Actual SidDataset.pre still has row-label mismatch")
    result = {"scope": "Actual SidDataset methods, all train/valid rows; CPU without ML constructors. Not full GPU/Arrow integration.",
              "train_rows": len(train), "valid_rows": len(valid),
              "legacy_train_only_lookup_mismatches": mismatch_before,
              "legacy_train_rows_with_target_changed_by_valid_merge": changed_after_merge,
              "repaired_train_row_target_mismatches": corrected_mismatch,
              "all_own_completions_reward_one_after_legacy_map_poisoning": True,
              "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                               (("train", args.train), ("valid", args.valid))},
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                              ("data.py", "grpo_utils.py", "rl.py", "minionerec_trainer.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
