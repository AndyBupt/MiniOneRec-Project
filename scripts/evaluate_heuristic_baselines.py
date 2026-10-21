"""Valid-only SID baselines: train-target popularity and last-item plus popularity.

Uses no GPU and no validation targets to construct recommendation lists.
"""

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from score_recommendations import score_predictions


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    root = Path(__file__).resolve().parents[1]
    stem = "Industrial_and_Scientific_5_2016-10-2018-11"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=root / "data/Amazon/train" / f"{stem}.csv")
    parser.add_argument("--valid", type=Path, default=root / "data/Amazon/valid" / f"{stem}.csv")
    parser.add_argument("--catalog", type=Path, default=root / "data/Amazon/info" / f"{stem}.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = read_csv(args.train)
    valid = read_csv(args.valid)
    with args.catalog.open() as handle:
        catalog = Counter(line.split("\t")[0].strip() for line in handle if line.strip())
    counts = Counter(row["item_sid"].strip() for row in train)
    ranked = sorted(counts, key=lambda sid: (-counts[sid], sid))
    if len(ranked) < 10:
        raise ValueError("Need at least 10 training-target SIDs for the fixed Top-10 protocol")

    methods = {}
    for name in ("train_target_popularity", "last_item_then_popularity"):
        predictions = []
        for row in valid:
            history = ast.literal_eval(row["history_item_sid"])
            candidates = []
            if name == "last_item_then_popularity" and history:
                candidates.append(history[-1])
            for sid in ranked:
                if sid not in candidates:
                    candidates.append(sid)
                if len(candidates) == 10:
                    break
            predictions.append({"output": row["item_sid"], "predict": candidates})
        methods[name] = score_predictions(predictions, catalog)

    sid_repeats = sum(
        bool(history := ast.literal_eval(row["history_item_sid"])) and history[-1] == row["item_sid"]
        for row in valid
    )
    item_repeats = sum(
        bool(history := ast.literal_eval(row["history_item_id"])) and str(history[-1]) == str(row["item_id"])
        for row in valid
    )
    result = {
        "protocol": "full valid; train-target SID counts only; descending frequency and lexicographic tie break; Top-10; no history exclusion",
        "train_rows": len(train),
        "valid_rows": len(valid),
        "input_sha256": {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in (("train", args.train), ("valid", args.valid), ("catalog", args.catalog))
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).with_name("score_recommendations.py").read_bytes()).hexdigest(),
        "valid_target_equals_last_history_sid": sid_repeats,
        "valid_target_equals_last_history_item": item_repeats,
        "train_popular_top10": [{"sid": sid, "train_target_count": counts[sid]} for sid in ranked[:10]],
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, values in methods.items():
        print(name, {key: values[key] for key in ("HR@5", "HR@10", "NDCG@5", "NDCG@10")})
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
