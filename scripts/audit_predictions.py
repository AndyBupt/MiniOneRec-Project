"""CPU-only alignment, candidate concentration and cohort audit for SID predictions."""

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from score_recommendations import normalize, score_predictions


def main():
    root = Path(__file__).resolve().parents[1]
    stem = "Industrial_and_Scientific_5_2016-10-2018-11"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--valid", type=Path, default=root / "data/Amazon/valid" / f"{stem}.csv")
    parser.add_argument("--catalog", type=Path, default=root / "data/Amazon/info" / f"{stem}.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text())
    with args.valid.open(newline="") as handle:
        valid = list(csv.DictReader(handle))
    with args.catalog.open() as handle:
        catalog = Counter(line.split("\t")[0].strip() for line in handle if line.strip())
    seen = set()
    cohorts = {"seen_raw_item_in_history": [], "new_raw_item_in_history": []}
    candidates = Counter()
    top1 = Counter()
    lists = Counter()
    for record in records:
        index = record["source_row_index"]
        if index in seen or not 0 <= index < len(valid):
            raise ValueError(f"Duplicate or invalid source row: {index}")
        seen.add(index)
        row = valid[index]
        history = ast.literal_eval(row["history_item_sid"])
        expected = (
            f"The user has interacted with items {', '.join(history)} in chronological order. "
            "Can you predict the next possible item that the user may expect?"
        )
        if (str(record["user_id"]) != row["user_id"]
                or str(record["target_item_id"]) != row["item_id"]
                or normalize(record["output"]) != normalize(row["item_sid"])
                or record["input"] != expected):
            raise ValueError(f"Prediction/CSV alignment failed at row {index}")
        ranked = [normalize(sid) for sid in record["predict"][:10]]
        candidates.update(ranked)
        if ranked:
            top1.update(ranked[:1])
        lists.update([tuple(ranked)])
        item_history = {str(item) for item in ast.literal_eval(row["history_item_id"])}
        key = "seen_raw_item_in_history" if row["item_id"] in item_history else "new_raw_item_in_history"
        cohorts[key].append(record)
    result = {
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                         (("predictions", args.predictions), ("valid", args.valid), ("catalog", args.catalog))},
        "alignment": {"matched_queries": len(records), "full_valid_coverage": len(seen) == len(valid)},
        "metrics": score_predictions(records, catalog),
        "concentration": {
            "unique_sids_in_top10": len(candidates),
            "catalog_sid_coverage_at10": len(candidates) / len(catalog),
            "unique_top1_sids": len(top1),
            "unique_ordered_top10_lists": len(lists),
            "top1_frequencies": [{"sid": sid, "queries": count, "fraction": count / len(records)}
                                 for sid, count in top1.most_common()],
        },
        "cohorts": {name: score_predictions(rows, catalog) if rows else None for name, rows in cohorts.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
