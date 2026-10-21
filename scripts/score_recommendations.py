"""Score raw ranked SID predictions without requiring a GPU or ML packages."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def normalize(value):
    if not isinstance(value, str):
        raise ValueError("Expected a SID string")
    return value.strip().strip('"').strip()


def score_predictions(records, catalog_counts, ks=(5, 10)):
    if not records or not catalog_counts:
        raise ValueError("Predictions and catalog must be nonempty")
    ks = tuple(sorted(set(ks)))
    if not ks or min(ks) < 1:
        raise ValueError("K must be positive")
    totals = {k: Counter() for k in ks}
    unique_targets = {k: Counter() for k in ks}
    unique_target_queries = 0
    collision_queries = 0
    for row in records:
        target = row["output"]
        if isinstance(target, list):
            if len(target) != 1:
                raise ValueError("This protocol requires one target per query")
            target = target[0]
        target = normalize(target)
        if target not in catalog_counts:
            raise ValueError(f"Target not in SID catalog: {target}")
        predictions = row["predict"]
        if not isinstance(predictions, list):
            raise ValueError("predict must be a ranked list")
        predictions = [normalize(value) for value in predictions]
        unique_target = catalog_counts[target] == 1
        unique_target_queries += int(unique_target)
        collision_queries += int(not unique_target)
        for k in ks:
            top = predictions[:k]
            rank = next((i + 1 for i, sid in enumerate(top) if sid == target), None)
            hit = int(rank is not None)
            ndcg = 1 / math.log2(rank + 1) if rank else 0.0
            totals[k]["hits"] += hit
            totals[k]["ndcg"] += ndcg
            # Count every position, including positions AFTER the first hit.
            missing = max(0, k - len(top))
            totals[k]["missing"] += missing
            totals[k]["invalid"] += sum(sid not in catalog_counts for sid in top) + missing
            totals[k]["duplicates"] += len(top) - len(set(top))
            if unique_target:
                unique_targets[k]["hits"] += hit
                unique_targets[k]["ndcg"] += ndcg

    n = len(records)
    result = {
        "queries": n,
        "metric_unit": "SID exact match; raw ranks retained, no deduplication or history filtering",
        "catalog_items": sum(catalog_counts.values()),
        "catalog_unique_sids": len(catalog_counts),
        "target_collision_queries": collision_queries,
        "target_collision_query_fraction": collision_queries / n,
        "invalid_definition": "out-of-catalog or missing slots divided by queries*K",
    }
    subset = {"queries": unique_target_queries}
    for k in ks:
        result[f"HR@{k}"] = totals[k]["hits"] / n
        result[f"NDCG@{k}"] = totals[k]["ndcg"] / n
        result[f"invalid_sid@{k}"] = totals[k]["invalid"] / (n * k)
        result[f"missing_slots@{k}"] = totals[k]["missing"] / (n * k)
        result[f"duplicate_slots@{k}"] = totals[k]["duplicates"] / (n * k)
        subset[f"HR@{k}"] = unique_targets[k]["hits"] / unique_target_queries if unique_target_queries else None
        subset[f"NDCG@{k}"] = unique_targets[k]["ndcg"] / unique_target_queries if unique_target_queries else None
    result["unique_target_sid_subset"] = subset
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--catalog", required=True, help="SID-tab-title-tab-item_id info file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10])
    args = parser.parse_args()
    records = json.loads(Path(args.predictions).read_text())
    with open(args.catalog) as handle:
        catalog = Counter(normalize(line.split("\t")[0]) for line in handle if line.strip())
    result = score_predictions(records, catalog, args.ks)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
