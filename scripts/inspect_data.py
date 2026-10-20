"""Read-only sanity check for MiniOneRec's bundled Amazon data.

Run from the repository root: python3 scripts/inspect_data.py
No model weights, GPU, or third-party packages are needed.
"""

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Amazon"


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="Industrial_and_Scientific")
    args = parser.parse_args()
    category = args.category
    index_path = DATA / "index" / f"{category}.index.json"
    with index_path.open(encoding="utf-8") as handle:
        indices = json.load(handle)

    sid_counts = Counter("".join(tokens) for tokens in indices.values())
    print(f"Category: {category}")
    print(f"Items: {len(indices)}; unique SID: {len(sid_counts)}")
    print(f"Collision groups: {sum(count > 1 for count in sid_counts.values())}")
    print(f"Items in collision groups: {sum(count for count in sid_counts.values() if count > 1)}")

    for split in ("train", "valid", "test"):
        path = DATA / split / f"{category}_5_2016-10-2018-11.csv"
        rows = load_rows(path)
        print(f"{split}: {len(rows)} rows; columns: {', '.join(rows[0]) if rows else '(empty)'}")
        if split == "train" and rows:
            row = rows[0]
            history = ast.literal_eval(row["history_item_sid"])
            print(f"First history SID: {history[:3]}")
            print(f"First target SID: {row['item_sid']}")
            print(f"First target is in index: {row['item_sid'] in sid_counts}")


if __name__ == "__main__":
    main()
