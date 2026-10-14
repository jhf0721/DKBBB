"""Check unique IDs, image/mask paths, and disjoint single-label ISIC splits."""

import argparse
import json
from itertools import combinations
from pathlib import Path

EXPECTED = {"isic2016": {"train": 720, "val": 180, "test": 379},
            "isic2018": {"train": 2594, "val": 100, "test": 1000}}


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_splits(splits, expected=None, check_files=False):
    keys = {}
    for name, records in splits.items():
        if not records:
            raise ValueError(f"Empty split: {name}")
        keys[name] = {field: set() for field in ("id", "image", "mask")}
        for row in records:
            for field in keys[name]:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Missing/non-string {field} in {name}")
                value = row[field].strip() if field == "id" else str(Path(row[field]).expanduser().resolve())
                if value in keys[name][field]:
                    raise ValueError(f"Duplicate {field} in {name}")
                keys[name][field].add(value)
                if check_files and field != "id" and not Path(value).is_file():
                    raise FileNotFoundError(value)
        if expected and len(records) != expected[name]:
            raise ValueError(f"Expected {expected[name]} {name} cases, got {len(records)}")
    for left, right in combinations(splits, 2):
        for field in ("id", "image", "mask"):
            if keys[left][field] & keys[right][field]:
                raise ValueError(f"Overlapping {field}: {left}/{right}")
    return {"status": "passed", "counts": {name: len(rows) for name, rows in splits.items()},
            "unique_within_split": True, "disjoint_ids_and_paths": True,
            "file_existence_checked": check_files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--dataset", choices=sorted(EXPECTED))
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    splits = {name: read_manifest(getattr(args, name)) for name in ("train", "val", "test")}
    print(json.dumps(check_splits(splits, EXPECTED.get(args.dataset), args.check_files), indent=2))


if __name__ == "__main__":
    main()
