"""Create a reproducible starter train/validation split, not historical paper IDs."""

import argparse
import json
import random
from pathlib import Path

from check_splits import check_splits, read_manifest


def split_records(records, val_fraction=0.2, seed=42):
    if not 0 < val_fraction < 1 or len(records) < 2:
        raise ValueError("Need at least two cases and 0 < val_fraction < 1")
    check_splits({"all": records})
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = min(len(records) - 1, max(1, round(len(records) * val_fraction)))
    val, train = shuffled[:val_count], shuffled[val_count:]
    check_splits({"train": train, "val": val})
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_output.resolve() == args.val_output.resolve():
        raise ValueError("Train and validation output paths must differ")
    for path in (args.train_output, args.val_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    train, val = split_records(read_manifest(args.input), args.val_fraction, args.seed)
    for path, rows in ((args.train_output, train), (args.val_output, val)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"train": len(train), "val": len(val), "seed": args.seed,
                      "scope": "starter split; not verified historical manuscript case IDs"}))


if __name__ == "__main__":
    main()
