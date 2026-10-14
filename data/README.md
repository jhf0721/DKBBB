# Data manifests

Datasets and historical server-specific manifests are not redistributed. Obtain the data under their original terms and run `build_manifest.py` from the repository root.

Each non-empty JSONL line must contain one paired case:

```json
{"id": "example_case", "image": "/your/data/images/example_case.jpg", "mask": "/your/data/masks/example_case_segmentation.png"}
```

The example ID is a placeholder, not a real subject. `build_manifest.py` writes absolute paths for the user's own downloaded files. These user-generated manifests should not be committed: they may expose local paths and dataset identifiers.

Use `isic2018_train.jsonl`, `isic2018_val.jsonl`, `isic2018_test.jsonl` or their `isic2016_*` equivalents. IDs and image/mask paths must be unique within each split and disjoint across splits. Use `tools/check_splits.py` to check this, optionally checking actual file existence with `--check-files`.

The current manuscript uses ISIC 2018 2594/100/1000 and ISIC 2016 720/180/379. Exact historical case-ID lists must be checked before claiming paper reproduction. To create a starter ISIC 2016 split from the official 900 training cases:

```bash
python build_manifest.py --image-dir datasets/ISIC2016/official_train/images --mask-dir datasets/ISIC2016/official_train/masks --output data/isic2016_official_train.jsonl
python tools/split_manifest.py --input data/isic2016_official_train.jsonl --train-output data/isic2016_train.jsonl --val-output data/isic2016_val.jsonl --val-fraction 0.2 --seed 42
```

The split script refuses to overwrite existing manifests. It does not claim to reproduce the historical manuscript split. Build the separate official test manifest and audit all three splits before training. Bridge generation must use training pairs only; validation and test labels must not condition exported training augmentation.
