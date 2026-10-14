"""CPU release checks; no dataset download, checkpoint load or GPU execution."""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from check_splits import check_splits
from check_release import audit
from split_manifest import split_records
from dual_bbdm.legacy import ensure_legacy_ddbm_on_path
from dual_bbdm.priors import build_prior_stack


def record(index):
    return {"id": f"fixture_{index}", "image": f"fixture/images/{index}.jpg",
            "mask": f"fixture/masks/{index}.png"}


class ReleasePortabilityTests(unittest.TestCase):
    def test_default_bridge_is_bundled(self):
        with patch.dict(os.environ, {"DIFF_BBDM_ROOT": ""}):
            self.assertEqual(ensure_legacy_ddbm_on_path(), ROOT / "vendor/diff_bbdm")

    def test_prior_has_five_finite_normalized_channels(self):
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 10:22] = 1
        prior = build_prior_stack(image, mask)["prior"]
        self.assertEqual(prior.shape, (5, 32, 32))
        self.assertTrue(np.isfinite(prior).all())
        self.assertGreaterEqual(float(prior.min()), 0.0)
        self.assertLessEqual(float(prior.max()), 1.0)
        self.assertGreater(prior[0, 16, 16], 0.5)
        self.assertLess(prior[0, 0, 0], 0.5)

    def test_split_is_reproducible_disjoint_and_preserves_input(self):
        rows = [record(index) for index in range(10)]
        train, val = split_records(rows, seed=42)
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertEqual((train, val), split_records(rows, seed=42))
        self.assertEqual(rows, [record(index) for index in range(10)])
        self.assertEqual(check_splits({"train": train, "val": val})["status"], "passed")

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_splits({"train": [record(1), record(1)]})

    def test_rejects_cross_split_leakage(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            check_splits({"train": [record(1)], "val": [record(1)]})

    def test_configs_are_portable_and_score_only(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            with self.subTest(path=path.name):
                text = path.read_text()
                self.assertNotIn("/home/", text)
                self.assertNotIn("/mnt/", text)
                cfg = yaml.safe_load(text)
                self.assertNotIn("segmentation", cfg)
                self.assertNotIn("segmentor_train", cfg)
                self.assertEqual(cfg["data"]["val_manifest_path"].split("/")[-1],
                                 cfg["data"]["manifest_path"].split("/")[-1].replace("_train", "_val"))
                self.assertEqual(cfg["export"]["sample_steps"], 3)
                self.assertEqual(cfg["inference"]["bridge_steps"], 3)

    def test_only_core_model_is_exported(self):
        import dual_bbdm
        self.assertFalse((ROOT / "vendor/seg").exists())
        self.assertNotIn("build_segmentor", dual_bbdm.__all__)
        self.assertNotIn("SimpleUNet", dual_bbdm.__all__)
        self.assertIn("refine_masks", dual_bbdm.__all__)

    def test_vendor_sources_are_not_ignored_by_audit(self):
        with tempfile.TemporaryDirectory(prefix="dkb3-audit-fixture-") as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/SNAPSHOT_MANIFEST.json").write_text(json.dumps({"copied_source_files": []}))
            runtime = root / "vendor/diff_bbdm/ddbm"
            runtime.mkdir(parents=True)
            (runtime / "invalid.py").write_text("print(\n")
            result = audit(root)
            self.assertEqual(result["python_files"], 1)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("Syntax error" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
