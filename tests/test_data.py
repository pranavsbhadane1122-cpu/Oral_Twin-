"""Phase 1 sanity checks (stdlib unittest — run with:
  python -m unittest tests.test_data -v
).

Checks:
  - all processed images load
  - per-class counts on disk match the manifest written by the prepare scripts
  - no image appears in two splits (dihedral pHash within the configured
    conservative threshold)
  - every longitudinal pair has visit1/visit2 and a valid ground_truth.json
    with an invertible 3x3 homography
"""

import json
import unittest
from pathlib import Path

import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import dihedral_phashes, hamming_pairs, imread_unicode

CFG = load_config()
SPLITS = ("train", "val", "test")


def split_hashes(task_root):
    """{split: (paths, uint64 hash array)} for every image in the task."""
    out = {}
    for split in SPLITS:
        paths = sorted((task_root / split).rglob("*.jpg"))
        hashes = []
        for p in paths:
            img = imread_unicode(p)
            assert img is not None, f"unreadable image: {p}"
            hashes.extend(dihedral_phashes(img))
        out[split] = (paths, np.asarray(hashes, dtype=np.uint64))
    return out


class TaskChecks:
    """Shared checks for one processed task folder (classification / opmd)."""

    task_dir: str

    @classmethod
    def setUpClass(cls):
        cls.root = PROJECT_ROOT / CFG["paths"]["processed"] / cls.task_dir
        cls.manifest = json.loads((cls.root / "manifest.json").read_text(encoding="utf-8"))
        cls.data = split_hashes(cls.root)  # loading also asserts readability

    def test_counts_match_manifest(self):
        for split in SPLITS:
            for cls_name in self.manifest["classes"]:
                on_disk = len(list((self.root / split / cls_name).glob("*.jpg")))
                self.assertEqual(
                    on_disk,
                    self.manifest["counts"][split][cls_name],
                    f"{self.task_dir}/{split}/{cls_name}",
                )

    def test_no_cross_split_near_duplicates(self):
        threshold = CFG["dedup"]["phash_threshold"]
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        for a, b in pairs:
            paths_a, ha = self.data[a]
            paths_b, hb = self.data[b]
            if not len(ha) or not len(hb):
                continue
            leaks = set()
            for i, j in hamming_pairs(ha, hb, threshold):
                leaks.add((str(paths_a[i // 8]), str(paths_b[j // 8])))
            self.assertFalse(
                leaks, f"near-duplicates across {a}/{b} in {self.task_dir}: {sorted(leaks)[:5]}"
            )


class TestClassificationSplits(TaskChecks, unittest.TestCase):
    task_dir = "classification"


class TestOpmdSplits(TaskChecks, unittest.TestCase):
    task_dir = "opmd"


class TestLongitudinalPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = PROJECT_ROOT / CFG["paths"]["longitudinal"]
        cls.pairs = sorted(p for p in cls.root.glob("pair_*") if p.is_dir())

    def test_enough_pairs(self):
        self.assertGreaterEqual(len(self.pairs), CFG["longitudinal"]["n_pairs"])

    def test_pairs_complete_and_valid(self):
        for pair in self.pairs:
            for name in ("visit1.jpg", "visit2.jpg"):
                img = imread_unicode(pair / name)
                self.assertIsNotNone(img, f"unreadable {pair.name}/{name}")
            gt = json.loads((pair / "ground_truth.json").read_text(encoding="utf-8"))
            H = np.asarray(gt["homography"], dtype=np.float64)
            self.assertEqual(H.shape, (3, 3), pair.name)
            self.assertGreater(abs(np.linalg.det(H)), 1e-6, f"{pair.name}: singular homography")
            if gt["lesion_change"] is not None:
                for key in ("center_xy", "radius_px", "darken_factor", "enlarge_factor"):
                    self.assertIn(key, gt["lesion_change"], pair.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
