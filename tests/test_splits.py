"""Leakage guarantees for the three split sets.

These are the tests that answer "how do you know there is no leakage" in review:

  - no patient appears in two Piyarathne splits
  - no pHash cluster appears in two MIO splits
  - SMART-OM appears in no training split anywhere
  - every excluded image appears nowhere, including in the Kaggle image copies
  - SMART-OM's confounded Oral Cancer subset cannot be scored without an
    explicit opt-in

Run with: python -m unittest tests.test_splits -v
"""

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import hamming_pairs
from src.utils.smartom import (
    CONFOUNDED_CATEGORY,
    ConfoundedSubsetError,
    assert_scoreable,
    scoreable_categories,
)

CFG = load_config()
SPLITS_DIR = PROJECT_ROOT / CFG["paths"]["processed"] / "splits"
SPLITS = ("train", "val", "test")


def read_manifest(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifests(source):
    return {split: read_manifest(SPLITS_DIR / source / f"{split}.csv")
            for split in SPLITS}


def excluded_rows():
    path = PROJECT_ROOT / CFG["paths"]["processed"] / "exclusions.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPiyarathnePatientDisjoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = manifests("piyarathne")
        if not any(cls.data.values()):
            raise unittest.SkipTest("piyarathne splits not built")

    def test_no_patient_in_two_splits(self):
        patients = {s: {r["patient_id"] for r in rows} for s, rows in self.data.items()}
        for a in SPLITS:
            for b in SPLITS:
                if a >= b:
                    continue
                overlap = patients[a] & patients[b]
                self.assertEqual(overlap, set(),
                                 f"patients in both {a} and {b}: {sorted(overlap)[:5]}")

    def test_no_image_in_two_splits(self):
        seen = defaultdict(list)
        for split, rows in self.data.items():
            for row in rows:
                seen[row["path"]].append(split)
            self.assertTrue(rows)
        repeated = {p: s for p, s in seen.items() if len(s) > 1}
        self.assertEqual(repeated, {}, f"images in more than one split: "
                                       f"{list(repeated)[:5]}")

    def test_every_image_has_a_patient(self):
        for rows in self.data.values():
            for row in rows:
                self.assertTrue(row["patient_id"])
                self.assertTrue(row["path"].startswith("piyarathne_oral/"))

    def test_split_sizes_are_close_to_target(self):
        total = sum(len(rows) for rows in self.data.values())
        for split in SPLITS:
            share = len(self.data[split]) / total
            self.assertAlmostEqual(share, CFG["split"][split], delta=0.03,
                                   msg=f"{split} share {share:.3f}")


class TestMioFlatClusterDisjoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = manifests("mio_flat")
        if not any(cls.data.values()):
            raise unittest.SkipTest("mio_flat splits not built")

    def test_no_cluster_in_two_splits(self):
        clusters = {s: {r["cluster_id"] for r in rows} for s, rows in self.data.items()}
        for a in SPLITS:
            for b in SPLITS:
                if a >= b:
                    continue
                overlap = clusters[a] & clusters[b]
                self.assertEqual(overlap, set(),
                                 f"clusters in both {a} and {b}: {sorted(overlap)[:5]}")

    def test_split_unit_is_the_cluster_not_the_image(self):
        """Every image of a cluster must land in the same split."""
        split_of_cluster = {}
        for split, rows in self.data.items():
            for row in rows:
                previous = split_of_cluster.setdefault(row["cluster_id"], split)
                self.assertEqual(previous, split,
                                 f"cluster {row['cluster_id']} spans splits")

    def test_only_flat_folders_present(self):
        allowed = ("mio/GINGIVITIS/", "mio/PERIODONTITIS/", "mio/SANO/")
        for rows in self.data.values():
            for row in rows:
                self.assertTrue(row["path"].startswith(allowed), row["path"])


class TestSmartomNeverTrains(unittest.TestCase):
    def test_smartom_is_not_in_any_training_manifest(self):
        for source in ("piyarathne", "mio_flat"):
            for split, rows in manifests(source).items():
                for row in rows:
                    self.assertNotIn("SMART-OM", row["path"],
                                     f"SMART-OM leaked into {source}/{split}")

    def test_smartom_manifest_is_validation_only(self):
        path = SPLITS_DIR / "smartom_external" / "validation.csv"
        if not path.exists():
            self.skipTest("smartom manifest not built")
        rows = read_manifest(path)
        self.assertTrue(rows)
        for name in SPLITS:
            self.assertFalse((SPLITS_DIR / "smartom_external" / f"{name}.csv").exists(),
                             "SMART-OM must have no train/val/test split at all")

    def test_smartom_uses_the_unannotated_level_only(self):
        rows = read_manifest(SPLITS_DIR / "smartom_external" / "validation.csv")
        if not rows:
            self.skipTest("smartom manifest not built")
        for row in rows:
            self.assertIn("01. Unannotated", row["path"], row["path"])

    def test_confounded_category_is_flagged_and_unlabelled(self):
        rows = read_manifest(SPLITS_DIR / "smartom_external" / "validation.csv")
        if not rows:
            self.skipTest("smartom manifest not built")
        confounded = [r for r in rows if r["category"] == CONFOUNDED_CATEGORY]
        self.assertTrue(confounded)
        for row in confounded:
            self.assertEqual(row["excluded_from_metrics"], "1")
            self.assertEqual(row["lesion_present"], "",
                             "the excluded subset must carry no usable label")
            self.assertTrue(row["exclusion_reason"])


class TestConfoundedSubsetGuard(unittest.TestCase):
    def test_scoring_the_confounded_subset_raises(self):
        with self.assertRaises(ConfoundedSubsetError):
            assert_scoreable([CONFOUNDED_CATEGORY])
        with self.assertRaises(ConfoundedSubsetError):
            assert_scoreable(["01. Normal", CONFOUNDED_CATEGORY])

    def test_explicit_opt_in_is_allowed(self):
        self.assertTrue(assert_scoreable([CONFOUNDED_CATEGORY],
                                         allow_confounded_subset=True))

    def test_other_categories_score_freely(self):
        self.assertTrue(assert_scoreable(scoreable_categories(CFG)))
        self.assertNotIn(CONFOUNDED_CATEGORY, scoreable_categories(CFG))


class TestExclusionsApplied(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = excluded_rows()
        if not cls.rows:
            raise unittest.SkipTest("exclusions.csv not present")

    def test_excluded_images_are_in_no_manifest(self):
        banned = {f"{r['dataset']}/{r['path']}".replace("\\", "/") for r in self.rows}
        for source in ("piyarathne", "mio_flat"):
            for split, manifest in manifests(source).items():
                for row in manifest:
                    self.assertNotIn(row["path"], banned,
                                     f"excluded image in {source}/{split}")

    def test_excluded_images_are_in_no_smartom_manifest(self):
        banned = {f"{r['dataset']}/{r['path']}".replace("\\", "/") for r in self.rows}
        for row in read_manifest(SPLITS_DIR / "smartom_external" / "validation.csv"):
            self.assertNotIn(row["path"], banned)

    def test_excluded_kaggle_images_are_absent_from_the_image_copies(self):
        """Kaggle splits are resized copies, so paths cannot be compared: the
        check is perceptual instead."""
        from src.utils.licence_guard import image_phash

        root = PROJECT_ROOT / CFG["paths"]["raw"]
        processed = PROJECT_ROOT / CFG["paths"]["processed"] / "classification"
        if not processed.is_dir():
            self.skipTest("Kaggle splits not built")

        banned_hashes = []
        for row in self.rows:
            if row["dataset"] != "oral_diseases":
                continue
            source = root / row["dataset"] / row["path"]
            if source.exists():
                banned_hashes.append(image_phash(source))
        if not banned_hashes:
            self.skipTest("no Kaggle exclusions to check")

        copies = sorted(processed.rglob("*.jpg"))
        self.assertTrue(copies, "no Kaggle images found to check")
        copy_hashes = np.asarray([image_phash(p) for p in copies], dtype=np.uint64)
        banned = np.asarray(banned_hashes, dtype=np.uint64)

        hits = set()
        for i, j in hamming_pairs(banned, copy_hashes,
                                  CFG["dedup"]["phash_threshold"]):
            hits.add(str(copies[j].relative_to(processed)))
        self.assertEqual(hits, set(), f"excluded images survive as copies: {sorted(hits)[:5]}")

    def test_every_exclusion_carries_its_evidence(self):
        # the CSV stores NCC to four decimals, so a verifier value of 0.90004 is
        # written as 0.9000; compare at the recorded precision, not stricter
        for row in self.rows:
            self.assertTrue(row["reason"])
            self.assertGreaterEqual(float(row["evidence_ncc"]), 0.90, row["path"])
            self.assertGreaterEqual(int(row["evidence_ransac_inliers"]), 30, row["path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
