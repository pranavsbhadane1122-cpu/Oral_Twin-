"""Phase 5 delta checks (run with: python -m unittest tests.test_delta -v).

  - identical masks report ~0% change and no notable finding
  - a mask dilated by a known factor recovers that area change within tolerance
  - empty vs non-empty reports a "new lesion"; the reverse reports "resolved"
  - confidence below the floor is refused with a reason and no numbers
  - the report JSON schema is stable and serializable
"""

import json
import unittest

import cv2
import numpy as np

from src.delta.compare import DISCLAIMER, compare
from src.utils.config import load_config

CFG = load_config()
DCFG = CFG["delta"]
SIZE = 224


def disc(radius, center=(112, 112), size=SIZE):
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, center, int(radius), 255, -1)
    return mask


def flat_image(bgr=(120, 140, 200), size=SIZE):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = bgr
    return img


class TestNoChange(unittest.TestCase):
    def test_identical_masks_report_no_change(self):
        mask = disc(40)
        img = flat_image()
        report = compare(mask, mask.copy(), img, img.copy(), 0.9, CFG)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(report["lesions"]), 1)
        lesion = report["lesions"][0]
        self.assertEqual(lesion["state"], "matched")
        self.assertAlmostEqual(lesion["area_change_pct"], 0.0, places=6)
        self.assertFalse(lesion["notable_area_change"])
        self.assertFalse(report["summary"]["any_notable_change"])
        self.assertIn("no notable change", lesion["message"].lower())


class TestKnownGrowth(unittest.TestCase):
    def test_area_change_recovered(self):
        r1 = 30.0
        for factor in (1.2, 1.5):
            with self.subTest(factor=factor):
                mask1, mask2 = disc(r1), disc(r1 * factor)
                img = flat_image()
                report = compare(mask1, mask2, img, img.copy(), 0.9, CFG)
                lesion = report["lesions"][0]

                expected = 100.0 * (factor ** 2 - 1.0)  # area scales with radius^2
                self.assertAlmostEqual(lesion["area_change_pct"], expected, delta=3.0)
                self.assertTrue(lesion["notable_area_change"])
                self.assertIn("checked by a dentist", lesion["message"])

    def test_shrinkage_reported_as_such(self):
        report = compare(disc(45), disc(30), flat_image(), flat_image(), 0.9, CFG)
        lesion = report["lesions"][0]
        self.assertLess(lesion["area_change_pct"], 0)
        self.assertIn("shrank", lesion["message"])


class TestNewAndResolved(unittest.TestCase):
    def test_empty_to_lesion_is_new(self):
        empty = np.zeros((SIZE, SIZE), np.uint8)
        report = compare(empty, disc(35), flat_image(), flat_image(), 0.9, CFG)
        self.assertEqual(report["summary"]["n_new"], 1)
        self.assertEqual(report["summary"]["n_resolved"], 0)
        lesion = report["lesions"][0]
        self.assertEqual(lesion["state"], "new")
        self.assertIn("new area", lesion["message"])
        self.assertIn("checked by a dentist", lesion["message"])

    def test_lesion_to_empty_is_resolved(self):
        empty = np.zeros((SIZE, SIZE), np.uint8)
        report = compare(disc(35), empty, flat_image(), flat_image(), 0.9, CFG)
        self.assertEqual(report["summary"]["n_resolved"], 1)
        self.assertEqual(report["lesions"][0]["state"], "resolved")

    def test_subthreshold_component_ignored(self):
        tiny = int(np.sqrt(DCFG["min_lesion_area_px"] / np.pi) * 0.5)
        report = compare(np.zeros((SIZE, SIZE), np.uint8), disc(tiny),
                         flat_image(), flat_image(), 0.9, CFG)
        self.assertEqual(report["lesions"], [])
        self.assertFalse(report["summary"]["any_notable_change"])


class TestConfidenceFloor(unittest.TestCase):
    def test_low_confidence_refused(self):
        low = DCFG["confidence_floor"] - 0.05
        report = compare(disc(30), disc(45), flat_image(), flat_image(), low, CFG)

        self.assertEqual(report["status"], "unreliable")
        self.assertIsInstance(report["reason"], str)
        self.assertIn("confidence", report["reason"])
        self.assertEqual(report["lesions"], [], "no numbers may be reported when refused")
        self.assertFalse(report["summary"]["comparable"])

    def test_at_floor_is_compared(self):
        report = compare(disc(30), disc(45), flat_image(), flat_image(),
                         DCFG["confidence_floor"], CFG)
        self.assertEqual(report["status"], "ok")


class TestSchema(unittest.TestCase):
    def test_report_schema_stable_and_serializable(self):
        report = compare(disc(30), disc(40), flat_image(), flat_image(), 0.8, CFG)
        for key in ("status", "reason", "alignment_confidence", "reliability",
                    "lesions", "summary", "disclaimer"):
            self.assertIn(key, report)
        self.assertEqual(report["disclaimer"], DISCLAIMER)

        for key in ("comparable", "n_lesions_visit1", "n_lesions_visit2", "n_matched",
                    "n_new", "n_resolved", "n_notable_changes", "any_notable_change",
                    "headline"):
            self.assertIn(key, report["summary"])

        for key in ("id", "state", "area_visit1_px", "area_visit2_px", "area_change_pct",
                    "notable_area_change", "color_shift", "edge_irregularity",
                    "confidence", "reliability", "message"):
            self.assertIn(key, report["lesions"][0])

        round_tripped = json.loads(json.dumps(report))
        self.assertEqual(round_tripped["summary"]["n_matched"], 1)

    def test_no_diagnostic_language(self):
        report = compare(disc(30), disc(45), flat_image(), flat_image(), 0.8, CFG)
        text = json.dumps(report).lower()
        for word in ("diagnosis", "diagnosed", "cancer", "malignant", "you have"):
            self.assertNotIn(word, text, f"diagnostic language '{word}' in report")
        self.assertIn("checked by a dentist", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
