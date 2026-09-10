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

from src.delta.compare import (
    DISCLAIMER,
    FULL,
    PARTIAL,
    compare,
    overlap_valid_mask,
)
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


def textured_image(size=SIZE, tissue=(110, 130, 205), lesion_bgr=None, lesion_mask=None):
    """Background 'tissue' with an optionally distinct lesion patch."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = tissue
    if lesion_mask is not None and lesion_bgr is not None:
        img[lesion_mask.astype(bool)] = lesion_bgr
    return img


def shift_illumination(img, gain=1.25, bias=18):
    """A global camera/lighting change: affects lesion and surroundings alike."""
    return np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


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


class TestRelativeColor(unittest.TestCase):
    """Colour must react to the lesion, not to the camera."""

    def test_global_illumination_shift_does_not_flag(self):
        mask = disc(35)
        img1 = textured_image(lesion_bgr=(85, 105, 175), lesion_mask=mask)
        img2 = shift_illumination(img1, gain=1.15, bias=20)  # whole frame changes

        report = compare(mask, mask.copy(), img1, img2, 0.9, CFG)
        color = report["lesions"][0]["color_shift"]
        self.assertTrue(color["measurable"])
        self.assertFalse(color["notable"],
                         f"global lighting change flagged as colour change: {color}")

    def test_lesion_only_color_change_flags(self):
        mask = disc(35)
        img1 = textured_image(lesion_bgr=(105, 125, 200), lesion_mask=mask)
        img2 = textured_image(lesion_bgr=(40, 45, 130), lesion_mask=mask)  # lesion only

        report = compare(mask, mask.copy(), img1, img2, 0.9, CFG)
        color = report["lesions"][0]["color_shift"]
        self.assertTrue(color["measurable"])
        self.assertTrue(color["notable"],
                        f"a real lesion colour change was missed: {color}")
        self.assertIn("checked by a dentist", report["lesions"][0]["message"])

    def test_ring_too_small_is_not_measurable(self):
        full_frame = np.full((SIZE, SIZE), 255, np.uint8)  # lesion fills the frame
        img = flat_image()
        report = compare(full_frame, full_frame.copy(), img, img.copy(), 0.9, CFG)
        color = report["lesions"][0]["color_shift"]
        self.assertFalse(color["measurable"])
        self.assertIn("surrounding tissue", color["reason"])
        self.assertNotIn("magnitude", color)


class TestOutOfFrame(unittest.TestCase):
    """A lesion only partly inside the newer photo cannot have its area compared."""

    @staticmethod
    def half_valid(cut_x=150):
        valid = np.zeros((SIZE, SIZE), np.uint8)
        valid[:, :cut_x] = 255
        return valid

    def test_lesion_crossing_boundary_is_partial(self):
        mask1 = disc(30, center=(140, 112))   # spans x 110-170, crosses x=150
        mask2 = disc(30, center=(140, 112))
        img = flat_image()
        report = compare(mask1, mask2, img, img.copy(), 0.9, CFG,
                         valid_mask=self.half_valid())

        lesion = report["lesions"][0]
        self.assertEqual(lesion["comparability"], PARTIAL)
        self.assertIsNone(lesion["area_change_pct"], "no area number may be reported")
        self.assertIsNone(lesion["color_shift"])
        self.assertFalse(lesion["notable_area_change"])
        self.assertIn("partly outside the newer photo", lesion["message"])
        self.assertEqual(report["summary"]["n_partial_out_of_frame"], 1)
        self.assertEqual(report["summary"]["n_comparable"], 0)

    def test_lesion_inside_boundary_is_full(self):
        mask = disc(25, center=(70, 112))     # spans x 45-95, well inside x<150
        img = flat_image()
        report = compare(mask, mask.copy(), img, img.copy(), 0.9, CFG,
                         valid_mask=self.half_valid())
        lesion = report["lesions"][0]
        self.assertEqual(lesion["comparability"], FULL)
        self.assertIsNotNone(lesion["area_change_pct"])

    def test_shrinkage_from_clipping_is_not_reported_as_change(self):
        """The pre-fix bug: mask2 clipped by the frame looked like real shrinkage."""
        mask1 = disc(30, center=(140, 112))
        clipped = mask1.copy()
        clipped[:, 150:] = 0                  # the part outside the newer photo
        img = flat_image()

        naive = compare(mask1, clipped, img, img.copy(), 0.9, CFG)
        self.assertLess(naive["lesions"][0]["area_change_pct"], -10,
                        "fixture should reproduce the spurious shrinkage")

        fixed = compare(mask1, clipped, img, img.copy(), 0.9, CFG,
                        valid_mask=self.half_valid())
        self.assertEqual(fixed["lesions"][0]["comparability"], PARTIAL)
        self.assertIsNone(fixed["lesions"][0]["area_change_pct"])
        self.assertFalse(fixed["summary"]["any_notable_change"])

    def test_overlap_valid_mask_shrinks_under_translation(self):
        H = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        valid = overlap_valid_mask((SIZE, SIZE), H, (SIZE, SIZE))
        self.assertEqual(int((valid[:, :40] > 127).sum()), 0, "shifted-in region is not shared")
        self.assertGreater(int((valid > 127).sum()), 0)
        self.assertLess(int((valid > 127).sum()), SIZE * SIZE)


class TestSchema(unittest.TestCase):
    def test_report_schema_stable_and_serializable(self):
        report = compare(disc(30), disc(40), flat_image(), flat_image(), 0.8, CFG)
        for key in ("status", "reason", "alignment_confidence", "reliability",
                    "lesions", "summary", "disclaimer"):
            self.assertIn(key, report)
        self.assertEqual(report["disclaimer"], DISCLAIMER)

        for key in ("comparable", "n_lesions_visit1", "n_lesions_visit2", "n_matched",
                    "n_new", "n_resolved", "n_comparable", "n_partial_out_of_frame",
                    "n_notable_changes", "any_notable_change", "headline"):
            self.assertIn(key, report["summary"])

        for key in ("id", "state", "comparability", "area_visit1_px", "area_visit2_px",
                    "area_change_pct", "notable_area_change", "color_shift",
                    "edge_irregularity", "confidence", "reliability", "message"):
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
