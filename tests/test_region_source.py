"""Tests for the region-source swap: cavity model vs HSV heuristic.

The point of these is the FALLBACK path. A model-backed mask that silently
degrades to something else, without saying so, is worse than no model at all -
the alignment numbers would shift and nothing in the output would explain why.
So the tests assert that the fallback fires, and that it is reported.

Run with: python -m unittest tests.test_region_source -v
"""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment import cavity_region, features  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def synthetic_mouth(size=256):
    """A dark frame with a bright pinkish ellipse: enough for the heuristic."""
    img = np.full((size, size, 3), 20, np.uint8)
    import cv2

    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 5), 0, 0, 360,
                (120, 140, 220), -1)      # BGR: a reddish-pink region
    return img


class TestRegionSourceDispatch(unittest.TestCase):

    def setUp(self):
        self.cfg = copy.deepcopy(load_config())
        self.img = synthetic_mouth()

    def test_hsv_arm_never_calls_the_model(self):
        self.cfg["alignment"]["region_source"] = "hsv_heuristic"
        called = []
        original = cavity_region.cavity_mask
        cavity_region.cavity_mask = lambda *a, **k: called.append(1) or (None, "x")
        try:
            mask, source = features.region_mask(self.img, self.cfg)
        finally:
            cavity_region.cavity_mask = original
        self.assertEqual(called, [])
        self.assertIn(source, (features.HSV, features.FULL_FRAME))
        self.assertEqual(mask.shape, self.img.shape[:2])

    def test_unsupported_region_source_raises(self):
        self.cfg["alignment"]["region_source"] = "magic"
        with self.assertRaises(ValueError) as ctx:
            features.region_mask(self.img, self.cfg)
        self.assertIn("magic", str(ctx.exception))

    def test_falls_back_and_says_so_when_the_model_is_unavailable(self):
        self.cfg["alignment"]["region_source"] = "cavity_model"
        original = cavity_region.cavity_mask
        cavity_region.cavity_mask = lambda *a, **k: (None, "checkpoint not found")
        try:
            mask, source = features.region_mask(self.img, self.cfg)
        finally:
            cavity_region.cavity_mask = original
        self.assertIsNotNone(mask)
        self.assertIn("fell back", source)
        self.assertIn("checkpoint not found", source)

    def test_uses_the_model_mask_when_it_is_usable(self):
        self.cfg["alignment"]["region_source"] = "cavity_model"
        fake = np.zeros(self.img.shape[:2], np.uint8)
        fake[50:200, 50:200] = 255
        original = cavity_region.cavity_mask
        cavity_region.cavity_mask = lambda *a, **k: (fake, None)
        try:
            mask, source = features.region_mask(self.img, self.cfg)
        finally:
            cavity_region.cavity_mask = original
        self.assertEqual(source, features.CAVITY)
        self.assertTrue((mask == fake).all())


class TestDegeneracyGuard(unittest.TestCase):

    def setUp(self):
        self.cfg = copy.deepcopy(load_config())
        self.shape = (256, 256, 3)

    def test_empty_mask_is_degenerate(self):
        mask = np.zeros(self.shape[:2], np.uint8)
        self.assertTrue(cavity_region.is_degenerate(mask, self.shape, self.cfg))

    def test_none_is_degenerate(self):
        self.assertTrue(cavity_region.is_degenerate(None, self.shape, self.cfg))

    def test_a_tiny_speck_is_degenerate(self):
        mask = np.zeros(self.shape[:2], np.uint8)
        mask[0:10, 0:10] = 255          # 100 px of 65,536 = 0.15%
        self.assertTrue(cavity_region.is_degenerate(mask, self.shape, self.cfg))

    def test_a_plausible_mouth_is_not_degenerate(self):
        mask = np.zeros(self.shape[:2], np.uint8)
        mask[40:200, 40:200] = 255      # ~39% of the frame
        self.assertFalse(cavity_region.is_degenerate(mask, self.shape, self.cfg))

    def test_threshold_comes_from_config_not_a_literal(self):
        mask = np.zeros(self.shape[:2], np.uint8)
        mask[0:80, 0:80] = 255          # 9.8% of the frame
        self.cfg["alignment"]["cavity_min_area_frac"] = 0.05
        self.assertFalse(cavity_region.is_degenerate(mask, self.shape, self.cfg))
        self.cfg["alignment"]["cavity_min_area_frac"] = 0.20
        self.assertTrue(cavity_region.is_degenerate(mask, self.shape, self.cfg))


class TestValidRegionDispatch(unittest.TestCase):

    def setUp(self):
        self.cfg = copy.deepcopy(load_config())
        self.img = synthetic_mouth()
        self.H = np.eye(3, dtype=float)

    def test_frame_intersection_is_the_default_path(self):
        from src.delta.compare import valid_region

        self.cfg["delta"]["valid_region_source"] = "frame_intersection"
        mask, source = valid_region(self.img, self.H, self.img.shape, self.cfg)
        self.assertEqual(source, "frame_intersection")
        # identity homography: the whole frame is visible in both
        self.assertTrue((mask > 0).all())

    def test_unsupported_valid_region_raises(self):
        from src.delta.compare import valid_region

        self.cfg["delta"]["valid_region_source"] = "elsewhere"
        with self.assertRaises(ValueError):
            valid_region(self.img, self.H, self.img.shape, self.cfg)

    def test_cavity_arm_falls_back_to_the_frame_when_unavailable(self):
        from src.delta.compare import valid_region

        self.cfg["delta"]["valid_region_source"] = "cavity"
        original = cavity_region.cavity_mask
        cavity_region.cavity_mask = lambda *a, **k: (None, "no checkpoint")
        try:
            mask, source = valid_region(self.img, self.H, self.img.shape, self.cfg)
        finally:
            cavity_region.cavity_mask = original
        self.assertIn("fell back", source)
        self.assertTrue((mask > 0).all())

    def test_cavity_and_frame_intersects_both(self):
        from src.delta.compare import valid_region

        self.cfg["delta"]["valid_region_source"] = "cavity_and_frame"
        fake = np.zeros(self.img.shape[:2], np.uint8)
        fake[0:100, :] = 255
        original = cavity_region.cavity_mask
        cavity_region.cavity_mask = lambda *a, **k: (fake, None)
        try:
            mask, source = valid_region(self.img, self.H, self.img.shape, self.cfg)
        finally:
            cavity_region.cavity_mask = original
        self.assertEqual(source, "cavity_and_frame")
        self.assertTrue((mask[0:100, :] > 0).all())
        self.assertFalse((mask[100:, :] > 0).any())


class TestConfiguredDefaults(unittest.TestCase):

    def test_defaults_are_what_the_pipeline_expects(self):
        cfg = load_config()
        self.assertIn(cfg["alignment"]["region_source"],
                      ("cavity_model", "hsv_heuristic"))
        self.assertIn(cfg["delta"]["valid_region_source"],
                      ("frame_intersection", "cavity", "cavity_and_frame"))
        self.assertGreater(cfg["alignment"]["cavity_min_area_frac"], 0.0)
        self.assertLessEqual(cfg["alignment"]["cavity_threshold"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
