"""Tests for the Phase 3 segmentation pipeline.

The rasterisation tests are hand-computed rather than compared against the
pipeline's own output - a test that asserts the code agrees with itself would
have passed happily while the annotation renderer was drawing bounding boxes
instead of polygons.

Run with: python -m unittest tests.test_segmentation -v
"""

import csv
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import (  # noqa: E402
    CAVITY,
    LESION,
    SPLITS,
    cache_dir,
    polygon_rings,
    rasterise,
    read_manifest,
    ring_area,
)
from src.utils.config import load_config  # noqa: E402


class TestFlatPolygonParsing(unittest.TestCase):
    """The `segmentation` field is FLAT [x1,y1,x2,y2,...], not COCO's nesting."""

    def test_flat_segmentation_is_one_ring(self):
        rings = polygon_rings({"segmentation": [0, 0, 10, 0, 10, 10, 0, 10]})
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0].shape, (4, 2))
        self.assertEqual(rings[0][1].tolist(), [10.0, 0.0])

    def test_flat_list_is_not_mistaken_for_a_bounding_box(self):
        """An L-shape must stay an L-shape, not become its bounding rectangle."""
        rings = polygon_rings(
            {"segmentation": [0, 0, 100, 0, 100, 50, 50, 50, 50, 100, 0, 100]})
        mask = rasterise(rings, (100, 100), 100)
        self.assertEqual(mask[10, 10], 255)       # inside the L
        self.assertEqual(mask[90, 90], 0)         # the notch a bbox would fill

    def test_nested_segmentation_still_parses(self):
        rings = polygon_rings({"segmentation": [[0, 0, 4, 0, 4, 4],
                                                [8, 8, 12, 8, 12, 12]]})
        self.assertEqual(len(rings), 2)

    def test_absent_or_too_short(self):
        self.assertEqual(polygon_rings({}), [])
        self.assertEqual(polygon_rings({"segmentation": []}), [])
        self.assertEqual(polygon_rings({"segmentation": [1, 2, 3, 4]}), [])


class TestRasterisation(unittest.TestCase):
    """Hand-computed cases: the expected answers are worked out, not recorded."""

    def test_square_fills_the_expected_area(self):
        # a 50x50 square inside a 100x100 image, rasterised at 100px -> ~2500 px
        rings = polygon_rings({"segmentation": [10, 10, 60, 10, 60, 60, 10, 60]})
        mask = rasterise(rings, (100, 100), 100)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertLessEqual(set(np.unique(mask).tolist()), {0, 255})
        filled = int((mask > 0).sum())
        self.assertTrue(2400 <= filled <= 2700, filled)
        self.assertEqual(mask[30, 30], 255)       # [row, col] = [y, x]: inside
        self.assertEqual(mask[5, 5], 0)           # outside

    def test_coordinates_scale_from_source_to_output_size(self):
        # the same polygon in a 200x200 source covers the top-left quarter
        rings = polygon_rings({"segmentation": [0, 0, 100, 0, 100, 100, 0, 100]})
        mask = rasterise(rings, (200, 200), 100)
        self.assertEqual(mask[10, 10], 255)
        self.assertEqual(mask[90, 90], 0)
        self.assertTrue(2300 <= int((mask > 0).sum()) <= 2700)

    def test_non_square_source_scales_each_axis_separately(self):
        # full width, half height, in a 400x200 source
        rings = polygon_rings({"segmentation": [0, 0, 400, 0, 400, 100, 0, 100]})
        mask = rasterise(rings, (400, 200), 100)
        self.assertEqual(mask[10, 50], 255)       # upper half filled
        self.assertEqual(mask[90, 50], 0)         # lower half empty

    def test_shoelace_area(self):
        ring = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        self.assertAlmostEqual(ring_area(ring), 100.0, places=6)


class TestDegeneratePolygons(unittest.TestCase):
    """N-226-01 and C-24-7-5 carry zero-area cavity rings. Skip, do not raise."""

    def test_zero_area_polygon_is_skipped_gracefully(self):
        rings = polygon_rings({"segmentation": [5, 5, 5, 5, 5, 5, 5, 5]})
        self.assertFalse(rasterise(rings, (100, 100), 100).any())

    def test_collinear_polygon_is_skipped(self):
        rings = polygon_rings({"segmentation": [0, 0, 10, 0, 20, 0, 30, 0]})
        self.assertFalse(rasterise(rings, (100, 100), 100).any())

    def test_degenerate_ring_does_not_suppress_a_valid_one(self):
        rings = (polygon_rings({"segmentation": [5, 5, 5, 5, 5, 5]})
                 + polygon_rings({"segmentation": [10, 10, 60, 10, 60, 60, 10, 60]}))
        self.assertTrue(rasterise(rings, (100, 100), 100).any())

    def test_zero_sized_source_returns_an_empty_mask(self):
        rings = polygon_rings({"segmentation": [0, 0, 10, 0, 10, 10]})
        self.assertFalse(rasterise(rings, (0, 0), 64).any())


class TestSplitsUnchanged(unittest.TestCase):
    """Phase 3 reuses the patient-disjoint manifests; it must not re-split."""

    def setUp(self):
        self.cfg = load_config()
        self.directory = (PROJECT_ROOT / self.cfg["paths"]["processed"]
                          / "splits" / "piyarathne")
        if not (self.directory / "train.csv").exists():
            raise unittest.SkipTest("piyarathne splits not built")

    def test_no_patient_crosses_a_split(self):
        patients = {}
        for split in SPLITS:
            for row in read_manifest(split, self.cfg):
                patients.setdefault(row["patient_id"], set()).add(split)
        crossing = {p: sorted(s) for p, s in patients.items() if len(s) > 1}
        self.assertFalse(crossing, f"patients in more than one split: {crossing}")

    def test_manifests_are_internally_consistent(self):
        for split in SPLITS:
            path = self.directory / f"{split}.csv"
            self.assertTrue(path.exists(), path)
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows, path)
            self.assertTrue(all(r["split"] == split for r in rows), path)


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.segmentation.model import build_and_compile

        cls.cfg = load_config()
        cls.model = build_and_compile(cls.cfg)

    def test_model_builds_with_the_configured_shape(self):
        size = self.cfg["segmentation"]["img_size"]
        channels = len(self.cfg["segmentation"]["channels"])
        self.assertEqual(tuple(self.model.input_shape), (None, size, size, 3))
        self.assertEqual(tuple(self.model.output_shape),
                         (None, size, size, channels))

    def test_predicted_mask_shape_and_range(self):
        size = self.cfg["segmentation"]["img_size"]
        channels = len(self.cfg["segmentation"]["channels"])
        predicted = self.model.predict(np.zeros((2, size, size, 3), np.float32),
                                       verbose=0)
        self.assertEqual(predicted.shape, (2, size, size, channels))
        self.assertGreaterEqual(predicted.min(), 0.0)
        self.assertLessEqual(predicted.max(), 1.0)

    def test_dice_of_a_mask_with_itself_is_one(self):
        import tensorflow as tf

        from src.segmentation.model import dice_coefficient, dice_loss

        mask = np.zeros((1, 16, 16, 2), np.float32)
        mask[0, 4:12, 4:12, :] = 1.0
        tensor = tf.constant(mask)
        self.assertAlmostEqual(float(dice_coefficient(tensor, tensor)), 1.0,
                               places=2)
        self.assertAlmostEqual(float(dice_loss(tensor, tensor)), 0.0, places=2)

    def test_dice_of_disjoint_masks_is_near_zero(self):
        import tensorflow as tf

        from src.segmentation.model import dice_coefficient

        truth = np.zeros((1, 16, 16, 1), np.float32)
        truth[0, 0:8, :, 0] = 1.0
        predicted = np.zeros((1, 16, 16, 1), np.float32)
        predicted[0, 8:16, :, 0] = 1.0
        value = float(dice_coefficient(tf.constant(truth), tf.constant(predicted)))
        self.assertLess(value, 0.02, value)


class TestCache(unittest.TestCase):

    def setUp(self):
        self.cfg = load_config()
        self.directory = cache_dir(self.cfg)
        self.masks = (sorted(self.directory.glob("*_mask.png"))[:5]
                      if self.directory.exists() else [])
        if not self.masks:
            raise unittest.SkipTest("segmentation cache not built")

    def test_cached_masks_are_two_channel_binary(self):
        import cv2

        size = self.cfg["segmentation"]["img_size"]
        for path in self.masks:
            mask = cv2.imread(str(path))
            self.assertEqual(mask.shape, (size, size, 3), path)
            # cv2 reads BGR, so the red (cavity) channel is index 2
            self.assertLessEqual(set(np.unique(mask[..., 2]).tolist()), {0, 255})
            self.assertFalse(mask[..., 0].any(),
                             f"blue channel should be unused: {path}")

    def test_channel_names_are_configured(self):
        self.assertEqual(self.cfg["segmentation"]["channels"],
                         ["oral_cavity", "lesion"])
        self.assertEqual((CAVITY, LESION), ("Oral Cavity", "Lesion"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
