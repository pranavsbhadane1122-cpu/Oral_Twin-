"""Phase 4 alignment checks (run with: python -m unittest tests.test_alignment -v).

  - identity: aligning an image to itself gives a near-identity H and high confidence
  - recovery: a known synthetic warp is recovered within tolerance
  - rejection: an unmatchable pair returns confidence 0 with a reason
  - corner_error is invariant to the arbitrary scale of a homography
"""

import unittest

import cv2
import numpy as np

from src.alignment.confidence import score
from src.alignment.features import detect_and_match
from src.alignment.homography import corner_error, estimate, normalize_h, warp_visit2
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode

CFG = load_config()


def a_test_image():
    """A real processed mouth photo (falls back to a longitudinal visit1)."""
    root = PROJECT_ROOT / CFG["paths"]["processed"] / "classification" / "test"
    for p in sorted(root.rglob("*.jpg")):
        img = imread_unicode(p)
        if img is not None:
            return img
    raise unittest.SkipTest("no processed test images available")


class TestIdentity(unittest.TestCase):
    def test_self_alignment_is_identity(self):
        img = a_test_image()
        pts1, pts2, _ = detect_and_match(img, img, CFG)
        H, info = estimate(pts1, pts2, CFG)
        self.assertIsNotNone(H, f"self-alignment was rejected: {info['reason']}")
        np.testing.assert_allclose(normalize_h(H), np.eye(3), atol=1e-3)

        conf, breakdown = score(info["inlier_ratio"], info["n_inliers"],
                                info["reproj_error"], 500.0, cfg=CFG)
        self.assertGreater(conf, 0.8, f"identity confidence too low: {breakdown}")
        self.assertLessEqual(conf, 1.0)


class TestRecovery(unittest.TestCase):
    def test_known_warp_recovered(self):
        img1 = a_test_image()
        h, w = img1.shape[:2]
        # known modest transform: rotate 8 deg about centre, scale 0.95, shift
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 8.0, 0.95)
        M[0, 2] += 6.0
        M[1, 2] -= 4.0
        H_true = np.vstack([M, [0, 0, 1]])          # maps img1 -> img2
        img2 = cv2.warpPerspective(img1, H_true, (w, h))

        pts1, pts2, _ = detect_and_match(img1, img2, CFG)
        H_est, info = estimate(pts1, pts2, CFG)     # maps img2 -> img1
        self.assertIsNotNone(H_est, f"rejected: {info['reason']}")

        err = corner_error(H_est, np.linalg.inv(H_true), w, h)
        self.assertLess(err, 5.0, f"corner error {err:.2f}px too high")

        warped = warp_visit2(img2, H_est, img1.shape)
        self.assertEqual(warped.shape, img1.shape)


class TestRejection(unittest.TestCase):
    def test_unmatchable_pair_scores_zero(self):
        rng = np.random.default_rng(0)
        img1 = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        img2 = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        pts1, pts2, _ = detect_and_match(img1, img2, CFG)
        H, info = estimate(pts1, pts2, CFG)
        self.assertIsNone(H, "random noise pair should not produce a homography")
        self.assertIsInstance(info["reason"], str)
        self.assertTrue(info["reason"])

        conf, breakdown = score(info["inlier_ratio"], info["n_inliers"],
                                info["reproj_error"], 10.0, cfg=CFG, rejected=True)
        self.assertEqual(conf, 0.0)
        self.assertTrue(breakdown["rejected"])

    def test_implausible_homography_rejected(self):
        # 5x zoom is outside the configured scale band; supply well over
        # min_inliers points so the scale check is what rejects it
        H_bad = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 1.0]])
        rng = np.random.default_rng(1)
        pts2 = rng.uniform(5, 200, size=(4 * CFG["alignment"]["min_inliers"], 2)).astype(np.float32)
        pts1 = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), H_bad).reshape(-1, 2)
        H, info = estimate(pts1, pts2, CFG)
        self.assertIsNone(H)
        self.assertIn("scale", info["reason"])


class TestCornerError(unittest.TestCase):
    def test_scale_invariance(self):
        H_a = np.array([[1.02, -0.03, 5.0], [0.02, 0.99, -3.0], [1e-5, -2e-5, 1.0]])
        H_b = np.array([[0.98, 0.01, -2.0], [-0.02, 1.01, 4.0], [2e-5, 1e-5, 1.0]])
        base = corner_error(H_a, H_b, 224, 224)
        for factor in (0.25, 3.0, -7.0):
            scaled = corner_error(H_a * factor, H_b, 224, 224)
            self.assertAlmostEqual(base, scaled, places=6,
                                   msg=f"corner error changed when H scaled by {factor}")

    def test_identical_homographies_have_zero_error(self):
        H = np.array([[1.05, 0.02, 3.0], [-0.01, 0.98, -2.0], [1e-5, 1e-5, 1.0]])
        self.assertLess(corner_error(H, H, 224, 224), 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
