"""Phase 2 sanity checks (run with: python -m unittest tests.test_training -v).

  - model builds with the number of classes from config
  - loaders yield correctly shaped, preprocessed batches with valid labels
  - class weights cover every class and their weighted sum equals train size
  - the checkpoint written by the debug run loads back and predicts
"""

import unittest

import cv2
import numpy as np
import tensorflow as tf

from src.classification.dataset import class_counts, compute_class_weights, make_dataset
from src.classification.model import build_model, compile_model, unfreeze_top
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode

CFG = load_config()
CLASSES = CFG["classification"]["classes"]
IMG = CFG["classification"]["img_size"]


class TestModel(unittest.TestCase):
    def test_build_and_finetune(self):
        model = build_model(len(CLASSES), IMG, CFG["classification"]["dropout"])
        self.assertEqual(model.output_shape, (None, len(CLASSES)))
        head_trainable = model.count_params() - model.get_layer("mobilenetv2_1.00_224").count_params()
        self.assertGreater(head_trainable, 0)
        n_before = sum(np.prod(w.shape) for w in model.trainable_weights)
        unfreeze_top(model, CFG["classification"]["fine_tune_layers"])
        n_after = sum(np.prod(w.shape) for w in model.trainable_weights)
        self.assertGreater(n_after, n_before, "fine-tuning should unfreeze extra weights")
        compile_model(model, CFG["classification"]["learning_rate"])
        probs = model.predict(np.zeros((1, IMG, IMG, 3), dtype=np.float32), verbose=0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


class TestLoaders(unittest.TestCase):
    def test_batch_shapes_and_labels(self):
        for split, augment in (("train", True), ("val", False)):
            ds = make_dataset(CFG, split, augment=augment)
            x, y = next(iter(ds))
            self.assertEqual(tuple(x.shape[1:]), (IMG, IMG, 3), split)
            self.assertLessEqual(int(x.shape[0]), CFG["classification"]["batch_size"])
            self.assertTrue(bool(tf.reduce_all(y >= 0)) and bool(tf.reduce_all(y < len(CLASSES))))
            self.assertGreaterEqual(float(tf.reduce_min(x)), -1.001, split)
            self.assertLessEqual(float(tf.reduce_max(x)), 1.001, split)

    def test_class_weights(self):
        weights = compute_class_weights(CFG)
        counts = class_counts(CFG, "train")
        self.assertEqual(set(weights), set(range(len(CLASSES))))
        self.assertTrue(all(w > 0 for w in weights.values()))
        weighted_total = sum(weights[i] * counts[cls] for i, cls in enumerate(CLASSES))
        self.assertAlmostEqual(weighted_total, sum(counts.values()), delta=1e-3)


class TestWarpStability(unittest.TestCase):
    """A model reading pathology should not change its mind when the same photo
    is re-framed slightly. v1 failed this badly: pair_0036 flipped from
    p(caries)=0.99 to 0.00 under a small warp of the same image.
    """

    MIN_STABLE_FRACTION = 0.90

    @staticmethod
    def random_homography(rng, w, h):
        """Small, realistic re-framing: +/-8 deg, 0.92-1.08 scale, slight shift."""
        angle = rng.uniform(-8.0, 8.0)
        scale = rng.uniform(0.92, 1.08)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += rng.uniform(-0.03, 0.03) * w
        M[1, 2] += rng.uniform(-0.03, 0.03) * h
        return np.vstack([M, [0, 0, 1]])

    def test_predictions_survive_small_warps(self):
        path = PROJECT_ROOT / CFG["paths"]["models"] / f"classifier_{len(CLASSES)}class.h5"
        if not path.exists():
            self.skipTest("no trained model available")
        model = tf.keras.models.load_model(path, compile=False)

        images = []
        for cls in CLASSES:
            images += sorted(
                (PROJECT_ROOT / CFG["paths"]["processed"] / "classification" / "test" / cls)
                .glob("*.jpg")
            )
        if not images:
            self.skipTest("no test images available")

        rng = np.random.default_rng(CFG["split"]["seed"])
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        originals, warped = [], []
        for p in images:
            # the training loader decodes JPEG to RGB; OpenCV reads BGR
            img = cv2.cvtColor(imread_unicode(p), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            H = self.random_homography(rng, w, h)
            originals.append(preprocess(img.astype(np.float32)))
            warped.append(preprocess(
                cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)
                .astype(np.float32)))

        pred_original = model.predict(np.stack(originals), verbose=0).argmax(axis=1)
        pred_warped = model.predict(np.stack(warped), verbose=0).argmax(axis=1)
        stable = float((pred_original == pred_warped).mean())

        self.assertGreaterEqual(
            stable, self.MIN_STABLE_FRACTION,
            f"only {100*stable:.1f}% of predictions survived a small re-framing "
            f"({int((pred_original != pred_warped).sum())} of {len(images)} flipped); "
            f"a model reading pathology should be far more stable",
        )


class TestCheckpoint(unittest.TestCase):
    def test_debug_checkpoint_loads(self):
        path = PROJECT_ROOT / CFG["paths"]["models"] / f"classifier_{len(CLASSES)}class.h5"
        self.assertTrue(path.exists(), "run `python -m src.classification.train --debug` first")
        model = tf.keras.models.load_model(path)
        probs = model.predict(np.zeros((1, IMG, IMG, 3), dtype=np.float32), verbose=0)
        self.assertEqual(probs.shape, (1, len(CLASSES)))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
