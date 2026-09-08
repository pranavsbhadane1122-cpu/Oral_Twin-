"""Phase 2 sanity checks (run with: python -m unittest tests.test_training -v).

  - model builds with the number of classes from config
  - loaders yield correctly shaped, preprocessed batches with valid labels
  - class weights cover every class and their weighted sum equals train size
  - the checkpoint written by the debug run loads back and predicts
"""

import unittest

import numpy as np
import tensorflow as tf

from src.classification.dataset import class_counts, compute_class_weights, make_dataset
from src.classification.model import build_model, compile_model, unfreeze_top
from src.utils.config import PROJECT_ROOT, load_config

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
