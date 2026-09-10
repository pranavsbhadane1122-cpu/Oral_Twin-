"""Compare the contaminated v1 classifier with the de-biased v2 classifier.

Both models are evaluated on the SAME (clean, de-biased) test split and with
the SAME random re-framings, so the comparison is apples-to-apples. v1's
original 0.8996 was measured on the old contaminated split and is reported
separately for context - it is not comparable to anything here.

Usage: python scripts/compare_models.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode  # noqa: E402

MODELS = {
    "v1_contaminated": "classifier_3class_v1_contaminated.h5",
    "v2_debiased": "classifier_3class_v2_debiased.h5",
}


def random_homography(rng, w, h):
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += rng.uniform(-0.03, 0.03) * w
    M[1, 2] += rng.uniform(-0.03, 0.03) * h
    return np.vstack([M, [0, 0, 1]])


def per_class_metrics(cm):
    out = []
    total = cm.sum()
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(dict(precision=precision, recall=recall, specificity=specificity,
                        f1=f1, support=int(cm[i, :].sum())))
    return out


def main():
    cfg = load_config()
    classes = cfg["classification"]["classes"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    test_root = PROJECT_ROOT / cfg["paths"]["processed"] / "classification" / "test"

    paths, labels = [], []
    for label, cls in enumerate(classes):
        for p in sorted((test_root / cls).glob("*.jpg")):
            paths.append(p)
            labels.append(label)
    y_true = np.array(labels)
    print(f"clean test split: {len(paths)} images "
          f"({', '.join(f'{c}={labels.count(i)}' for i, c in enumerate(classes))})")

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    rng = np.random.default_rng(cfg["split"]["seed"])
    originals, warped = [], []
    for p in paths:
        # the training loader decodes JPEG to RGB, so convert from OpenCV's BGR
        img = cv2.cvtColor(imread_unicode(p), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        H = random_homography(rng, w, h)
        originals.append(preprocess(img.astype(np.float32)))
        warped.append(preprocess(
            cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.float32)))
    x_orig, x_warp = np.stack(originals), np.stack(warped)

    results = {}
    for name, filename in MODELS.items():
        path = models_dir / filename
        if not path.exists():
            print(f"  {name}: {filename} missing, skipped")
            continue
        model = tf.keras.models.load_model(path, compile=False)
        pred = model.predict(x_orig, verbose=0).argmax(axis=1)
        pred_w = model.predict(x_warp, verbose=0).argmax(axis=1)

        cm = np.zeros((len(classes), len(classes)), int)
        for t, p_ in zip(y_true, pred):
            cm[t, p_] += 1
        results[name] = {
            "accuracy_clean_test": float((pred == y_true).mean()),
            "warp_stability": float((pred == pred_w).mean()),
            "n_flipped_by_warp": int((pred != pred_w).sum()),
            "confusion_matrix": cm.tolist(),
            "per_class": {c: {k: round(v, 4) for k, v in m.items()}
                          for c, m in zip(classes, per_class_metrics(cm))},
        }
        print(f"\n{name}:")
        print(f"  accuracy on clean test split: {results[name]['accuracy_clean_test']:.4f}")
        print(f"  warp stability:               {results[name]['warp_stability']:.4f} "
              f"({results[name]['n_flipped_by_warp']} of {len(paths)} flipped)")
        for c, m in results[name]["per_class"].items():
            print(f"    {c:<12} P {m['precision']:.3f}  R {m['recall']:.3f}  "
                  f"F1 {m['f1']:.3f}  n={m['support']}")

    out = models_dir / "logs" / "model_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
