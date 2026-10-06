"""How much of SMART-OM's Oral Cancer class is separable from acquisition alone?

This classifier never sees a pixel. Its only features are the file extension
(PNG or not) and whether the file carries EXIF camera metadata. If it can pick
out the Oral Cancer class from those two bits, then any model scored on that
subset is partly being graded on how the picture was captured rather than on the
tissue in it.

This is evidence for the paper, not a model we keep.

Writes data/processed/confound_baseline.txt. READ-ONLY on data/raw.

Usage: python scripts/confound_baseline.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.smartom import CONFOUNDED_CATEGORY, category_of, list_images  # noqa: E402


def acquisition_features(path):
    """(is_png, has_exif) - nothing about image content."""
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
            try:
                exif = im.getexif()
                has_exif = bool(exif and len(exif))
            except Exception:
                has_exif = False
    except Exception:
        return None
    return [1.0 if fmt == "PNG" else 0.0, 1.0 if has_exif else 0.0]


def main():
    cfg = load_config()
    out = PROJECT_ROOT / cfg["paths"]["processed"] / "confound_baseline.txt"
    seed = cfg["split"]["seed"]
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("CONFOUND BASELINE - SMART-OM Oral Cancer from acquisition metadata alone")
    log("The classifier sees no image content: only file extension and EXIF presence.")
    log("")

    images = list_images(cfg)
    rows, labels, kept = [], [], []
    for path in images:
        features = acquisition_features(path)
        if features is None:
            continue
        rows.append(features)
        labels.append(1 if category_of(path, cfg) == CONFOUNDED_CATEGORY else 0)
        kept.append(path)

    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels)
    positives = int(y.sum())
    log(f"images: {len(y)}   Oral Cancer: {positives}   other: {len(y) - positives}")
    log(f"features: is_png, has_exif")
    log("")
    log("raw contingency:")
    for name, column in (("is_png", 0), ("has_exif", 1)):
        for value in (0, 1):
            mask = x[:, column] == value
            log(f"  {name}={int(value)}: {int(mask.sum()):>5} images, "
                f"{int(y[mask].sum()):>3} of them Oral Cancer")
    log("")

    # 20 positives is too few for a single hold-out, so pool stratified folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predicted = np.zeros_like(y)
    for train_index, test_index in folds.split(x, y):
        model = DecisionTreeClassifier(max_depth=3, random_state=seed)
        model.fit(x[train_index], y[train_index])
        predicted[test_index] = model.predict(x[test_index])

    true_positive = int(((predicted == 1) & (y == 1)).sum())
    false_positive = int(((predicted == 1) & (y == 0)).sum())
    false_negative = int(((predicted == 0) & (y == 1)).sum())
    true_negative = int(((predicted == 0) & (y == 0)).sum())
    accuracy = (true_positive + true_negative) / len(y)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    log("5-fold stratified cross-validation, metrics pooled over the held-out folds:")
    log(f"  accuracy  {accuracy:.4f}")
    log(f"  precision {precision:.4f}   (of images called Oral Cancer, how many are)")
    log(f"  recall    {recall:.4f}   (of Oral Cancer images, how many are found)")
    log(f"  F1        {f1:.4f}")
    log(f"  confusion: TP {true_positive}  FP {false_positive}  "
        f"FN {false_negative}  TN {true_negative}")
    log("")
    majority = 1.0 - positives / len(y)
    log(f"  for scale: always predicting 'not cancer' scores accuracy "
        f"{majority:.4f} with recall 0.000")
    log("")

    final = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(x, y)
    log("the rule it learns, in full:")
    for line in export_text(final, feature_names=["is_png", "has_exif"]).splitlines():
        log(f"  {line}")

    log("")
    if recall >= 0.5 and precision >= 0.5:
        verdict = (
            f"INTERPRETATION: with no access to the image at all, file format and EXIF "
            f"presence alone recover {recall:.0%} of SMART-OM's Oral Cancer class at "
            f"{precision:.0%} precision. Any sensitivity figure computed on this subset "
            f"is therefore partly a measurement of how the photographs were acquired, "
            f"not of disease. The subset is excluded from every performance figure; "
            f"Piyarathne's OCA images are the cancer estimate."
        )
    else:
        verdict = (
            f"INTERPRETATION: acquisition metadata alone recovers {recall:.0%} of the "
            f"Oral Cancer class at {precision:.0%} precision - a weaker shortcut than "
            f"the file-level audit suggested, but still present and still a reason to "
            f"keep the subset out of headline figures."
        )
    log(verdict)
    log("")
    log("This baseline is evidence for the paper, not a model that is kept.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
