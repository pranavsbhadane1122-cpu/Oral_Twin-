"""Does the prediction survive a small re-framing? (acceptance gate c)

A model reading tissue should not change its mind when the same photograph is
taken from a slightly different angle. The Phase 2 classifier failed this badly
- one image flipped from p=0.99 to p=0.00 - which was part of the evidence that
it was reading provenance rather than pathology.

Each test image is warped by a small random homography (rotation +/-8 degrees,
scale 0.92-1.08, shift +/-3%) and the predicted class is compared before and
after. Reported with a 95% interval.

Writes models/logs/warp_stability_<task>.txt.

Usage: python scripts/warp_stability.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.classification.manifest_dataset import read_split, task_config  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode  # noqa: E402
from src.utils.stats import format_rate  # noqa: E402

TARGET = 0.90


def random_homography(rng, width, height):
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.92, 1.08)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += rng.uniform(-0.03, 0.03) * width
    matrix[1, 2] += rng.uniform(-0.03, 0.03) * height
    return np.vstack([matrix, [0, 0, 1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config()
    task, classes, _ = task_config(cfg, args.task)
    model_path = PROJECT_ROOT / cfg["paths"]["models"] / f"classifier_{task}.h5"
    model = tf.keras.models.load_model(model_path, compile=False)

    paths, labels, _, categories = read_split(args.split, cfg, args.task)
    rng = np.random.default_rng(cfg["split"]["seed"])
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    originals, warped = [], []
    for path in paths:
        # the training loader decodes JPEG to RGB; OpenCV reads BGR
        image = cv2.cvtColor(imread_unicode(path), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        matrix = random_homography(rng, width, height)
        originals.append(preprocess(image.astype(np.float32)))
        warped.append(preprocess(cv2.warpPerspective(
            image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT
        ).astype(np.float32)))

    before = model.predict(np.stack(originals), verbose=0).argmax(axis=1)
    after = model.predict(np.stack(warped), verbose=0).argmax(axis=1)
    stable = int((before == after).sum())
    total = len(before)

    lines = ["=" * 70, f"WARP STABILITY - {task} ({args.split} split)", "=" * 70, "",
             "Each image is re-framed by a small random homography (rotation +/-8",
             "degrees, scale 0.92-1.08, shift +/-3%) and the predicted class is",
             "compared before and after.", "",
             f"  stability  {format_rate(stable, total)}",
             f"  target     >= {TARGET:.2f}",
             f"  verdict    {'PASS' if stable / total >= TARGET else 'FAIL'}",
             f"  flipped    {total - stable} of {total}", ""]

    categories = np.asarray(categories)
    lines.append("  by original category:")
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        agree = int((before[mask] == after[mask]).sum())
        lines.append(f"    {category:<10} {format_rate(agree, int(mask.sum()))}")
    lines.append("")

    report = "\n".join(lines)
    out = PROJECT_ROOT / cfg["paths"]["models"] / "logs" / f"warp_stability_{task}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
