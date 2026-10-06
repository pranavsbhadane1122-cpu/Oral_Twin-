"""Stratified, blind Grad-CAM panel for the acceptance gate.

Two things make this harder to fool than the Phase 7 panel:

  stratified  equal numbers per TRUE class, and within each class the original
              categories are spread as evenly as the split allows, so the sample
              cannot quietly become all-Normal.
  blind       the rendered image carries only a case number. The true class, the
              prediction and the probability go into key.json, which is meant to
              stay unread until the verdicts are written down.

Verdicts are then scored by TRUE class, not by what the model predicted.

Writes data/processed/gradcam_gate/case_NN.jpg and key.json.

Usage:
  python scripts/gradcam_panel.py            # render the blind panel
  python scripts/gradcam_panel.py --reveal   # print the key afterwards
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.classification.manifest_dataset import read_split, task_config  # noqa: E402
from src.explainability.gradcam import build_grad_model, compute, overlay  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode, imwrite_jpg  # noqa: E402

N_CASES = 10


def choose_cases(cfg, task, n=N_CASES):
    """Equal numbers per true class, categories spread within each class."""
    paths, labels, classes, categories = read_split("test", cfg, task)
    seed = cfg["split"]["seed"]
    rng = np.random.default_rng(seed)

    by_class = defaultdict(lambda: defaultdict(list))
    for index, (label, category) in enumerate(zip(labels, categories)):
        by_class[label][category].append(index)

    per_class = n // len(classes)
    chosen = []
    for label in sorted(by_class):
        buckets = by_class[label]
        names = sorted(buckets)
        # round-robin across categories so one does not dominate
        picked, position = [], 0
        pools = {name: list(rng.permutation(buckets[name])) for name in names}
        while len(picked) < per_class and any(pools.values()):
            name = names[position % len(names)]
            if pools[name]:
                picked.append(int(pools[name].pop()))
            position += 1
        chosen.extend((label, i) for i in picked)
    return chosen, paths, classes, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reveal", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    task, classes, _ = task_config(cfg)
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "gradcam_gate"
    key_path = out_dir / "key.json"

    if args.reveal:
        if not key_path.exists():
            raise SystemExit("no key.json; render the panel first")
        key = json.loads(key_path.read_text(encoding="utf-8"))
        print(f"{'case':<8} {'true class':<10} {'category':<9} {'predicted':<10} "
              f"{'p':>6}  file")
        for row in key["cases"]:
            print(f"{row['case']:<8} {row['true_class']:<10} {row['category']:<9} "
                  f"{row['predicted']:<10} {row['probability']:>6.3f}  "
                  f"{Path(row['path']).name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    model_path = PROJECT_ROOT / cfg["paths"]["models"] / f"classifier_{task}.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    grad_model = build_grad_model(model, cfg=cfg)

    chosen, paths, classes, categories = choose_cases(cfg, task)
    rng = np.random.default_rng(cfg["split"]["seed"] + 1)
    order = rng.permutation(len(chosen))   # so class order is not positional

    key = {"task": task, "model": str(model_path.relative_to(PROJECT_ROOT)),
           "classes": classes, "cases": []}
    for case_number, position in enumerate(order, 1):
        label, index = chosen[position]
        path = paths[index]
        image = imread_unicode(path)
        result = compute(model, image, cfg=cfg, grad_model=grad_model)
        blended = overlay(image, result["heatmap"], cfg)

        panel = np.hstack([image, blended])
        banner = np.zeros((26, panel.shape[1], 3), np.uint8)
        # the case number is the ONLY thing written on the image
        cv2.putText(banner, f"case {case_number:02d}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        imwrite_jpg(out_dir / f"case_{case_number:02d}.jpg",
                    np.vstack([banner, panel]), quality=92)

        key["cases"].append({
            "case": f"case_{case_number:02d}",
            "path": path,
            "true_class": classes[label],
            "category": categories[index],
            "predicted": classes[int(result["class_index"])],
            "probability": float(result["probability"]),
            "cam_flat": bool(result["flat"]),
        })

    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"rendered {len(key['cases'])} blind cases to {out_dir}")
    print("each image shows: original | Grad-CAM overlay, labelled only by case number")
    print(f"key withheld in {key_path.name}; score the cases before reading it")
    composition = defaultdict(int)
    for row in key["cases"]:
        composition[(row["true_class"], row["category"])] += 1
    print(f"composition (hidden in the images): {dict(composition)}")


if __name__ == "__main__":
    main()
