"""Figures for the classifier-findings paper section.

fig_gradcam_v1_v2.jpg
    One row per case: input photo | v1 (contaminated) Grad-CAM | v2 (de-biased)
    Grad-CAM. Both models are explained on the SAME image for the class each
    predicts, so the only difference between columns is the training data.

The cases are the visit1 photos of the ten pairs used in the Phase 2b
inspection (data/processed/explain/index.json).

Usage: python scripts/make_paper_figures.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import build_grad_model, compute, overlay  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode, imwrite_jpg  # noqa: E402

MODELS = {
    "v1 (contaminated data)": "classifier_3class_v1_contaminated.h5",
    "v2 (de-biased data)": "classifier_3class_v2_debiased.h5",
}
CELL = 224


def caption(img, text, height=22):
    bar = np.zeros((height, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text[:40], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    cfg = load_config()
    out_dir = PROJECT_ROOT / "docs" / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads(
        (PROJECT_ROOT / cfg["paths"]["processed"] / "explain" / "index.json").read_text("utf-8"))
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    cases = []
    for entry in index:
        gt = json.loads((long_root / entry["pair"] / "ground_truth.json").read_text("utf-8"))
        cases.append((entry["pair"], gt["source_class"], long_root / entry["pair"] / "visit1.jpg"))

    loaded = {}
    for label, filename in MODELS.items():
        model = tf.keras.models.load_model(PROJECT_ROOT / cfg["paths"]["models"] / filename,
                                           compile=False)
        loaded[label] = (model, build_grad_model(model, cfg=cfg))

    rows, record = [], []
    for pair, true_cls, path in cases:
        img = cv2.resize(imread_unicode(path), (CELL, CELL), interpolation=cv2.INTER_AREA)
        cells = [caption(img, f"{pair}  true: {true_cls}")]
        entry = {"pair": pair, "true_class": true_cls}
        for label, (model, grad_model) in loaded.items():
            result = compute(model, img, cfg=cfg, grad_model=grad_model)
            tag = "v1" if label.startswith("v1") else "v2"
            cells.append(caption(overlay(img, result["heatmap"], cfg),
                                 f"{tag}: {result['class_name']} p={result['probability']:.2f}"))
            entry[tag] = {"predicted": result["class_name"],
                          "probability": round(result["probability"], 4),
                          "flat": result["flat"]}
        rows.append(np.hstack(cells))
        record.append(entry)

    header = np.zeros((30, rows[0].shape[1], 3), np.uint8)
    for i, text in enumerate(["input photo"] + list(MODELS)):
        cv2.putText(header, text, (6 + i * CELL, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    figure = np.vstack([header] + rows)
    imwrite_jpg(out_dir / "fig_gradcam_v1_v2.jpg", figure, quality=92)
    (out_dir / "fig_gradcam_v1_v2.json").write_text(json.dumps(record, indent=2),
                                                    encoding="utf-8")
    print(f"wrote {out_dir / 'fig_gradcam_v1_v2.jpg'} ({len(rows)} cases)")
    for entry in record:
        print(f"  {entry['pair']} true={entry['true_class']:<11} "
              f"v1={entry['v1']['predicted']:<11} {entry['v1']['probability']:.2f}  "
              f"v2={entry['v2']['predicted']:<11} {entry['v2']['probability']:.2f}")


if __name__ == "__main__":
    main()
