"""Evaluate the segmenter on the held-out test split, and draw overlays to look at.

Two things are reported separately and must not be averaged together:

  lesion-bearing images   the only images on which lesion Dice means anything.
                          The gate is written against this number.
  images with no lesion   every Healthy image. Dice is undefined when the truth
                          is empty, so instead we report how much area the model
                          hallucinates there. An all-zero prediction scoring a
                          perfect 1.0 by convention would flatter the model into
                          meaninglessness.

Confidence intervals are bootstrap percentile intervals over images, not Wilson:
Dice is a continuous per-image quantity, not a success count.

The overlays are the point of this script. A Dice figure can look respectable
while the predicted mask sits on teeth and lips, so the overlays get looked at
and a verdict written down, exactly as the Grad-CAM panels were.

Outputs (both git-ignored - these are CC BY-NC-ND derivatives):
  data/processed/segmentation_overlays/*.jpg
  models/logs/segmentation_eval.txt

Usage: python -m src.segmentation.evaluate
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.segmentation.dataset import cached_pair, make_dataset, read_manifest
from src.utils.config import PROJECT_ROOT, load_config

OVERLAY_COUNT = 10
THRESHOLD = 0.5
TRUTH_COLOUR = (0, 255, 0)        # BGR: green  = annotated ground truth
PREDICTED_COLOUR = (0, 0, 255)    # BGR: red    = model prediction
RNG_SEED = 20251006


def dice(truth, predicted):
    total = truth.sum() + predicted.sum()
    if total == 0:
        return None                       # undefined, not perfect
    return 2.0 * np.logical_and(truth, predicted).sum() / total


def iou(truth, predicted):
    union = np.logical_or(truth, predicted).sum()
    if union == 0:
        return None
    return np.logical_and(truth, predicted).sum() / union


def bootstrap_interval(values, resamples=2000, seed=RNG_SEED):
    values = np.asarray([v for v in values if v is not None], float)
    if len(values) < 2:
        return (float("nan"), float("nan"), float("nan"), len(values))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high), len(values)


def contour_overlay(image, truth, predicted):
    canvas = image.copy()
    for mask, colour in ((truth, TRUTH_COLOUR), (predicted, PREDICTED_COLOUR)):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 2)
    return canvas


def main():
    cfg = load_config()
    scfg = cfg["segmentation"]
    size = scfg["img_size"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    checkpoint = models_dir / "segmenter.h5"
    out_text = models_dir / "logs" / "segmentation_eval.txt"
    overlay_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "segmentation_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("=" * 74)
    log("PHASE 3 - SEGMENTATION EVALUATION (held-out test split)")
    log("=" * 74)
    log("")

    model = tf.keras.models.load_model(checkpoint, compile=False)
    log(f"model: {checkpoint.name}   threshold {THRESHOLD}")

    rows = read_manifest("test", cfg)
    records = []
    for row in rows:
        stem = Path(row["path"]).stem
        image_path, mask_path = cached_pair(stem, cfg)
        if image_path.exists() and mask_path.exists():
            records.append((stem, row["category"], image_path, mask_path))
    log(f"test images: {len(records)}")
    log("")

    scores = {"cavity": {"dice": [], "iou": []},
              "lesion": {"dice": [], "iou": []}}
    empty_truth_area = []        # predicted lesion fraction where truth has none
    per_image = []

    batch_size = scfg["batch_size"]
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        images, truths = [], []
        for _, _, image_path, mask_path in chunk:
            bgr = cv2.imread(str(image_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(rgb.astype(np.float32) / 127.5 - 1.0)
            mask = cv2.imread(str(mask_path))      # BGR: R last, G middle
            truths.append(np.stack([mask[..., 2] > 127, mask[..., 1] > 127], -1))
        predicted = model.predict(np.stack(images), verbose=0) > THRESHOLD

        for (stem, category, _, _), truth, prediction in zip(chunk, truths, predicted):
            entry = {"stem": stem, "category": category}
            for index, name in ((0, "cavity"), (1, "lesion")):
                t, p = truth[..., index], prediction[..., index]
                d, j = dice(t, p), iou(t, p)
                entry[f"{name}_dice"] = d
                entry[f"{name}_truth_px"] = int(t.sum())
                if d is not None:
                    scores[name]["dice"].append(d)
                    scores[name]["iou"].append(j)
            if not truth[..., 1].any():
                empty_truth_area.append(prediction[..., 1].mean())
            per_image.append(entry)
        if (start // batch_size) % 5 == 0:
            print(f"  {min(start + batch_size, len(records))}/{len(records)}",
                  flush=True)

    log("-" * 74)
    log("DICE and IoU, mean with bootstrap 95% CI over images")
    log("-" * 74)
    summary = {}
    for name in ("cavity", "lesion"):
        d_mean, d_low, d_high, n = bootstrap_interval(scores[name]["dice"])
        j_mean, j_low, j_high, _ = bootstrap_interval(scores[name]["iou"])
        summary[name] = d_mean
        scope = ("all images with an annotated cavity" if name == "cavity"
                 else "lesion-bearing images only")
        log(f"  {name:<7} n={n:<4} ({scope})")
        log(f"      Dice {d_mean:.3f}  [{d_low:.3f}, {d_high:.3f}]")
        log(f"      IoU  {j_mean:.3f}  [{j_low:.3f}, {j_high:.3f}]")
    log("")
    if empty_truth_area:
        area = np.asarray(empty_truth_area)
        clean = int((area < 0.001).sum())
        log(f"  images with NO annotated lesion (Healthy): {len(area)}")
        log(f"      mean predicted lesion area: {area.mean() * 100:.2f}% of frame")
        log(f"      predicted essentially nothing (<0.1% of frame): "
            f"{clean}/{len(area)} ({clean / len(area) * 100:.1f}%)")
        log("      Dice is undefined here and is excluded from the figures above.")
    log("")

    # ---------------------------------------------------------------- overlays
    rng = np.random.default_rng(RNG_SEED)
    candidates = [e for e in per_image if e["lesion_truth_px"] > 0]
    chosen = [candidates[i] for i in
              rng.choice(len(candidates), min(OVERLAY_COUNT, len(candidates)),
                         replace=False)]
    chosen.sort(key=lambda e: e["stem"])

    log("-" * 74)
    log(f"OVERLAYS - {len(chosen)} lesion-bearing test images, chosen at random "
        f"(seed {RNG_SEED})")
    log("  green = annotated ground truth    red = model prediction")
    log("-" * 74)
    for entry in chosen:
        image_path, mask_path = cached_pair(entry["stem"], cfg)
        bgr = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        prediction = model.predict(rgb[None], verbose=0)[0] > THRESHOLD

        panel = contour_overlay(bgr, mask[..., 1] > 127, prediction[..., 1])
        cv2.putText(panel, entry["stem"], (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, entry["stem"], (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        out = overlay_dir / f"{entry['stem']}_lesion.jpg"
        cv2.imwrite(str(out), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])

        d = entry["lesion_dice"]
        log(f"  {entry['stem']:<14} {entry['category']:<10} "
            f"lesion Dice {d:.3f}   {out.name}")
    log("")
    log(f"overlays written to {overlay_dir}")
    log("(git-ignored: these are derivatives of a CC BY-NC-ND dataset)")
    log("")

    # ------------------------------------------------------------------ gate
    log("=" * 74)
    log("GATE")
    log("=" * 74)
    lesion_dice = summary["lesion"]
    numeric = "PASS" if lesion_dice >= 0.70 else "FAIL"
    log(f"  required: lesion Dice >= 0.70 on lesion-bearing test images")
    log(f"  measured: {lesion_dice:.3f}   -> {numeric}")
    log(f"  required: >= 8 of {len(chosen)} predicted masks land on annotated tissue")
    log("  measured: see the written verdict below - judged by looking at the")
    log("            overlays, not inferred from the Dice figure.")
    log("")
    log("OralTwin is a screening aid, not a diagnostic tool.")

    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_text.write_text("\n".join(lines), encoding="utf-8")

    csv_path = models_dir / "logs" / "segmentation_per_image.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    print(f"\nsaved {out_text}\nsaved {csv_path}")


if __name__ == "__main__":
    main()
