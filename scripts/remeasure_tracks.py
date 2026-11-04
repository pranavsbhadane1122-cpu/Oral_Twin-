"""Re-measure every model about to be surfaced as Track A, B or C.

Nothing is surfaced until its CURRENT performance is known, and every figure
here is destined to travel with a score in front of a reader. So each is
computed fresh from the saved checkpoint and the stored splits, not copied
from a log.

  Track A  gum condition        models/classifier_3class.h5        (MIO/Kaggle)
  Track B  lesion segmentation  models/segmenter.h5, lesion channel (Piyarathne)
  Track C  OPMD / referral      models/classifier_binary_referral.h5 (Piyarathne)

CLASS BALANCE IS PRINTED BESIDE EVERY FIGURE. The external accuracy of 0.915
in this project is an artefact of a 94.5% negative base rate, and a rate
quoted without its denominator is how that happens.

Writes models/logs/track_remeasurement.txt and appends a dated run to
models/logs/experiments.md.

Usage: python scripts/remeasure_tracks.py
"""

import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode  # noqa: E402
from src.utils.stats import wilson_interval  # noqa: E402

LINES = []


def log(text=""):
    print(text, flush=True)
    LINES.append(text)


def rate(numerator, denominator):
    if denominator == 0:
        return "n/a (0 cases)"
    value, low, high = wilson_interval(numerator, denominator)
    return f"{value:.3f} [{low:.3f}, {high:.3f}]  ({numerator}/{denominator})"


def binary_metrics(truth, predicted, positive_label, name, balance_note):
    """Sensitivity, specificity, PPV, NPV, each with its denominator."""
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    tp = int(((truth == positive_label) & (predicted == positive_label)).sum())
    fn = int(((truth == positive_label) & (predicted != positive_label)).sum())
    tn = int(((truth != positive_label) & (predicted != positive_label)).sum())
    fp = int(((truth != positive_label) & (predicted == positive_label)).sum())

    positives = tp + fn
    total = len(truth)
    log(f"  {name}")
    log(f"    class balance : {positives}/{total} positive "
        f"({100.0 * positives / max(total, 1):.1f}%)  -  {balance_note}")
    log(f"    sensitivity   : {rate(tp, tp + fn)}")
    log(f"    specificity   : {rate(tn, tn + fp)}")
    log(f"    PPV           : {rate(tp, tp + fp)}")
    log(f"    NPV           : {rate(tn, tn + fn)}")
    log(f"    accuracy      : {rate(tp + tn, total)}")
    log(f"    confusion     : TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    return {"sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "npv": tn / (tn + fn) if tn + fn else None,
            "n": total, "positives": positives,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def load_images(paths, size):
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    batch = []
    for path in paths:
        image = imread_unicode(path)
        if image is None:
            batch.append(None)
            continue
        image = cv2.cvtColor(cv2.resize(image, (size, size),
                                        interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        batch.append(preprocess(image.astype(np.float32)))
    return batch


# --------------------------------------------------------------------------
# Track A - gum condition, three classes
# --------------------------------------------------------------------------

def track_a(cfg):
    log("=" * 74)
    log("TRACK A - gum condition (models/classifier_3class.h5)")
    log("=" * 74)
    size = cfg["classification"]["img_size"]
    classes = cfg["classification"]["classes"]
    root = PROJECT_ROOT / cfg["paths"]["processed"] / "classification" / "test"

    paths, truth = [], []
    for index, name in enumerate(classes):
        for path in sorted((root / name).glob("*.jpg")):
            paths.append(path)
            truth.append(index)
    log(f"internal test set: {len(paths)} images from {root.relative_to(PROJECT_ROOT)}")
    if not paths:
        log("  NO TEST IMAGES - cannot measure")
        return None

    model = tf.keras.models.load_model(
        PROJECT_ROOT / cfg["paths"]["models"] / "classifier_3class.h5", compile=False)
    batch = [b for b in load_images(paths, size) if b is not None]
    predicted = model.predict(np.stack(batch), verbose=0).argmax(axis=1)
    truth = np.asarray(truth[:len(predicted)])

    correct = int((predicted == truth).sum())
    log("")
    log(f"  overall accuracy: {rate(correct, len(truth))}")
    log("")
    log("  one-vs-rest per class (a 3-class model has no single sensitivity):")
    results = {}
    for index, name in enumerate(classes):
        n_pos = int((truth == index).sum())
        results[name] = binary_metrics(
            truth == index, predicted == index, True, f"[{name}]",
            f"{n_pos} of {len(truth)} images are {name}")
        log("")

    log("  EXTERNAL VALIDATION: none exists for this model.")
    log("  No second dataset shares these three labels. The only external set")
    log("  in the project, SMART-OM, carries Normal / OPMD / Oral Cancer, which")
    log("  do not map onto caries / calculus / gingivitis. Track A therefore has")
    log("  NO out-of-domain figure at all, and cannot be given one today.")
    log("")
    return results


# --------------------------------------------------------------------------
# Track C - OPMD / referral
# --------------------------------------------------------------------------

def track_c(cfg):
    log("=" * 74)
    log("TRACK C - OPMD / referral (models/classifier_binary_referral.h5)")
    log("=" * 74)
    size = cfg["classification"]["img_size"]
    task = cfg["classification"]["tasks"]["binary_referral"]
    classes = task["classes"]
    mapping = task["mapping"]
    positive = classes.index("refer")

    model = tf.keras.models.load_model(
        PROJECT_ROOT / cfg["paths"]["models"] / "classifier_binary_referral.h5",
        compile=False)
    raw = PROJECT_ROOT / cfg["paths"]["raw"]

    # ---- internal, Piyarathne test split
    manifest = (PROJECT_ROOT / cfg["paths"]["processed"] / "splits" / "piyarathne"
                / "test.csv")
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    paths = [raw / r["path"] for r in rows]
    truth = [classes.index(mapping[r["category"]]) for r in rows]
    batch, keep = [], []
    for i, b in enumerate(load_images(paths, size)):
        if b is not None:
            batch.append(b)
            keep.append(truth[i])
    predicted = model.predict(np.stack(batch), verbose=0).argmax(axis=1)
    log(f"internal: Piyarathne test split, {len(keep)} images")
    log("")
    internal = binary_metrics(
        np.asarray(keep) == positive, predicted == positive, True,
        "[refer vs no_refer, internal]",
        "patient-disjoint split; refer = OPMD + OCA")
    log("")

    # ---- external, SMART-OM
    usable = {"01. Normal": "no_refer", "03. OPMD": "refer"}
    ext_manifest = (PROJECT_ROOT / cfg["paths"]["processed"] / "splits"
                    / "smartom_external" / "validation.csv")
    with open(ext_manifest, newline="", encoding="utf-8") as f:
        ext_rows = [r for r in csv.DictReader(f) if r["category"] in usable]
    log(f"external: SMART-OM, {len(ext_rows)} images, never trained on")

    ext_truth, ext_pred = [], []
    chunk, chunk_truth = [], []
    for row in ext_rows:
        image = imread_unicode(raw / row["path"])
        if image is None:
            continue
        image = cv2.cvtColor(cv2.resize(image, (size, size),
                                        interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        chunk.append(tf.keras.applications.mobilenet_v2.preprocess_input(
            image.astype(np.float32)))
        chunk_truth.append(classes.index(usable[row["category"]]))
        if len(chunk) == 64:
            ext_pred.extend(model.predict(np.stack(chunk), verbose=0).argmax(axis=1))
            ext_truth.extend(chunk_truth)
            chunk, chunk_truth = [], []
    if chunk:
        ext_pred.extend(model.predict(np.stack(chunk), verbose=0).argmax(axis=1))
        ext_truth.extend(chunk_truth)

    log("")
    external = binary_metrics(
        np.asarray(ext_truth) == positive, np.asarray(ext_pred) == positive, True,
        "[refer vs no_refer, external SMART-OM]",
        "different clinic, different cameras; accuracy here is base-rate dominated")
    log("")
    return {"internal": internal, "external": external}


# --------------------------------------------------------------------------
# Track B - lesion segmentation
# --------------------------------------------------------------------------

def track_b(cfg):
    log("=" * 74)
    log("TRACK B - lesion segmentation (models/segmenter.h5, lesion channel)")
    log("=" * 74)
    from src.segmentation.dataset import cached_pair, read_manifest

    threshold = 0.5
    model = tf.keras.models.load_model(
        PROJECT_ROOT / cfg["paths"]["models"] / "segmenter.h5", compile=False)

    rows = read_manifest("test", cfg)
    records = []
    for row in rows:
        stem = Path(row["path"]).stem
        image_path, mask_path = cached_pair(stem, cfg)
        if image_path.exists() and mask_path.exists():
            records.append((stem, image_path, mask_path))
    log(f"internal: Piyarathne test split, {len(records)} cached pairs")

    # Segmentation has no sensitivity in the per-image sense the brief asks
    # for, so it is computed PER PIXEL over the lesion channel - which is the
    # only reading on which those four words mean anything here.
    tp = fp = fn = tn = 0
    dices = []
    for start in range(0, len(records), 16):
        chunk = records[start:start + 16]
        images, truths = [], []
        for _, image_path, mask_path in chunk:
            bgr = cv2.imread(str(image_path))
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                          / 127.5 - 1.0)
            mask = cv2.imread(str(mask_path))
            truths.append(mask[..., 1] > 127)          # G channel = lesion
        predicted = model.predict(np.stack(images), verbose=0)[..., 1] > threshold
        for truth, pred in zip(truths, predicted):
            tp += int(np.logical_and(truth, pred).sum())
            fp += int(np.logical_and(~truth, pred).sum())
            fn += int(np.logical_and(truth, ~pred).sum())
            tn += int(np.logical_and(~truth, ~pred).sum())
            total = truth.sum() + pred.sum()
            if total:
                dices.append(2.0 * np.logical_and(truth, pred).sum() / total)

    pixels = tp + fp + fn + tn
    log("")
    log("  [lesion pixels, internal]")
    log(f"    class balance : {tp + fn}/{pixels} lesion pixels "
        f"({100.0 * (tp + fn) / pixels:.2f}%)  -  heavily imbalanced, which is "
        f"why pixel accuracy is meaningless here")
    log(f"    sensitivity   : {rate(tp, tp + fn)}")
    log(f"    specificity   : {rate(tn, tn + fp)}")
    log(f"    PPV           : {rate(tp, tp + fp)}")
    log(f"    NPV           : {rate(tn, tn + fn)}")
    log(f"    mean Dice     : {np.mean(dices):.3f}  (n={len(dices)} images with "
        f"a lesion in truth or prediction)")
    log("")
    log("  EXTERNAL VALIDATION: none exists for this model.")
    log("  No second dataset carries lesion polygons. SMART-OM annotations were")
    log("  excluded in Phase 1 as burned into the images. Track B has NO")
    log("  out-of-domain figure and cannot be given one today.")
    log("")
    return {"sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "npv": tn / (tn + fn) if tn + fn else None,
            "dice": float(np.mean(dices)), "n_images": len(records)}


def _with_intervals(results):
    """Attach a Wilson interval to every rate, from its own numerator pair."""
    out = {}
    for name, m in results.items():
        entry = dict(m)
        for metric, (num, den) in {
            "sensitivity": (m["tp"], m["tp"] + m["fn"]),
            "specificity": (m["tn"], m["tn"] + m["fp"]),
            "ppv": (m["tp"], m["tp"] + m["fp"]),
            "npv": (m["tn"], m["tn"] + m["fn"]),
        }.items():
            if den:
                value, low, high = wilson_interval(num, den)
                entry[metric] = {"value": round(value, 4),
                                 "ci": [round(low, 4), round(high, 4)],
                                 "numerator": int(num), "denominator": int(den)}
            else:
                entry[metric] = None
        out[name] = entry
    return out


def main():
    cfg = load_config()
    log(f"TRACK RE-MEASUREMENT - {date.today().isoformat()}")
    log("Every figure below is computed fresh from the saved checkpoint and the")
    log("stored splits. None is copied from a log.")
    log("")

    a = track_a(cfg)
    c = track_c(cfg)
    b = track_b(cfg)

    log("=" * 74)
    log("SUMMARY - what would travel with each track score")
    log("=" * 74)
    if c:
        log(f"  Track C internal sensitivity : {c['internal']['sensitivity']:.3f}")
        log(f"  Track C external sensitivity : {c['external']['sensitivity']:.3f}"
            f"   ({c['external']['positives']}/{c['external']['n']} positive)")
    if b:
        log(f"  Track B pixel sensitivity    : {b['sensitivity']:.3f}")
        log(f"  Track B mean Dice            : {b['dice']:.3f}")
    if a:
        for name, m in a.items():
            log(f"  Track A {name:<12} sens    : {m['sensitivity']:.3f}")
    log("")
    log("OralTwin is a screening aid, not a diagnostic tool.")

    out = PROJECT_ROOT / cfg["paths"]["models"] / "logs" / "track_remeasurement.txt"
    out.write_text("\n".join(LINES), encoding="utf-8")

    # Machine-readable, because src/tracks/ reads these figures rather than
    # carrying typed copies of them. A number typed into the code beside the
    # number it is supposed to equal is two numbers, and they diverge.
    payload = {
        "measured_on": date.today().isoformat(),
        "script": "scripts/remeasure_tracks.py",
        "track_a": _with_intervals(a) if a else None,
        "track_b": b,
        "track_c": {k: _with_intervals({k: v})[k] for k, v in c.items()} if c else None,
    }
    json_path = (PROJECT_ROOT / cfg["paths"]["models"] / "logs"
                 / "track_remeasurement.json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
