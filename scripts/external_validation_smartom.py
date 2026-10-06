"""External validation of the referral model on SMART-OM.

SMART-OM never contributed to training, so this is the honest estimate of how
the model behaves on photographs from a different clinic, different cameras and
a different population. A large drop from internal to external performance is a
finding to report, not a failure to hide.

Scope, fixed by the split design:
  OPMD vs Normal only, coarse labels (Normal -> no_refer, OPMD -> refer)
  Variation from normal is left out: it is neither clearly referable nor clearly
  not, and forcing it into a binary would make the number meaningless
  Oral Cancer is excluded entirely - acquisition is confounded with the label,
  and assert_scoreable() refuses it

Writes models/logs/external_validation_smartom.txt.

Usage: python scripts/external_validation_smartom.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.classification.manifest_dataset import task_config  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode  # noqa: E402
from src.utils.smartom import (  # noqa: E402
    CONFOUNDED_CATEGORY,
    ConfoundedSubsetError,
    assert_scoreable,
)
from src.utils.stats import format_rate, wilson_interval  # noqa: E402

USABLE = {"01. Normal": "no_refer", "03. OPMD": "refer"}


def main():
    cfg = load_config()
    task, classes, _ = task_config(cfg)
    size = cfg["classification"]["img_size"]
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    manifest = (PROJECT_ROOT / cfg["paths"]["processed"] / "splits"
                / "smartom_external" / "validation.csv")
    out = PROJECT_ROOT / cfg["paths"]["models"] / "logs" / "external_validation_smartom.txt"
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("=" * 74)
    log("EXTERNAL VALIDATION - SMART-OM (never trained on)")
    log("=" * 74)
    log("")

    # the guard must refuse the confounded subset; demonstrate it rather than
    # merely assert it in a test
    try:
        assert_scoreable([CONFOUNDED_CATEGORY])
        log("WARNING: the confounded-subset guard did NOT refuse. Investigate.")
    except ConfoundedSubsetError:
        log(f"guard check: scoring '{CONFOUNDED_CATEGORY}' was refused, as intended.")
    log(f"scope: {list(USABLE)} only; Variation from normal and Oral Cancer excluded.")
    log("")

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["category"] in USABLE]
    log(f"images: {len(rows)}")

    index = {name: i for i, name in enumerate(classes)}
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    model_path = PROJECT_ROOT / cfg["paths"]["models"] / f"classifier_{task}.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    log(f"model: {model_path.name}")

    batch, images, truth, categories = [], [], [], []
    predictions = []

    def flush():
        if not batch:
            return
        predictions.extend(model.predict(np.stack(batch), verbose=0).argmax(axis=1))
        batch.clear()

    for i, row in enumerate(rows, 1):
        image = imread_unicode(raw / row["path"])
        if image is None:
            continue
        image = cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        batch.append(preprocess(image.astype(np.float32)))
        truth.append(index[USABLE[row["category"]]])
        categories.append(row["category"])
        if len(batch) == 64:
            flush()
        if i % 500 == 0:
            log(f"  {i}/{len(rows)} ...")
    flush()

    predicted = np.asarray(predictions)
    truth = np.asarray(truth)
    categories = np.asarray(categories)
    positive = index["refer"]
    negative = index["no_refer"]

    true_positive = int(((truth == positive) & (predicted == positive)).sum())
    false_negative = int(((truth == positive) & (predicted == negative)).sum())
    true_negative = int(((truth == negative) & (predicted == negative)).sum())
    false_positive = int(((truth == negative) & (predicted == positive)).sum())

    log("")
    log("results on SMART-OM (OPMD vs Normal):")
    log(f"  accuracy     {format_rate(true_positive + true_negative, len(truth))}")
    log(f"  sensitivity  {format_rate(true_positive, true_positive + false_negative)}"
        f"   (OPMD referred)")
    log(f"  specificity  {format_rate(true_negative, true_negative + false_positive)}"
        f"   (Normal spared)")
    log(f"  PPV          {format_rate(true_positive, max(true_positive + false_positive, 1))}")
    log(f"  confusion: TP {true_positive}  FP {false_positive}  "
        f"FN {false_negative}  TN {true_negative}")
    log("")
    log("  by category:")
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        target = index[USABLE[category]]
        correct = int((predicted[mask] == target).sum())
        log(f"    {category:<28} {format_rate(correct, int(mask.sum()))}")

    # compare with the internal test figures if they are available
    internal = (PROJECT_ROOT / cfg["paths"]["models"] / "logs"
                / f"eval_report_{task}.txt")
    log("")
    if internal.exists():
        log(f"internal (Piyarathne test) figures are in {internal.name};")
        log("the drop from internal to external is the number that matters here.")
    sensitivity = wilson_interval(true_positive, true_positive + false_negative)[0]
    specificity = wilson_interval(true_negative, true_negative + false_positive)[0]
    log("")
    log("Note: the base rate differs sharply from the internal split - SMART-OM is")
    log(f"overwhelmingly Normal ({int((truth == negative).sum())} of {len(truth)}), so "
        f"accuracy here is dominated by")
    log("specificity. Read sensitivity and specificity separately, never accuracy alone.")
    log("")
    log(f"summary: sensitivity {sensitivity:.3f}, specificity {specificity:.3f}")
    log("")
    log("OralTwin is a screening aid, not a diagnostic tool.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
