"""Evaluate the referral classifier, with intervals on every rate.

Primary: the binary referral task - accuracy, sensitivity, specificity, PPV and
NPV, each with a 95% Wilson interval.

Secondary: the breakdown over the four original categories. The model is binary,
so for OPMD and OCA this is the share correctly referred, and for Healthy and
Benign the share correctly not referred. The OCA cell is flagged underpowered:
19 test images cannot support a sensitivity claim.

Writes models/logs/eval_report_<task>.txt and a confusion-matrix image.

Usage: python -m src.classification.evaluate_referral
"""

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.classification.manifest_dataset import make_dataset, read_split, task_config
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.stats import format_rate, interval_width, wilson_interval

UNDERPOWERED_BELOW = 30
DISCLAIMER = ("OralTwin is a screening aid, not a diagnostic tool. "
              "Have any finding checked by a dentist.")


def save_confusion(matrix, classes, path, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config()
    task, classes, mapping = task_config(cfg, args.task)
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    logs_dir = models_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"classifier_{task}.h5"

    model = tf.keras.models.load_model(model_path, compile=False)
    dataset = make_dataset(args.split, cfg, args.task)
    paths, labels, _, categories = read_split(args.split, cfg, args.task)

    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    truth = np.asarray(labels)
    categories = np.asarray(categories)

    lines = ["=" * 74, f"EVALUATION - {task} ({args.split} split)", "=" * 74, "",
             f"model: {model_path.relative_to(PROJECT_ROOT)}",
             f"split: Piyarathne, patient-disjoint (see data/processed/SPLIT_DESIGN.md)",
             f"images: {len(truth)}", "",
             "Every rate carries a 95% Wilson interval. At these sample sizes a point",
             "estimate on its own would overstate what the data supports.", ""]

    matrix = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(truth, predicted):
        matrix[t, p] += 1

    if len(classes) == 2:
        positive = classes.index("refer") if "refer" in classes else 1
        negative = 1 - positive
        true_positive = int(matrix[positive, positive])
        false_negative = int(matrix[positive, negative])
        true_negative = int(matrix[negative, negative])
        false_positive = int(matrix[negative, positive])

        lines += ["PRIMARY - binary referral", ""]
        lines.append(f"  accuracy     {format_rate(true_positive + true_negative, len(truth))}")
        lines.append(f"  sensitivity  {format_rate(true_positive, true_positive + false_negative)}"
                     f"   (of cases needing referral, how many were referred)")
        lines.append(f"  specificity  {format_rate(true_negative, true_negative + false_positive)}"
                     f"   (of cases not needing referral, how many were spared)")
        lines.append(f"  PPV          {format_rate(true_positive, true_positive + false_positive)}")
        lines.append(f"  NPV          {format_rate(true_negative, true_negative + false_negative)}")
        lines += ["", f"  confusion: TP {true_positive}  FP {false_positive}  "
                      f"FN {false_negative}  TN {true_negative}", ""]
        sensitivity = wilson_interval(true_positive, true_positive + false_negative)[0]
        lines.append(f"  missed referrals: {false_negative} of "
                     f"{true_positive + false_negative} "
                     f"({1 - sensitivity:.1%} of cases that needed one)")
        lines.append("")

    lines += ["SECONDARY - breakdown by original category", ""]
    lines.append(f"  {'category':<10} {'n':>5}  {'handled correctly':<34} note")
    for category in sorted(set(categories.tolist())):
        mask = categories == category
        target = classes.index(mapping[category])
        correct = int((predicted[mask] == target).sum())
        total = int(mask.sum())
        note = ""
        if total < UNDERPOWERED_BELOW:
            note = f"UNDERPOWERED (n={total}); interval spans {interval_width(correct, total):.2f}"
        wanted = "referred" if mapping[category] == "refer" else "not referred"
        lines.append(f"  {category:<10} {total:>5}  {format_rate(correct, total):<34} "
                     f"{wanted}{'  ' + note if note else ''}")
    lines.append("")
    lines.append("  The OCA cell is the cancer sensitivity estimate and it is small by")
    lines.append("  design: patient-disjoint splitting left 19 OCA images in test. Read")
    lines.append("  its interval, not its point estimate.")

    lines += ["", "confusion matrix (rows true, cols predicted): " + str(classes),
              str(matrix), "", DISCLAIMER, ""]

    report = "\n".join(lines)
    (logs_dir / f"eval_report_{task}.txt").write_text(report, encoding="utf-8")
    save_confusion(matrix, classes, logs_dir / f"confusion_matrix_{task}.png",
                   f"{task} ({args.split})")
    print(report)
    print(f"saved {logs_dir / f'eval_report_{task}.txt'}")


if __name__ == "__main__":
    main()
