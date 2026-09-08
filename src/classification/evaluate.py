"""Evaluate the saved classifier on the test split.

Reports overall accuracy plus per-class precision, recall (sensitivity),
specificity and F1; saves the confusion matrix image to
models/logs/confusion_matrix_<n>class.png and the text report to
models/logs/eval_report_<n>class.txt.

Usage: python -m src.classification.evaluate
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from src.classification.dataset import make_dataset
from src.utils.config import PROJECT_ROOT, load_config

DISCLAIMER = ("NOTE: OralTwin is a screening aid, not a diagnostic tool. Any finding "
              "must be checked by a dentist.")


def per_class_metrics(cm):
    """Rows = per class dict(precision, recall/sensitivity, specificity, f1)."""
    metrics = []
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
        metrics.append(dict(precision=precision, recall=recall,
                            specificity=specificity, f1=f1, support=int(cm[i, :].sum())))
    return metrics


def save_confusion_png(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test split)")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    classes = cfg["classification"]["classes"]
    n = len(classes)
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    logs_dir = models_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"classifier_{n}class.h5"

    model = tf.keras.models.load_model(model_path)
    print(f"loaded {model_path}")

    test_ds = make_dataset(cfg, "test")
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = y_prob.argmax(axis=1)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    metrics = per_class_metrics(cm)

    lines = [f"Evaluation report - classifier_{n}class ({', '.join(classes)})",
             f"test images: {len(y_true)}", f"overall accuracy: {acc:.4f}", "",
             f"{'class':<12} {'precision':>9} {'recall':>9} {'specificity':>11} "
             f"{'f1':>7} {'support':>8}"]
    for cls, m in zip(classes, metrics):
        lines.append(f"{cls:<12} {m['precision']:>9.4f} {m['recall']:>9.4f} "
                     f"{m['specificity']:>11.4f} {m['f1']:>7.4f} {m['support']:>8}")
    lines += ["", "confusion matrix (rows=true, cols=predicted):",
              str(cm), "", DISCLAIMER, ""]
    report = "\n".join(lines)

    png_path = logs_dir / f"confusion_matrix_{n}class.png"
    txt_path = logs_dir / f"eval_report_{n}class.txt"
    save_confusion_png(cm, classes, png_path)
    txt_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved {png_path}\nsaved {txt_path}")


if __name__ == "__main__":
    main()
