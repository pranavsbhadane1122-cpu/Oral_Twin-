"""Train the oral-condition classifier (MobileNetV2, transfer learning).

Two stages: (a) frozen ImageNet base, head only at classification.learning_rate;
(b) last classification.fine_tune_layers unfrozen at classification.fine_tune_lr.
Class weights compensate imbalance; EarlyStopping on val_loss; best model saved
to models/classifier_<n>class.h5; history appended to
models/logs/classifier_<n>class_history.csv.

Runs unchanged locally (CPU) and on Google Colab (GPU): the device is detected
at runtime and printed; every path is relative to the project root.

Usage:
  python -m src.classification.train           # full training
  python -m src.classification.train --debug   # 1 epoch/stage, few steps (CPU sanity run)
"""

import argparse

import tensorflow as tf

from src.classification.dataset import load_datasets
from src.classification.model import build_model, compile_model, unfreeze_top
from src.utils.config import PROJECT_ROOT, load_config

DEBUG_STEPS = 5


def describe_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"training on GPU: {[g.name for g in gpus]}")
    else:
        print("training on CPU (no GPU detected) - expect this to be slow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true",
                        help="cap to 1 epoch and a few steps per stage (pipeline sanity run)")
    args = parser.parse_args()

    cfg = load_config()
    ccfg = cfg["classification"]
    n_classes = len(ccfg["classes"])

    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    logs_dir = models_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"classifier_{n_classes}class.h5"
    history_path = logs_dir / f"classifier_{n_classes}class_history.csv"

    describe_device()
    train_ds, val_ds, _, class_names, class_weights = load_datasets(cfg)
    print(f"classes: {class_names}")
    print(f"class weights: {class_weights}")

    model = build_model(n_classes, ccfg["img_size"], ccfg["dropout"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=ccfg["early_stop_patience"],
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_path), append=True),
    ]

    epochs = 1 if args.debug else ccfg["epochs"]
    steps = DEBUG_STEPS if args.debug else None
    val_steps = 2 if args.debug else None
    fit_kwargs = dict(
        validation_data=val_ds, class_weight=class_weights, callbacks=callbacks,
        epochs=epochs, steps_per_epoch=steps, validation_steps=val_steps,
    )

    print("\n=== stage (a): frozen base, training head ===")
    compile_model(model, ccfg["learning_rate"])
    model.fit(train_ds, **fit_kwargs)

    print(f"\n=== stage (b): fine-tuning last {ccfg['fine_tune_layers']} layers ===")
    unfreeze_top(model, ccfg["fine_tune_layers"])
    compile_model(model, ccfg["fine_tune_lr"])
    model.fit(train_ds, **fit_kwargs)

    print(f"\nbest model saved to {ckpt_path}")
    print(f"history log: {history_path}")
    if args.debug:
        print("DEBUG RUN ONLY - not a trained model; run without --debug on GPU for real training")


if __name__ == "__main__":
    main()
