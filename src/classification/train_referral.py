"""Train the referral classifier on the Piyarathne patient-disjoint splits.

Same two-stage recipe as Phase 2: ImageNet-initialised MobileNetV2 with the base
frozen while the head trains, then the last N layers unfrozen at a lower rate.
Class weights compensate imbalance, EarlyStopping watches val_loss.

The task comes from configs/config.yaml (classification.task). The default,
binary_referral, asks the question the product actually asks - does this need a
dentist to look at it - rather than naming a condition.

Training runs locally. The Piyarathne dataset is CC BY-NC-ND, so its images (and
derivatives such as the 224px cache) must not be uploaded to a third-party
service like Colab.

Usage:
  python -m src.classification.train_referral
  python -m src.classification.train_referral --debug
"""

import argparse

import tensorflow as tf

from src.classification.manifest_dataset import (
    class_weights,
    describe,
    make_dataset,
    task_config,
)
from src.classification.model import build_model, compile_model, unfreeze_top
from src.utils.config import PROJECT_ROOT, load_config

DEBUG_STEPS = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    cfg = load_config()
    ccfg = cfg["classification"]
    task, classes, _ = task_config(cfg, args.task)

    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    logs_dir = models_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = models_dir / f"classifier_{task}.h5"
    history = logs_dir / f"classifier_{task}_history.csv"

    gpus = tf.config.list_physical_devices("GPU")
    print(f"device: {'GPU ' + str([g.name for g in gpus]) if gpus else 'CPU'}")
    print(describe(cfg, args.task))

    train_ds = make_dataset("train", cfg, args.task, shuffle=True, augment=True)
    val_ds = make_dataset("val", cfg, args.task)
    weights = class_weights(cfg, args.task)
    print(f"class weights: { {classes[i]: round(w, 3) for i, w in weights.items()} }")

    model = build_model(len(classes), ccfg["img_size"], ccfg["dropout"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=ccfg["early_stop_patience"],
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(history), append=True),
    ]
    epochs = 1 if args.debug else ccfg["epochs"]
    fit_kwargs = dict(validation_data=val_ds, class_weight=weights,
                      callbacks=callbacks, epochs=epochs,
                      steps_per_epoch=DEBUG_STEPS if args.debug else None,
                      validation_steps=2 if args.debug else None)

    print(f"\n=== stage (a): frozen base, task={task} ===")
    compile_model(model, ccfg["learning_rate"])
    model.fit(train_ds, **fit_kwargs)

    print(f"\n=== stage (b): fine-tuning last {ccfg['fine_tune_layers']} layers ===")
    unfreeze_top(model, ccfg["fine_tune_layers"])
    compile_model(model, ccfg["fine_tune_lr"])
    model.fit(train_ds, **fit_kwargs)

    print(f"\nbest model: {checkpoint}")
    print(f"history:    {history}")
    if args.debug:
        print("DEBUG RUN ONLY - not a trained model")


if __name__ == "__main__":
    main()
