"""Train the U-Net segmenter on CPU.

Loss is bce_weight * BCE + (1 - bce_weight) * Dice; see model.combined_loss for
why BCE alone is not enough on this data.

Checkpoints on the best validation LESION Dice, not on total loss. The cavity
channel is easy - it fills most of the frame - so total loss is dominated by a
channel we do not actually care about. The lesion channel is the one the gate is
written against, so that is what selects the saved weights.

CONSTRAINT: Piyarathne never leaves this machine. Local CPU training only.

Usage:
  python -m src.segmentation.train --epochs 2 --timing   # the timing probe
  python -m src.segmentation.train                       # the full run
"""

import argparse
import csv
import time
from datetime import timedelta

import tensorflow as tf

from src.segmentation.dataset import describe, make_dataset
from src.segmentation.model import build_and_compile
from src.utils.config import PROJECT_ROOT, load_config


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._start
        self.times.append(elapsed)
        print(f"  epoch {epoch + 1} wall-clock: {elapsed / 60:.2f} min",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="override segmentation.epochs")
    parser.add_argument("--timing", action="store_true",
                        help="timing probe: do not write the checkpoint or history")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N images of each split")
    args = parser.parse_args()

    cfg = load_config()
    scfg = cfg["segmentation"]
    epochs = args.epochs or scfg["epochs"]

    print("=" * 74)
    print("PHASE 3 - U-Net segmentation" + ("  [TIMING PROBE]" if args.timing else ""))
    print("=" * 74)
    print(describe(cfg))
    print(f"img_size {scfg['img_size']}  batch {scfg['batch_size']}  "
          f"base_filters {scfg['base_filters']}  depth {scfg['depth']}")

    train_ds = make_dataset("train", cfg, shuffle=True, augment_data=True,
                            limit=args.limit)
    val_ds = make_dataset("val", cfg, limit=args.limit)

    model = build_and_compile(cfg)
    params = model.count_params()
    print(f"parameters: {params:,}")
    print(f"epochs requested: {epochs}")
    print("")

    timer = EpochTimer()
    callbacks = [timer]
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    checkpoint = models_dir / "segmenter.h5"
    history_path = models_dir / "logs" / "segmenter_history.csv"

    if not args.timing:
        models_dir.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks += [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_dice_lesion", mode="max",
                save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_dice_lesion", mode="max",
                patience=scfg["early_stop_patience"],
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.CSVLogger(str(history_path)),
        ]

    started = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=1)
    total = time.time() - started

    print("")
    print("-" * 74)
    print(f"epochs run: {len(timer.times)}   total {timedelta(seconds=int(total))}")
    if timer.times:
        # the first epoch pays for tf.data warm-up and graph tracing, so the
        # honest per-epoch figure is the median of the rest when we have them
        steady = timer.times[1:] or timer.times
        per_epoch = sorted(steady)[len(steady) // 2]
        print(f"first epoch {timer.times[0] / 60:.2f} min, "
              f"steady-state {per_epoch / 60:.2f} min/epoch")
        if args.timing:
            full = scfg["epochs"] * per_epoch + (timer.times[0] - per_epoch)
            print("")
            print(f"EXTRAPOLATION to the full {scfg['epochs']}-epoch run: "
                  f"{timedelta(seconds=int(full))}")
            print("  (early stopping would likely cut this short; "
                  "this is the worst case)")

    if not args.timing:
        best = max(history.history.get("val_dice_lesion", [0.0]))
        print(f"best val lesion Dice: {best:.4f}")
        print(f"checkpoint: {checkpoint}")
        print(f"history:    {history_path}")
    print("OralTwin is a screening aid, not a diagnostic tool.")


if __name__ == "__main__":
    main()
