"""tf.data loaders for the processed classification splits.

Class list, image size, batch size and augmentation strengths all come from
configs/config.yaml. Splits live in data/processed/classification/{train,val,
test}/<class>/. Folders for classes not in the config (e.g. the pending
"healthy" class) are simply ignored.

Augmentation (train only): random horizontal flip, rotation, zoom,
brightness/contrast jitter. Pixels are fed through MobileNetV2's
preprocess_input ([-1, 1] scaling).
"""

from pathlib import Path

import tensorflow as tf

from src.utils.config import PROJECT_ROOT, load_config

AUTOTUNE = tf.data.AUTOTUNE


def split_dir(cfg, split):
    return PROJECT_ROOT / cfg["paths"]["processed"] / "classification" / split


def list_files(cfg, split):
    """Return (paths, labels) for one split, ordered deterministically."""
    classes = cfg["classification"]["classes"]
    paths, labels = [], []
    for label, cls in enumerate(classes):
        for p in sorted((split_dir(cfg, split) / cls).glob("*.jpg")):
            paths.append(str(p))
            labels.append(label)
    return paths, labels


def class_counts(cfg, split="train"):
    classes = cfg["classification"]["classes"]
    _, labels = list_files(cfg, split)
    return {cls: labels.count(i) for i, cls in enumerate(classes)}


def compute_class_weights(cfg):
    """Inverse-frequency weights, mean-normalized so loss scale stays stable:
    weight[c] = n_total / (n_classes * n_c)."""
    counts = class_counts(cfg, "train")
    total = sum(counts.values())
    n = len(counts)
    return {
        i: total / (n * max(c, 1))
        for i, c in enumerate(counts[cls] for cls in cfg["classification"]["classes"])
    }


def build_augmenter(cfg):
    a = cfg["classification"]["augment"]
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(a["rotation_deg"] / 360.0, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(a["zoom"], fill_mode="reflect"),
            tf.keras.layers.RandomBrightness(a["brightness"], value_range=(0.0, 255.0)),
            tf.keras.layers.RandomContrast(a["contrast"]),
        ],
        name="augmentation",
    )


def make_dataset(cfg, split, shuffle=False, augment=False):
    img_size = cfg["classification"]["img_size"]
    batch_size = cfg["classification"]["batch_size"]
    seed = cfg["split"]["seed"]

    paths, labels = list_files(cfg, split)
    if not paths:
        raise FileNotFoundError(f"no images for split '{split}' - run prepare_classification")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (img_size, img_size))
        return img, label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size)

    if augment:
        augmenter = build_augmenter(cfg)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def load_datasets(cfg=None):
    """Returns (train_ds, val_ds, test_ds, class_names, class_weights)."""
    cfg = cfg or load_config()
    train = make_dataset(cfg, "train", shuffle=True, augment=True)
    val = make_dataset(cfg, "val")
    test = make_dataset(cfg, "test")
    return train, val, test, list(cfg["classification"]["classes"]), compute_class_weights(cfg)
