"""tf.data pipelines built from the split manifests.

The Piyarathne splits are CSV manifests naming images in the read-only raw tree,
not copied image folders, so this loader reads a manifest and maps each row's
category through the task mapping in configs/config.yaml.

Two tasks are defined there and both are available:
  binary_referral  Healthy+Benign -> no_refer, OPMD+OCA -> refer
  four_class       the original categories

Images are read from a 224px cache built once by scripts/cache_piyarathne.py;
decoding the originals (median 4000x3000) every epoch would dominate training
time. The cache lives under data/processed and is never redistributed.
"""

import csv
from collections import Counter
from pathlib import Path

import tensorflow as tf

from src.utils.config import PROJECT_ROOT, load_config

AUTOTUNE = tf.data.AUTOTUNE
SPLITS = ("train", "val", "test")


def task_config(cfg=None, task=None):
    cfg = cfg or load_config()
    name = task or cfg["classification"]["task"]
    spec = cfg["classification"]["tasks"][name]
    return name, list(spec["classes"]), dict(spec["mapping"])


def manifest_path(split, cfg=None, source="piyarathne"):
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"]["processed"] / "splits" / source / f"{split}.csv"


def cached_image_path(row, cfg=None):
    """Where the 224px copy of a manifest row lives."""
    cfg = cfg or load_config()
    cache = PROJECT_ROOT / cfg["classification"]["piyarathne_cache"]
    return cache / (Path(row["path"]).stem + ".jpg")


def read_split(split, cfg=None, task=None, source="piyarathne"):
    """(paths, labels, class_names, raw_categories) for one split."""
    cfg = cfg or load_config()
    _, classes, mapping = task_config(cfg, task)
    index = {name: i for i, name in enumerate(classes)}

    paths, labels, categories = [], [], []
    with open(manifest_path(split, cfg, source), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            category = row["category"]
            if category not in mapping:
                continue
            cached = cached_image_path(row, cfg)
            if not cached.exists():
                continue
            paths.append(str(cached))
            labels.append(index[mapping[category]])
            categories.append(category)
    return paths, labels, classes, categories


def class_weights(cfg=None, task=None, source="piyarathne"):
    _, labels, classes, _ = read_split("train", cfg, task, source)
    counts = Counter(labels)
    total = len(labels)
    return {i: total / (len(classes) * max(counts.get(i, 0), 1))
            for i in range(len(classes))}


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


def make_dataset(split, cfg=None, task=None, shuffle=False, augment=False,
                 source="piyarathne"):
    cfg = cfg or load_config()
    size = cfg["classification"]["img_size"]
    batch = cfg["classification"]["batch_size"]
    seed = cfg["split"]["seed"]

    paths, labels, _, _ = read_split(split, cfg, task, source)
    if not paths:
        raise FileNotFoundError(
            f"no cached images for split '{split}'. Run scripts/cache_piyarathne.py"
        )
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (size, size))
        return image, label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch)
    if augment:
        augmenter = build_augmenter(cfg)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    return ds.map(lambda x, y: (preprocess(x), y),
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def describe(cfg=None, task=None, source="piyarathne"):
    cfg = cfg or load_config()
    name, classes, mapping = task_config(cfg, task)
    lines = [f"task: {name}   classes: {classes}", f"mapping: {mapping}"]
    for split in SPLITS:
        paths, labels, _, categories = read_split(split, cfg, task, source)
        by_class = Counter(classes[i] for i in labels)
        by_category = Counter(categories)
        lines.append(f"  {split:<6} {len(paths):>5} images   "
                     + "  ".join(f"{c}={by_class.get(c, 0)}" for c in classes)
                     + "   | categories " + dict(by_category).__str__())
    return "\n".join(lines)
