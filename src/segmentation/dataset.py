"""Rasterise the Piyarathne polygon annotations into two-channel masks.

Channels: oral_cavity, lesion.

Two details this file exists to get right:

  flat polygons   the `segmentation` field is a FLAT [x1, y1, x2, y2, ...] list,
                  not COCO's list-of-lists. An earlier renderer assumed the COCO
                  shape and silently drew bounding boxes instead of polygons for
                  weeks, so parsing is centralised here and tested.
  degenerate ring N-226-01 carries an Oral Cavity polygon of zero area. It is
                  skipped without failing; that image simply has an empty cavity
                  channel.

Polygons are in ORIGINAL image coordinates and are scaled to the output size as
they are filled, so no full-resolution mask is ever allocated.

Splits come from the existing patient-disjoint manifests, unchanged.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.utils.config import PROJECT_ROOT, load_config

AUTOTUNE = tf.data.AUTOTUNE
SPLITS = ("train", "val", "test")
CAVITY, LESION = "Oral Cavity", "Lesion"


def polygon_rings(annotation):
    """Rings as Nx2 float arrays, accepting flat or nested `segmentation`."""
    seg = annotation.get("segmentation")
    if not isinstance(seg, list) or not seg:
        return []
    rings = [seg] if isinstance(seg[0], (int, float)) else [
        r for r in seg if isinstance(r, list)]
    out = []
    for ring in rings:
        if len(ring) >= 6:                      # at least a triangle
            out.append(np.asarray(ring, dtype=np.float64).reshape(-1, 2))
    return out


def ring_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def rasterise(rings, source_size, out_size, min_area=1.0):
    """Fill rings (original coordinates) into a uint8 mask of out_size.

    source_size: (width, height) of the original photograph.
    Degenerate rings - fewer than 3 points or essentially zero area - are
    skipped rather than raising.
    """
    width, height = source_size
    mask = np.zeros((out_size, out_size), np.uint8)
    if not width or not height:
        return mask
    scale_x, scale_y = out_size / float(width), out_size / float(height)
    polygons = []
    for ring in rings:
        if len(ring) < 3 or ring_area(ring) < min_area:
            continue
        scaled = np.empty_like(ring)
        scaled[:, 0] = ring[:, 0] * scale_x
        scaled[:, 1] = ring[:, 1] * scale_y
        polygons.append(np.round(scaled).astype(np.int32))
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    return mask


def load_annotations(cfg=None):
    """{image stem: {channel: [rings]}} plus {stem: (width, height)} if known."""
    cfg = cfg or load_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"] / "piyarathne_oral"
    annotation_file = max(raw.rglob("*.json"), key=lambda p: p.stat().st_size)
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    stem_of = {img["id"]: Path(img["file_name"]).stem for img in data["images"]}

    by_stem = {}
    for annotation in data["annotations"]:
        stem = stem_of.get(annotation.get("image_id"))
        channel = names.get(annotation.get("category_id"))
        if stem is None or channel not in (CAVITY, LESION):
            continue
        by_stem.setdefault(stem, {CAVITY: [], LESION: []})[channel].extend(
            polygon_rings(annotation))
    return by_stem


def read_manifest(split, cfg=None):
    cfg = cfg or load_config()
    path = (PROJECT_ROOT / cfg["paths"]["processed"] / "splits" / "piyarathne"
            / f"{split}.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cache_dir(cfg=None):
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["segmentation"]["cache"]


def cached_pair(stem, cfg=None):
    directory = cache_dir(cfg)
    return directory / f"{stem}.jpg", directory / f"{stem}_mask.png"


def read_split(split, cfg=None):
    """(image paths, mask paths) for the cached entries of one split."""
    cfg = cfg or load_config()
    images, masks = [], []
    for row in read_manifest(split, cfg):
        stem = Path(row["path"]).stem
        image_path, mask_path = cached_pair(stem, cfg)
        if image_path.exists() and mask_path.exists():
            images.append(str(image_path))
            masks.append(str(mask_path))
    return images, masks


def _rotate_pair(image, mask, degrees):
    """Rotate image (bilinear) and mask (nearest) by the same angle."""
    size = image.shape[0]
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), float(degrees), 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (size, size),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
    rotated_mask = cv2.warpAffine(mask, matrix, (size, size),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rotated_mask.ndim == 2:
        rotated_mask = rotated_mask[..., None]
    return rotated_image.astype(np.float32), rotated_mask.astype(np.float32)


def augment(image, mask, cfg):
    """Geometric transforms applied identically to image and mask.

    The mask is resampled with nearest-neighbour throughout: interpolating a
    label map would invent part-labelled pixels at every boundary.
    """
    acfg = cfg["segmentation"]["augment"]
    size = cfg["segmentation"]["img_size"]

    if tf.random.uniform(()) < 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    degrees = tf.random.uniform((), -acfg["rotation_deg"], acfg["rotation_deg"])
    image, mask = tf.numpy_function(
        _rotate_pair, [image, mask, degrees], [tf.float32, tf.float32])
    image.set_shape([size, size, 3])
    mask.set_shape([size, size, 2])

    scale = 1.0 + tf.random.uniform((), -acfg["zoom"], acfg["zoom"])
    scaled = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
    image = tf.image.resize(image, (scaled, scaled))
    mask = tf.image.resize(mask, (scaled, scaled), method="nearest")
    image = tf.image.resize_with_crop_or_pad(image, size, size)
    mask = tf.image.resize_with_crop_or_pad(mask, size, size)

    image = tf.image.random_brightness(image, acfg["brightness"] * 255.0)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, mask


def make_dataset(split, cfg=None, shuffle=False, augment_data=False, limit=None):
    cfg = cfg or load_config()
    size = cfg["segmentation"]["img_size"]
    batch = cfg["segmentation"]["batch_size"]
    seed = cfg["split"]["seed"]

    images, masks = read_split(split, cfg)
    if limit:
        images, masks = images[:limit], masks[:limit]
    if not images:
        raise FileNotFoundError(
            f"no cached segmentation data for '{split}'. "
            f"Run scripts/cache_segmentation.py")

    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)

    def load(image_path, mask_path):
        image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(image, (size, size))
        raw = tf.io.decode_png(tf.io.read_file(mask_path), channels=3)
        raw = tf.image.resize(raw, (size, size), method="nearest")
        mask = tf.cast(raw[..., :2] > 127, tf.float32)   # R = cavity, G = lesion
        return image, mask

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(lambda i, m: augment(i, m, cfg), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda i, m: (i / 127.5 - 1.0, m), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def describe(cfg=None):
    cfg = cfg or load_config()
    lines = []
    for split in SPLITS:
        images, _ = read_split(split, cfg)
        lines.append(f"  {split:<6} {len(images):>5} cached image/mask pairs")
    return "\n".join(lines)
