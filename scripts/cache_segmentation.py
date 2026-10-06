"""Build the segmentation cache: 256px images and their rasterised masks.

One JPEG and one PNG per image. The PNG carries both channels: R = oral cavity,
G = lesion. Rasterising once means training never re-reads a 4000x3000 original.

Licence: these are derivatives of a CC BY-NC-ND dataset and stay under
data/processed, which is git-ignored. Training is local for the same reason.

READ-ONLY on data/raw.

Usage: python scripts/cache_segmentation.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from src.segmentation.dataset import (  # noqa: E402
    CAVITY,
    LESION,
    SPLITS,
    cache_dir,
    load_annotations,
    rasterise,
    read_manifest,
)
from src.utils.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    size = cfg["segmentation"]["img_size"]
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    out_dir = cache_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading polygon annotations ...")
    annotations = load_annotations(cfg)
    print(f"  images with annotations: {len(annotations)}")

    rows = []
    for split in SPLITS:
        for row in read_manifest(split, cfg):
            rows.append((split, row))
    print(f"manifest rows: {len(rows)}")

    stats = Counter()
    empty_cavity, empty_lesion = [], []
    for i, (split, row) in enumerate(rows, 1):
        stem = Path(row["path"]).stem
        source = raw / row["path"]
        image_path = out_dir / f"{stem}.jpg"
        mask_path = out_dir / f"{stem}_mask.png"
        if image_path.exists() and mask_path.exists():
            stats["already cached"] += 1
            continue
        try:
            with Image.open(source) as im:
                width, height = im.size
                im.draft("RGB", (size * 2, size * 2))
                resized = im.convert("RGB").resize((size, size), Image.BILINEAR)
            resized.save(image_path, quality=92)
        except Exception as exc:
            stats["unreadable image"] += 1
            print(f"  skipped {stem}: {exc!r}")
            continue

        rings = annotations.get(stem, {CAVITY: [], LESION: []})
        cavity = rasterise(rings.get(CAVITY, []), (width, height), size)
        lesion = rasterise(rings.get(LESION, []), (width, height), size)
        if not cavity.any():
            empty_cavity.append(stem)
        if not lesion.any():
            empty_lesion.append(stem)

        mask = np.zeros((size, size, 3), np.uint8)
        mask[..., 0] = cavity          # R
        mask[..., 1] = lesion          # G
        cv2.imwrite(str(mask_path), mask[..., ::-1])   # cv2 writes BGR
        stats["written"] += 1
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print(f"\n{dict(stats)}")
    print(f"images with an EMPTY oral-cavity mask: {len(empty_cavity)} "
          f"{empty_cavity[:5]}")
    print(f"images with an empty lesion mask: {len(empty_lesion)} "
          f"(expected for every Healthy image)")
    print(f"cache: {out_dir}")


if __name__ == "__main__":
    main()
