"""Compute MD5 and perceptual hashes once for every raw dataset, and cache them.

The later audits (duplicate clustering, cross-dataset overlap) all need the same
hashes over ~9 GB of images, so they are computed once here and cached to
data/processed/audit_cache/hashes.json.

READ-ONLY on data/raw.

Usage: python scripts/audit_hashes.py [dataset ...]
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from src.utils.config import load_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
DATASETS = ["mio", "piyarathne_oral", "oral_diseases", "oral_cancer"]


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def phash64(path):
    """64-bit perceptual hash. JPEG draft mode decodes at reduced size, which
    makes this tractable over thousands of multi-megapixel photographs."""
    with Image.open(path) as im:
        try:
            im.draft("L", (64, 64))
        except Exception:
            pass
        im = im.convert("L").resize((32, 32), Image.BILINEAR)
        gray = np.asarray(im, dtype=np.float32)
    dct = cv2.dct(gray)
    block = dct[:8, :8].flatten()
    block = np.delete(block, 0)
    bits = block > np.median(block)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def main():
    cfg = load_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "audit_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hashes.json"

    names = sys.argv[1:] or DATASETS
    cache = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text(encoding="utf-8"))

    for name in names:
        root = raw / name
        if not root.exists():
            print(f"{name}: MISSING, skipped", flush=True)
            continue
        files = [p for p in sorted(root.rglob("*"))
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        print(f"{name}: hashing {len(files)} images ...", flush=True)
        entries, failed = [], 0
        for i, p in enumerate(files, 1):
            rel = str(p.relative_to(root)).replace("\\", "/")
            try:
                entries.append({"path": rel, "md5": md5_of(p), "phash": str(phash64(p))})
            except Exception:
                failed += 1
            if i % 1000 == 0:
                print(f"   {i}/{len(files)}", flush=True)
        cache[name] = {"root": str(root), "n_images": len(files),
                       "n_failed": failed, "entries": entries}
        print(f"   done: {len(entries)} hashed, {failed} unreadable", flush=True)
        out_path.write_text(json.dumps(cache), encoding="utf-8")

    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
