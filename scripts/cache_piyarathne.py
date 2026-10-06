"""Make 224px copies of the Piyarathne images listed in the split manifests.

The originals have a median resolution of 4000x3000; decoding them every epoch
would dominate training time. This writes one resized JPEG per manifest row into
data/processed/piyarathne_224/.

Licence: these are derivatives of a CC BY-NC-ND dataset. They stay under
data/processed, which is git-ignored, and must never be redistributed - which is
also why training runs locally rather than by uploading a bundle to Colab.

READ-ONLY on data/raw.

Usage: python scripts/cache_piyarathne.py
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.classification.manifest_dataset import SPLITS, manifest_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    size = cfg["classification"]["img_size"]
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    cache = PROJECT_ROOT / cfg["classification"]["piyarathne_cache"]
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in SPLITS:
        path = manifest_path(split, cfg)
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    print(f"manifest rows: {len(rows)}")

    written = skipped = failed = 0
    for i, row in enumerate(rows, 1):
        target = cache / (Path(row["path"]).stem + ".jpg")
        if target.exists():
            skipped += 1
            continue
        source = raw / row["path"]
        try:
            with Image.open(source) as im:
                im.draft("RGB", (size * 2, size * 2))
                im.convert("RGB").resize((size, size), Image.BILINEAR).save(
                    target, quality=92)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"  failed {source.name}: {exc!r}")
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print(f"\nwritten {written}, already present {skipped}, failed {failed}")
    print(f"cache: {cache}  ({len(list(cache.glob('*.jpg')))} files)")


if __name__ == "__main__":
    main()
