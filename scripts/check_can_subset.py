"""Is the "Ca N" subset a within-dataset provenance shortcut?

SMART-OM's Oral Cancer category is named differently from the rest of the
dataset. If those files also differ systematically in capture properties, then a
model could identify the cancer class from the way the picture was taken rather
than from the tissue - the same failure that invalidated the Phase 2 classifier,
but inside a single dataset.

Compares the "Ca N" / bare-numeric files against the SMITA-named files on
resolution, aspect ratio, file size, EXIF make/model, and per-channel mean and
standard deviation. Reports the numbers; the decision is the reader's.

Writes data/processed/can_subset_check.txt. READ-ONLY on data/raw.

Usage: python scripts/check_can_subset.py
"""

import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ExifTags  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.smartom import category_of, list_images, patient_id_of  # noqa: E402

LEGACY = re.compile(r"^(\d+)\s*-")


def scheme_of(path):
    if patient_id_of(path):
        return "SMITA"
    if LEGACY.match(path.name):
        return "legacy"
    return "CaN"


def exif_make_model(im):
    try:
        raw = im.getexif()
    except Exception:
        return None
    if not raw:
        return None
    tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
    make = str(tags.get("Make", "")).strip()
    model = str(tags.get("Model", "")).strip()
    both = f"{make} {model}".strip()
    return both or None


def measure(path):
    with Image.open(path) as im:
        width, height = im.size
        fmt = im.format
        exif = exif_make_model(im)
        small = im.convert("RGB").resize((128, 128), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
    return {
        "path": path,
        "width": width,
        "height": height,
        "megapixels": width * height / 1e6,
        "aspect": width / height if height else 0.0,
        "bytes": path.stat().st_size,
        "format": fmt,
        "exif": exif,
        "mean": arr.reshape(-1, 3).mean(axis=0),
        "std": arr.reshape(-1, 3).std(axis=0),
    }


def summarise(name, rows, log):
    if not rows:
        log(f"  {name}: none")
        return
    def stat(key):
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        return values.mean(), np.median(values), values.min(), values.max()

    log(f"  {name}  (n={len(rows)})")
    for key, unit in (("megapixels", "MP"), ("aspect", ""), ("bytes", "bytes")):
        mean, median, low, high = stat(key)
        if key == "bytes":
            log(f"    {key:<12} median {median/1e6:8.2f} MB   "
                f"range {low/1e6:.2f} - {high/1e6:.2f} MB")
        else:
            log(f"    {key:<12} median {median:8.2f} {unit:<6} "
                f"range {low:.2f} - {high:.2f}")
    means = np.stack([r["mean"] for r in rows])
    stds = np.stack([r["std"] for r in rows])
    log(f"    channel mean  R {means[:,0].mean():6.1f}  G {means[:,1].mean():6.1f}  "
        f"B {means[:,2].mean():6.1f}")
    log(f"    channel std   R {stds[:,0].mean():6.1f}  G {stds[:,1].mean():6.1f}  "
        f"B {stds[:,2].mean():6.1f}")
    log(f"    formats: {dict(Counter(r['format'] for r in rows))}")
    exifs = Counter(r["exif"] or "(none)" for r in rows)
    log(f"    EXIF make/model: {dict(exifs.most_common(5))}")


def main():
    cfg = load_config()
    out = PROJECT_ROOT / cfg["paths"]["processed"] / "can_subset_check.txt"
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    images = list_images(cfg)
    groups = defaultdict(list)
    for path in images:
        groups[(category_of(path, cfg), scheme_of(path))].append(path)

    log("IS THE \"Ca N\" SUBSET A PROVENANCE SHORTCUT? (data/raw is read-only)")
    log("")
    log("composition of SMART-OM by category and naming scheme:")
    for (category, scheme), paths in sorted(groups.items()):
        log(f"  {category:<28} {scheme:<8} {len(paths):>5}")

    can = [p for p in images if scheme_of(p) == "CaN"]
    smita = [p for p in images if scheme_of(p) == "SMITA"]
    smita_cancer = [p for p in can if patient_id_of(p)]
    log("")
    log(f"SMITA-named images in the Oral Cancer category: {len(smita_cancer)}")
    if not smita_cancer:
        log("  *** The comparison asked for cannot be run: the Oral Cancer category")
        log("  *** contains NO SMITA-named images. Naming scheme and cancer label are")
        log("  *** perfectly confounded, so the comparison below is against the")
        log("  *** SMITA-named images of the OTHER categories.")

    log("\nmeasuring ...")
    can_rows = [measure(p) for p in can]
    smita_rows = [measure(p) for p in smita]
    opmd_rows = [r for r in smita_rows
                 if category_of(r["path"], cfg) == "03. OPMD"]

    log("")
    summarise("Ca N / bare numeric (all Oral Cancer)", can_rows, log)
    log("")
    summarise("SMITA-named, all other categories", smita_rows, log)
    log("")
    summarise("SMITA-named, OPMD only (nearest disease category)", opmd_rows, log)

    # separability: can a trivial rule tell the two groups apart?
    log("\nseparability on capture properties alone:")
    can_mp = np.asarray([r["megapixels"] for r in can_rows])
    smita_mp = np.asarray([r["megapixels"] for r in smita_rows])
    log(f"  megapixels: Ca N range {can_mp.min():.2f}-{can_mp.max():.2f}, "
        f"SMITA range {smita_mp.min():.2f}-{smita_mp.max():.2f}")
    overlap = (can_mp.min() <= smita_mp.max()) and (smita_mp.min() <= can_mp.max())
    log(f"  ranges overlap: {overlap}")
    can_png = sum(1 for r in can_rows if r["format"] == "PNG")
    smita_png = sum(1 for r in smita_rows if r["format"] == "PNG")
    log(f"  PNG files: Ca N {can_png}/{len(can_rows)}  "
        f"SMITA {smita_png}/{len(smita_rows)}")
    can_exif = sum(1 for r in can_rows if r["exif"])
    smita_exif = sum(1 for r in smita_rows if r["exif"])
    log(f"  carrying EXIF make/model: Ca N {can_exif}/{len(can_rows)}  "
        f"SMITA {smita_exif}/{len(smita_rows)}")

    # duplicates inside the Ca N set (Ca 2.jpg vs Ca 2.png and friends)
    log("\nduplicate check inside the Ca N subset:")
    by_hash = defaultdict(list)
    for path in can:
        with Image.open(path) as im:
            small = im.convert("RGB").resize((64, 64), Image.BILINEAR)
            digest = hashlib.md5(np.asarray(small).tobytes()).hexdigest()
        by_hash[digest].append(path.name)
    dupes = {h: names for h, names in by_hash.items() if len(names) > 1}
    log(f"  visually identical groups: {len(dupes)}")
    for names in dupes.values():
        log(f"    {sorted(names)}")
    log(f"  => {len(can)} files represent {len(by_hash)} distinct photographs")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
