"""Parse the Piyarathne annotation file and render sample mask overlays.

Reports how many images carry lesion annotations, what geometry the annotations
use, and whether every image reference resolves to a file on disk. Writes
data/processed/annotations_summary.txt and renders a few overlays for review.

LICENCE: the Piyarathne dataset is CC BY-NC-ND 4.0 - no redistribution of the
images or of derivatives. The rendered overlays stay under data/processed,
which is git-ignored, and must not be committed, published or shared.

READ-ONLY on data/raw.

Usage: python scripts/audit_annotations.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode, imwrite_jpg  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def polygons_of(ann):
    """Return a list of [x, y] arrays.

    This file does not use the COCO convention: segmentation is a FLAT list
    [x1, y1, x2, y2, ...] rather than a list of such lists, so both shapes are
    accepted here.
    """
    seg = ann.get("segmentation")
    if not isinstance(seg, list) or not seg:
        return []
    if isinstance(seg[0], (int, float)):
        rings = [seg]
    else:
        rings = [r for r in seg if isinstance(r, list)]
    out = []
    for ring in rings:
        if len(ring) >= 6:
            out.append(np.asarray(ring, dtype=np.float32).reshape(-1, 2))
    return out


def load_annotation_file(root, log):
    candidates = sorted(p for p in root.rglob("*.json") if p.is_file())
    log(f"  JSON files under the dataset: {len(candidates)}")
    for p in candidates:
        log(f"    {p.relative_to(root)}  ({p.stat().st_size / 1e6:.1f} MB)")
    if not candidates:
        return None, None
    biggest = max(candidates, key=lambda p: p.stat().st_size)
    log(f"\n  parsing {biggest.relative_to(root)}")
    try:
        data = json.loads(biggest.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"  PARSE FAILED: {exc!r}")
        return biggest, None
    return biggest, data


def summarise_coco(data, root, cfg, log):
    if not isinstance(data, dict):
        log(f"  top level is {type(data).__name__}, not a COCO dict")
        return None
    log(f"  top-level keys: {list(data)}")
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    log(f"  images entries:      {len(images)}")
    log(f"  annotation entries:  {len(annotations)}")
    log(f"  categories:          {[c.get('name') for c in categories]}")

    if images:
        log(f"  example image entry: { {k: images[0][k] for k in list(images[0])[:8]} }")
    if annotations:
        first = annotations[0]
        log(f"  example annotation keys: {list(first)}")

    geometry = Counter()
    points_per_poly = []
    for ann in annotations:
        seg = ann.get("segmentation")
        rings = polygons_of(ann)
        if rings:
            geometry["polygon (flat list)" if isinstance(seg[0], (int, float))
                     else "polygon (nested)"] += 1
            points_per_poly += [len(r) for r in rings]
        elif isinstance(seg, dict):
            geometry["RLE"] += 1
        elif ann.get("bbox"):
            geometry["bbox only"] += 1
        else:
            geometry["none"] += 1
    log(f"\n  annotation geometry: {dict(geometry)}")
    if points_per_poly:
        arr = np.asarray(points_per_poly)
        log(f"  polygon vertices: min {arr.min()} median {int(np.median(arr))} "
            f"max {arr.max()}  ({len(arr)} polygons)")

    by_image = defaultdict(list)
    for ann in annotations:
        by_image[ann.get("image_id")].append(ann)
    log(f"\n  images carrying at least one annotation: {len(by_image)} "
        f"of {len(images)} ({100.0 * len(by_image) / max(len(images), 1):.1f}%)")
    per_image = [len(v) for v in by_image.values()]
    if per_image:
        arr = np.asarray(per_image)
        log(f"  annotations per annotated image: min {arr.min()} "
            f"median {int(np.median(arr))} max {arr.max()}")

    if categories:
        names = {c["id"]: c.get("name") for c in categories}
        per_cat = Counter(names.get(a.get("category_id"), "?") for a in annotations)
        log(f"  annotations per category: {dict(per_cat)}")

    disk = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            disk[p.name] = p
            disk.setdefault(p.stem, p)
    resolved, unresolved = {}, []
    for entry in images:
        name = entry.get("file_name") or entry.get("filename") or ""
        key = Path(name).name
        target = disk.get(key) or disk.get(Path(key).stem)
        if target is None:
            unresolved.append(name)
        else:
            resolved[entry.get("id")] = target
    log(f"\n  image references resolving to real files: {len(resolved)} / {len(images)}")
    log(f"  unresolved references: {len(unresolved)}")
    for name in unresolved[:5]:
        log(f"    {name}")
    return {"images": images, "annotations": annotations, "by_image": by_image,
            "resolved": resolved, "categories": categories}


def render_samples(parsed, cfg, out_dir, log):
    n_samples = cfg["audit"]["annotation_samples"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    names = {c["id"]: c.get("name") for c in parsed["categories"]}
    candidates = [(image_id, anns) for image_id, anns in parsed["by_image"].items()
                  if image_id in parsed["resolved"]]
    candidates.sort(key=lambda kv: -len(kv[1]))
    chosen = candidates[:n_samples]
    log(f"\n  rendering {len(chosen)} overlays to {out_dir}")

    for rank, (image_id, anns) in enumerate(chosen, 1):
        path = parsed["resolved"][image_id]
        img = imread_unicode(path)
        if img is None:
            log(f"    could not read {path.name}")
            continue
        scale = 900.0 / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        else:
            scale = 1.0
        overlay = img.copy()
        for ann in anns:
            lesion = names.get(ann.get("category_id")) == "Lesion"
            colour = (0, 80, 255) if lesion else (0, 230, 120)
            for ring in polygons_of(ann):
                pts = (ring * scale).astype(np.int32)
                cv2.polylines(overlay, [pts], True, colour, 3 if lesion else 2)
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4 and lesion:
                x, y, w, h = [v * scale for v in bbox]
                cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)),
                              (255, 190, 0), 1)
        blended = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)
        n_lesion = sum(1 for a in anns if names.get(a.get("category_id")) == "Lesion")
        label = (f"{path.name}  {len(anns)} annotations  "
                 f"({n_lesion} lesion, red) (cavity, green)")
        cv2.rectangle(blended, (0, 0), (blended.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(blended, label[:70], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        target = out_dir / f"sample_{rank:02d}.jpg"
        imwrite_jpg(target, blended)
        log(f"    {target.name}  <- {path.name}  ({len(anns)} annotations)")


def main():
    cfg = load_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"] / "piyarathne_oral"
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    out = processed / "annotations_summary.txt"
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("PIYARATHNE ANNOTATION AUDIT (data/raw is read-only)")
    log("LICENCE: CC BY-NC-ND 4.0 - renders stay in data/processed, never shared.")
    if not raw.exists():
        log("  dataset missing")
    else:
        path, data = load_annotation_file(raw, log)
        parsed = summarise_coco(data, raw, cfg, log) if data is not None else None
        if parsed:
            render_samples(parsed, cfg, processed / "annotation_samples", log)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
