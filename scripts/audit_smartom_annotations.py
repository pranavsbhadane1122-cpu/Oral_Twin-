"""Parse the SMART-OM VGG Image Annotator JSON files and render our own overlays.

The JSON files sit inside the annotation-level folders, but only their VECTOR
data is read here - never the pre-rendered images beside them. Overlays are
drawn by us onto the clean "01. Unannotated" photograph, which is also the only
image any pipeline is allowed to load (see src/utils/smartom.py).

Reports which unannotated images carry regions, what geometry is used, how many
regions each image has, the label vocabulary, and whether every JSON reference
resolves to a real unannotated file.

Writes data/processed/smartom_annotations_summary.txt and five overlays to
data/processed/smartom_annotation_samples/.

READ-ONLY on data/raw.

Usage: python scripts/audit_smartom_annotations.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.utils.config import PROJECT_ROOT as ROOT  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode, imwrite_jpg  # noqa: E402
from src.utils.smartom import category_of, list_images, smartom_root  # noqa: E402


def level_of(path, root):
    """Which annotation level a JSON file came from."""
    parts = Path(path).relative_to(root).parts
    for part in parts:
        low = part.lower()
        if low.endswith("annotation") and "unannotated" not in low:
            return part
    return "?"


def regions_of(entry):
    """VIA region list for one image entry.

    Some exports carry null entries in the list, so they are dropped here rather
    than guarded at every call site.
    """
    raw = entry.get("regions") or []
    if isinstance(raw, dict):          # older VIA keyed regions by index
        raw = list(raw.values())
    return [r for r in raw if isinstance(r, dict)]


def shape_points(shape):
    """Return (kind, Nx2 points or None) for a VIA shape_attributes block."""
    kind = shape.get("name", "?")
    if kind in ("polygon", "polyline"):
        xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
        if xs and ys and len(xs) == len(ys) and len(xs) >= 3:
            return kind, np.stack([np.asarray(xs, np.float32),
                                   np.asarray(ys, np.float32)], axis=1)
        return kind, None
    if kind == "rect":
        x, y = shape.get("x", 0), shape.get("y", 0)
        w, h = shape.get("width", 0), shape.get("height", 0)
        return kind, np.asarray([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
    if kind in ("circle", "ellipse"):
        cx, cy = shape.get("cx", 0), shape.get("cy", 0)
        rx = shape.get("rx", shape.get("r", 0))
        ry = shape.get("ry", shape.get("r", 0))
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return kind, np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=1)
    return kind, None


DESCRIPTOR_FOR_LEVEL = {
    "02. Region annotation": "01. Descriptors_for_region_annotation.xlsx",
    "03. Full annotation": "02. Descriptors_for_full_annotation.xlsx",
    "04. Lesion annotation": "03. Descriptors_for_lesion_annotation.xlsx",
}


def read_descriptors(path):
    """{file stem (lowercased): [label, ...]} from one Descriptors workbook."""
    import openpyxl

    table = {}
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for sheet in book.worksheets:
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if len(row) < 2 or row[1] is None:
                continue
            key = str(row[1]).strip().lower()
            values = [str(v).strip() for v in row[2:] if v is not None and str(v).strip()]
            if key:
                table[key] = values
    book.close()
    return table


def describe_descriptors(root, resolved, log):
    """Label vocabulary, and whether descriptor counts match region counts."""
    log("\nlabel vocabulary (from the Descriptors workbooks):")
    descriptors = {}
    for level, filename in DESCRIPTOR_FOR_LEVEL.items():
        path = root / "Descriptors" / filename
        if not path.exists():
            log(f"  {level}: workbook missing ({filename})")
            continue
        table = read_descriptors(path)
        descriptors[level] = table
        vocabulary = Counter(label for labels in table.values() for label in labels)
        log(f"\n  {level}  ({filename})")
        log(f"    files described: {len(table)}   distinct labels: {len(vocabulary)}")
        log("    most common labels:")
        for label, n in vocabulary.most_common(12):
            log(f"      {n:>5}  {label}")

    log("\n  do the descriptor counts line up with the region counts?")
    for level, table in descriptors.items():
        exact = mismatch = absent = 0
        examples = []
        for (entry_level, path), regions in resolved.items():
            if entry_level != level:
                continue
            key = Path(path).stem.lower()
            if key not in table:
                absent += 1
                continue
            if len(table[key]) == len(regions):
                exact += 1
            else:
                mismatch += 1
                if len(examples) < 3:
                    examples.append(f"{Path(path).name}: {len(regions)} regions vs "
                                    f"{len(table[key])} descriptors")
        total = exact + mismatch + absent
        log(f"    {level}: {exact}/{total} files match exactly, {mismatch} differ, "
            f"{absent} not in the workbook")
        for example in examples:
            log(f"      e.g. {example}")


def main():
    cfg = load_config()
    root = smartom_root(cfg)
    processed = ROOT / cfg["paths"]["processed"]
    out = processed / "smartom_annotations_summary.txt"
    samples_dir = processed / "smartom_annotation_samples"
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("SMART-OM ANNOTATION AUDIT (data/raw is read-only)")
    log("Only the JSON vector data is read; overlays are drawn by us onto the")
    log("clean unannotated photographs, never taken from the rendered levels.")

    if not root.is_dir():
        log("  SMART-OM missing")
        out.write_text("\n".join(lines), encoding="utf-8")
        return

    # index the only images any pipeline may load
    unannotated = list_images(cfg)
    by_name = {}
    for path in unannotated:
        by_name.setdefault(path.name.lower(), path)
        by_name.setdefault(path.stem.lower(), path)
    log(f"\nunannotated images available to the loader: {len(unannotated)}")

    json_files = sorted(p for p in root.rglob("*.json") if p.is_file())
    log(f"JSON files found: {len(json_files)}")
    per_level = Counter(level_of(p, root) for p in json_files)
    log(f"JSON files per annotation level: {dict(per_level)}")

    entries = []          # (level, filename, regions)
    bad_json = []
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            bad_json.append((path.relative_to(root), repr(exc)))
            continue
        metadata = data.get("_via_img_metadata")
        if metadata is None:
            # some VIA exports are a bare mapping of the same shape
            metadata = {k: v for k, v in data.items() if isinstance(v, dict)
                        and "filename" in v}
        level = level_of(path, root)
        for entry in metadata.values():
            name = entry.get("filename")
            if name:
                entries.append((level, name, regions_of(entry)))

    log(f"\nJSON files that failed to parse: {len(bad_json)}")
    for rel, err in bad_json[:5]:
        log(f"  {rel}  {err}")

    log(f"image entries inside those JSONs: {len(entries)}")
    log(f"  distinct filenames referenced: {len({n for _, n, _ in entries})}")

    # One image can be referenced by more than one VIA project file at the same
    # level. Merging their region lists would double the count, so each (level,
    # image) keeps the single richest entry and the overlap is reported.
    collected, unresolved = defaultdict(list), []
    for level, name, regions in entries:
        target = by_name.get(Path(name).name.lower()) or by_name.get(Path(name).stem.lower())
        if target is None:
            unresolved.append((level, name))
        else:
            collected[(level, str(target))].append(regions)

    duplicated = {key: lists for key, lists in collected.items() if len(lists) > 1}
    resolved = {key: max(lists, key=len) for key, lists in collected.items()}

    referenced_images = {path for (_, path) in resolved}
    log(f"\nreferences resolving to an existing unannotated file: "
        f"{len(entries) - len(unresolved)} of {len(entries)}")
    log(f"unresolved references: {len(unresolved)}")
    for level, name in unresolved[:8]:
        log(f"  {level}: {name}")
    log(f"\n(level, image) pairs referenced by more than one JSON project file: "
        f"{len(duplicated)}")
    log("  the richest entry is kept for each; merging them instead would double "
        "the region count and make it disagree with the descriptors")
    log(f"\ndistinct unannotated images carrying at least one region: "
        f"{len(referenced_images)} of {len(unannotated)} "
        f"({100.0 * len(referenced_images) / max(len(unannotated), 1):.1f}%)")

    by_category = Counter(category_of(Path(p), cfg) for p in referenced_images)
    log(f"annotated images per category: {dict(by_category)}")

    geometry = Counter()
    vertices = []
    labels = Counter()
    label_keys = Counter()
    per_image_counts = []
    for (level, path), regions in resolved.items():
        per_image_counts.append(len(regions))
        for region in regions:
            kind, points = shape_points(region.get("shape_attributes", {}))
            geometry[kind] += 1
            if points is not None and kind in ("polygon", "polyline"):
                vertices.append(len(points))
            for key, value in (region.get("region_attributes") or {}).items():
                label_keys[key] += 1
                if isinstance(value, str) and value.strip():
                    labels[f"{key}={value.strip()}"] += 1

    log(f"\ngeometry types: {dict(geometry)}")
    if vertices:
        arr = np.asarray(vertices)
        log(f"polygon vertices: min {arr.min()} median {int(np.median(arr))} "
            f"max {arr.max()}")
    if per_image_counts:
        arr = np.asarray(per_image_counts)
        log(f"regions per (image, level): min {arr.min()} median "
            f"{int(np.median(arr))} mean {arr.mean():.1f} max {arr.max()}")
    log(f"\nregion attribute keys inside the JSON: {dict(label_keys)}")
    if not label_keys:
        log("  The JSON carries GEOMETRY ONLY - no labels. The labels live in the")
        log("  Descriptors workbooks, one ordered row per file: the Nth descriptor")
        log("  names the Nth region of that file.")
    describe_descriptors(root, resolved, log)

    # five overlays, drawn by us on the unannotated image
    samples_dir.mkdir(parents=True, exist_ok=True)
    for old in samples_dir.glob("*.jpg"):
        old.unlink()
    richest = sorted(resolved.items(), key=lambda kv: -len(kv[1]))[:5]
    log(f"\nrendering {len(richest)} overlays onto UNANNOTATED images -> {samples_dir}")
    for rank, ((level, path), regions) in enumerate(richest, 1):
        img = imread_unicode(path)
        if img is None:
            log(f"  could not read {path}")
            continue
        scale = 1000.0 / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        else:
            scale = 1.0
        overlay = img.copy()
        for region in regions:
            kind, points = shape_points(region.get("shape_attributes", {}))
            if points is None:
                continue
            pts = (points * scale).astype(np.int32)
            cv2.polylines(overlay, [pts], kind != "polyline", (0, 80, 255), 2)
        blended = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)
        caption = f"{Path(path).name}  {level}  {len(regions)} regions"
        cv2.rectangle(blended, (0, 0), (blended.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(blended, caption[:78], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        target = samples_dir / f"smartom_sample_{rank:02d}.jpg"
        imwrite_jpg(target, blended)
        log(f"  {target.name}  <- {Path(path).name}  ({level}, {len(regions)} regions)")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
