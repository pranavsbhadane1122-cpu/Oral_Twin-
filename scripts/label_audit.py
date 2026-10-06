"""Can each Piyarathne image support the label it carries?

Two things make a label unsupportable:

  minimal intraoral view   the annotators' own Oral Cavity polygon covers very
                           little of the frame, so there is barely any mouth in
                           the picture to judge
  refer without a lesion   an image labelled OPMD or OCA that carries no Lesion
                           polygon - the annotators marked no lesion, yet the
                           label says refer

Areas are computed with the shoelace formula from the polygon itself. The
`area` field in the annotation file is not used: it disagrees with the polygon
by orders of magnitude (one annotation records area 6,990,438 for a 464x562
bounding box), so it cannot be trusted.

Writes data/processed/label_audit.csv, one row per image.
Diagnostic only - nothing is excluded and no split is touched.
READ-ONLY on data/raw.

Usage: python scripts/label_audit.py
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.classification.manifest_dataset import SPLITS, manifest_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402

REFER_CATEGORIES = {"OPMD", "OCA"}
MINIMAL_VIEW_DECILE = 0.10


def polygons_of(annotation):
    """Rings as Nx2 arrays. segmentation is a FLAT [x1,y1,...] list here."""
    seg = annotation.get("segmentation")
    if not isinstance(seg, list) or not seg:
        return []
    rings = [seg] if isinstance(seg[0], (int, float)) else [r for r in seg
                                                            if isinstance(r, list)]
    out = []
    for ring in rings:
        if len(ring) >= 6:
            out.append(np.asarray(ring, dtype=np.float64).reshape(-1, 2))
    return out


def shoelace_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main():
    cfg = load_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"] / "piyarathne_oral"
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    out_csv = processed / "label_audit.csv"

    annotation_file = max((p for p in raw.rglob("*.json")),
                          key=lambda p: p.stat().st_size)
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    file_of = {img["id"]: img["file_name"] for img in data["images"]}

    cavity_area, lesion_count = defaultdict(float), Counter()
    for annotation in data["annotations"]:
        name = names.get(annotation.get("category_id"))
        rings = polygons_of(annotation)
        if name == "Oral Cavity":
            if rings:
                cavity_area[annotation["image_id"]] += max(
                    shoelace_area(r) for r in rings)
        elif name == "Lesion":
            lesion_count[annotation["image_id"]] += 1

    # categories and splits
    imagewise = next(raw.rglob("Imagewise*.csv"))
    with open(imagewise, newline="", encoding="utf-8-sig") as f:
        category_of = {str(r["Image Name"]).strip(): str(r["Category"]).strip()
                       for r in csv.DictReader(f)}
    split_of = {}
    for split in SPLITS:
        path = manifest_path(split, cfg)
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    split_of[row["image_name"]] = split

    on_disk = {}
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            on_disk[p.stem] = p

    rows = []
    unreadable = 0
    for image_id, file_name in file_of.items():
        stem = Path(file_name).stem
        path = on_disk.get(stem)
        if path is None:
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            unreadable += 1
            continue
        frame = float(width * height)
        fraction = cavity_area.get(image_id, 0.0) / frame if frame else 0.0
        rows.append({
            "image_name": stem,
            "category": category_of.get(stem, "?"),
            "split": split_of.get(stem, "?"),
            "width": width,
            "height": height,
            "oral_cavity_area_fraction": round(min(fraction, 1.0), 5),
            "lesion_polygon_count": int(lesion_count.get(image_id, 0)),
        })
    print(f"images measured: {len(rows)}   unreadable: {unreadable}")

    fractions = np.asarray([r["oral_cavity_area_fraction"] for r in rows])
    cutoff = float(np.percentile(fractions, MINIMAL_VIEW_DECILE * 100))
    print(f"oral-cavity area fraction: min {fractions.min():.4f}  "
          f"p10 {cutoff:.4f}  median {np.median(fractions):.4f}  "
          f"p90 {np.percentile(fractions, 90):.4f}  max {fractions.max():.4f}")

    for row in rows:
        minimal = row["oral_cavity_area_fraction"] <= cutoff
        refer = row["category"] in REFER_CATEGORIES
        no_lesion = row["lesion_polygon_count"] == 0
        row["flag_minimal_view"] = int(minimal)
        row["flag_refer_no_lesion"] = int(refer and no_lesion)
        # labelled refer AND (no lesion polygon OR minimal intraoral view)
        row["flag_combined"] = int(refer and (no_lesion or minimal))

    fields = ["image_name", "category", "split", "width", "height",
              "oral_cavity_area_fraction", "lesion_polygon_count",
              "flag_minimal_view", "flag_refer_no_lesion", "flag_combined"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n(a) minimal intraoral view (bottom decile, fraction <= {cutoff:.4f}): "
          f"{sum(r['flag_minimal_view'] for r in rows)}")
    by_category = defaultdict(lambda: Counter())
    for row in rows:
        by_category[row["category"]]["n"] += 1
        by_category[row["category"]]["minimal"] += row["flag_minimal_view"]
        if row["lesion_polygon_count"] == 0:
            by_category[row["category"]]["no_lesion"] += 1

    print("\n(b) lesion-polygon presence by category:")
    print(f"  {'category':<10} {'n':>5} {'no lesion':>10} {'share':>7} "
          f"{'minimal view':>13}")
    for category in sorted(by_category):
        bucket = by_category[category]
        share = bucket["no_lesion"] / max(bucket["n"], 1)
        marker = "  <-- label-support candidates" if category in REFER_CATEGORIES else ""
        print(f"  {category:<10} {bucket['n']:>5} {bucket['no_lesion']:>10} "
              f"{share:>6.1%} {bucket['minimal']:>13}{marker}")

    flagged = [r for r in rows if r["flag_combined"]]
    print(f"\n(c) combined flag (refer AND (no lesion OR minimal view)): {len(flagged)}")
    refer_total = sum(1 for r in rows if r["category"] in REFER_CATEGORIES)
    print(f"    = {len(flagged) / max(refer_total, 1):.1%} of the referral class "
          f"({refer_total} images)")
    print(f"    reasons: no lesion polygon "
          f"{sum(1 for r in flagged if r['lesion_polygon_count'] == 0)}, "
          f"minimal view {sum(1 for r in flagged if r['flag_minimal_view'])}")

    print("\n    distribution across splits:")
    for split in SPLITS:
        in_split = [r for r in rows if r["split"] == split]
        refer_in_split = [r for r in in_split if r["category"] in REFER_CATEGORIES]
        flagged_in_split = [r for r in in_split if r["flag_combined"]]
        share = len(flagged_in_split) / max(len(refer_in_split), 1)
        print(f"      {split:<6} {len(flagged_in_split):>4} flagged of "
              f"{len(refer_in_split):>4} refer images ({share:>5.1%})")

    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
