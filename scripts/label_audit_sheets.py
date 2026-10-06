"""Contact sheets for the label audit: flagged images and unflagged controls.

Renders 30 flagged images (S-184-01 forced in as a known positive) and 15
unflagged controls, each captioned with its label, oral-cavity area fraction and
lesion-polygon count, so the flag rule can be judged by eye rather than trusted.

LICENCE: Piyarathne is CC BY-NC-ND. These sheets stay under data/processed,
which is git-ignored, and must not be committed, published or shared.

READ-ONLY on data/raw.

Usage: python scripts/label_audit_sheets.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode, imwrite_jpg  # noqa: E402

KNOWN_POSITIVE = "S-184-01"
N_FLAGGED = 30
N_CONTROLS = 15
CELL = 230


def sheet(rows, lookup, title, path, columns=6):
    count = len(rows)
    sheet_rows = int(np.ceil(count / columns))
    caption = 30
    canvas = np.full((sheet_rows * (CELL + caption) + 34, columns * CELL, 3), 22, np.uint8)
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    for i, row in enumerate(rows):
        path_on_disk = lookup.get(row["image_name"])
        if path_on_disk is None:
            continue
        image = imread_unicode(path_on_disk)
        if image is None:
            continue
        thumb = cv2.resize(image, (CELL, CELL), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, columns)
        y0 = 34 + r * (CELL + caption)
        x0 = c * CELL
        canvas[y0:y0 + CELL, x0:x0 + CELL] = thumb
        label = f"{row['category']}  cav {float(row['oral_cavity_area_fraction']):.3f}"
        cv2.putText(canvas, row["image_name"][:22], (x0 + 4, y0 + CELL + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 + 4, y0 + CELL + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    imwrite_jpg(path, canvas, quality=90)


def main():
    cfg = load_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"] / "piyarathne_oral"
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    out_dir = processed / "label_audit_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(processed / "label_audit.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lookup = {}
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lookup[p.stem] = p

    rng = np.random.default_rng(cfg["split"]["seed"])
    flagged = [r for r in rows if r["flag_combined"] == "1"]
    controls = [r for r in rows
                if r["flag_combined"] == "0" and r["category"] in {"OPMD", "OCA"}]

    known = [r for r in flagged if r["image_name"] == KNOWN_POSITIVE]
    others = [r for r in flagged if r["image_name"] != KNOWN_POSITIVE]
    picked = known + [others[i] for i in
                      rng.choice(len(others), size=min(N_FLAGGED - len(known),
                                                       len(others)), replace=False)]
    picked.sort(key=lambda r: float(r["oral_cavity_area_fraction"]))
    control_pick = [controls[i] for i in
                    rng.choice(len(controls), size=min(N_CONTROLS, len(controls)),
                               replace=False)]
    control_pick.sort(key=lambda r: float(r["oral_cavity_area_fraction"]))

    sheet(picked, lookup,
          f"FLAGGED ({len(flagged)} total) - refer label, minimal intraoral view "
          f"- sorted by cavity fraction", out_dir / "flagged.jpg")
    sheet(control_pick, lookup,
          f"CONTROLS - refer label, NOT flagged ({len(controls)} total)",
          out_dir / "controls.jpg")

    print(f"flagged sheet: {len(picked)} images (known positive "
          f"{KNOWN_POSITIVE} included)")
    print(f"control sheet: {len(control_pick)} images")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
