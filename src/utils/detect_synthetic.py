"""Flag cartoon / illustration images among the raw photos - no training involved.

data/raw is READ-ONLY: this module only measures, writes
data/processed/synthetic_flags.csv, and renders contact sheets for a human to
approve a threshold. Nothing is ever excluded here; exclusion happens later in
prepare_classification.py.

Why classical cues work here
----------------------------
A camera photograph and a drawn illustration differ in ways that need no model:

  local_entropy    Mean Shannon entropy of 16x16 luminance blocks. Biological
                   tissue is textured even after heavy JPEG compression;
                   drawn fills and smooth vector gradients are not. This is
                   the strongest cue for this dataset.
  noise_sigma      Every sensor leaves a noise floor. Immerkaer's estimator
                   (Immerkaer, CVIU 1996) convolves with a Laplacian-like
                   kernel that cancels smooth content, leaving noise.
                   Illustrations are rendered, not captured.
  midtone_flat_frac  Share of NON-CLIPPED pixels with essentially zero local
                   gradient. Drawn fills are perfectly flat.
  unique_color_frac  Distinct 5-bit-quantised colours per pixel. Photographs
                   spread across a continuum; drawings do not.

Why clipped pixels are excluded
-------------------------------
A first version of this detector scored blown-out augmented photographs at the
very top: aggressive brightness/contrast augmentation clips whole regions to
pure white or black, which looks exactly like a flat drawn fill and destroys
the local noise floor. Those images are degraded photographs, not
illustrations, and excluding them from a classifier is a separate decision from
excluding drawings. So every texture cue here is measured only over pixels that
are NOT clipped, and clip_frac is reported separately as its own quality flag.

Each cue is mapped to a 0-1 "looks drawn" indicator through a soft ramp, then
averaged with the weights in SCORE_WEIGHTS. The result is a score in [0, 1];
the decision threshold is a human's to set, not this module's.

Measurements are taken on a centre crop of the ORIGINAL file (never a resized
copy) because resampling destroys exactly the noise signature the detector
relies on.

Usage:
  python -m src.utils.detect_synthetic              # score + write CSV
  python -m src.utils.detect_synthetic --sheets 0.5 # also render contact sheets
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import IMAGE_EXTS, imread_unicode, imwrite_jpg

MAX_SIDE = 512  # centre-crop cap; no resizing, to preserve the noise floor
CLIP_FRAC_DEGRADED = 0.60  # above this the image is a blown-out photo, not a drawing

SCORE_WEIGHTS = {
    "entropy": 0.40,      # tissue is textured, drawn fills are not
    "flat": 0.35,         # flat *unclipped* fills separate drawings sharply
    "noise": 0.15,
    "unique_color": 0.10,
}

# Soft ramps: (value_that_scores_0, value_that_scores_1). Calibrated against
# measured values on this corpus - real photos sit near entropy 3.0-3.3 with
# flat_frac < 0.01, the known illustration at entropy 1.43 / flat_frac 0.29.
RAMPS = {
    "entropy": (3.4, 1.5),        # low block entropy -> drawn
    "flat": (0.02, 0.25),         # flat non-clipped fills -> drawn
    "noise": (2.0, 0.25),         # low noise floor -> drawn
    "unique_color": (0.12, 0.01),  # few distinct colours -> drawn
}

CLIP_LOW, CLIP_HIGH = 6, 249  # channel values treated as clipped


def centre_crop(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    ch, cw = min(h, max_side), min(w, max_side)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def clipped_mask(img_bgr):
    """Pixels where any channel is crushed to black or blown to white."""
    low = (img_bgr <= CLIP_LOW).any(axis=2)
    high = (img_bgr >= CLIP_HIGH).any(axis=2)
    return low | high


def noise_sigma(gray, keep=None):
    """Immerkaer noise estimate over the kept (non-clipped) pixels."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.0
    inner = np.abs(response[1:-1, 1:-1])
    if keep is not None:
        sel = keep[1:-1, 1:-1]
        inner = inner[sel] if sel.any() else inner
    return float(np.sqrt(np.pi / 2.0) * inner.mean() / 6.0)


def block_entropy(gray, keep, block=16, levels=32):
    """Mean Shannon entropy (bits) of luminance blocks that are mostly unclipped.

    Textured tissue keeps high entropy even through JPEG compression; drawn
    fills and smooth vector gradients collapse toward zero.
    """
    h, w = gray.shape[:2]
    quantised = (gray.astype(np.uint16) * levels // 256).astype(np.uint8)
    entropies = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            mask = keep[y:y + block, x:x + block]
            if mask.mean() < 0.6:            # mostly clipped: says nothing about texture
                continue
            patch = quantised[y:y + block, x:x + block][mask]
            if patch.size < 32:
                continue
            counts = np.bincount(patch, minlength=levels).astype(np.float64)
            probabilities = counts[counts > 0] / patch.size
            entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
    # None means "not measurable" (almost everything was clipped), which is very
    # different from "measured as perfectly smooth"
    return float(np.mean(entropies)) if entropies else None


def palette_stats(img_bgr):
    """(top-16 colour share, unique colours per pixel) after 5-bit quantisation."""
    quantised = (img_bgr >> 3).astype(np.uint16)
    packed = (quantised[..., 0] << 10) | (quantised[..., 1] << 5) | quantised[..., 2]
    flat = packed.ravel()
    counts = Counter(flat.tolist())
    total = flat.size
    top = sum(n for _, n in counts.most_common(16))
    return top / total, len(counts) / total


def flat_fraction(gray, keep=None):
    """Share of NON-CLIPPED pixels with essentially no local gradient."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    flat = magnitude < 2.0
    if keep is not None and keep.any():
        return float(flat[keep].mean())
    return float(flat.mean())


def ramp(value, low, high):
    """Map value to [0,1] along a linear ramp; handles inverted ramps."""
    if high == low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def measure(path):
    """Return the feature dict for one image, or None if unreadable."""
    img = imread_unicode(path)
    if img is None or img.size == 0:
        return None
    img = centre_crop(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    clipped = clipped_mask(img)
    keep = ~clipped
    top_frac, unique_frac = palette_stats(img)
    entropy = block_entropy(gray, keep)
    features = {
        "noise_sigma": noise_sigma(gray, keep),
        "local_entropy": entropy,
        "top_color_frac": top_frac,
        "unique_color_frac": unique_frac,
        "flat_frac": flat_fraction(gray, keep),
        "clip_frac": float(clipped.mean()),
        "sat_mean": float(hsv[..., 1].mean()),
        "height": img.shape[0],
        "width": img.shape[1],
    }
    indicators = {
        # unmeasurable texture is not evidence of a drawing
        "entropy": 0.0 if entropy is None else ramp(entropy, *RAMPS["entropy"]),
        "noise": ramp(features["noise_sigma"], *RAMPS["noise"]),
        "unique_color": ramp(unique_frac, *RAMPS["unique_color"]),
        "flat": ramp(features["flat_frac"], *RAMPS["flat"]),
    }
    features["score"] = float(sum(SCORE_WEIGHTS[k] * v for k, v in indicators.items()))
    features["entropy_measurable"] = entropy is not None
    features.update({f"ind_{k}": round(v, 4) for k, v in indicators.items()})
    return features


def source_dirs(cfg):
    """The raw folders that feed the classification task."""
    raw = PROJECT_ROOT / cfg["paths"]["raw"] / "oral_diseases"
    return {
        "caries": [raw / "Data caries" / "Data caries" / "caries orignal data set",
                   raw / "Data caries" / "Data caries" / "caries augmented data set"],
        "calculus": [raw / "Calculus"],
        "gingivitis": [raw / "Gingivitis"],
    }


def scan(cfg=None, log=print):
    cfg = cfg or load_config()
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    rows = []
    for cls, dirs in source_dirs(cfg).items():
        for directory in dirs:
            for path in sorted(Path(directory).rglob("*")):
                if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
                    continue
                features = measure(path)
                if features is None:
                    continue
                features["path"] = str(path.relative_to(raw_root))
                features["source_class"] = cls
                rows.append(features)
        log(f"  {cls}: {sum(1 for r in rows if r['source_class'] == cls)} scored")
    return rows


def label_for(row, threshold):
    """Three-way label. 'clipped' is a degraded photograph, not a drawing, and
    is kept separate so the two exclusions can be decided independently."""
    if row["clip_frac"] >= CLIP_FRAC_DEGRADED:
        return "clipped"
    return "synthetic" if row["score"] >= threshold else "photo"


def write_csv(rows, out_path, threshold):
    fields = ["path", "source_class", "score", "proposed_label", "entropy_measurable",
              "local_entropy",
              "noise_sigma", "flat_frac", "unique_color_frac", "clip_frac",
              "top_color_frac", "sat_mean", "ind_entropy", "ind_noise", "ind_flat",
              "ind_unique_color", "width", "height"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["score"] = round(row["score"], 4)
            row["proposed_label"] = label_for(row, threshold)
            for key in ("local_entropy", "noise_sigma", "top_color_frac",
                        "unique_color_frac", "flat_frac", "clip_frac", "sat_mean"):
                if row.get(key) is not None:
                    row[key] = round(row[key], 4)
            writer.writerow(row)


def contact_sheet(rows, raw_root, title, path, cols=5, cell=180):
    """Grid of thumbnails, each captioned with its score and source class."""
    rows = list(rows)
    n_rows = int(np.ceil(len(rows) / cols))
    caption_h = 26
    sheet = np.full((n_rows * (cell + caption_h) + 34, cols * cell, 3), 24, np.uint8)
    cv2.putText(sheet, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)

    for i, row in enumerate(rows):
        img = imread_unicode(raw_root / row["path"])
        if img is None:
            continue
        thumb = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = 34 + r * (cell + caption_h)
        x0 = c * cell
        sheet[y0:y0 + cell, x0:x0 + cell] = thumb
        label = f"{row['score']:.2f} {row['source_class'][:4]}"
        cv2.putText(sheet, label, (x0 + 4, y0 + cell + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, Path(row["path"]).name[:22], (x0 + 4, y0 + cell + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (170, 200, 255), 1, cv2.LINE_AA)
    imwrite_jpg(path, sheet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="proposed decision threshold (a human approves this)")
    parser.add_argument("--sheets", action="store_true", help="render contact sheets")
    parser.add_argument("--n", type=int, default=20, help="samples per contact sheet")
    args = parser.parse_args()

    cfg = load_config()
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"]

    print("scanning raw classification sources (read-only) ...")
    rows = scan(cfg)
    scores = np.array([r["score"] for r in rows])
    write_csv(rows, out_dir / "synthetic_flags.csv", args.threshold)

    print(f"\nscored {len(rows)} images")
    print(f"score distribution: min {scores.min():.3f}  p25 {np.percentile(scores, 25):.3f}  "
          f"median {np.median(scores):.3f}  p75 {np.percentile(scores, 75):.3f}  "
          f"max {scores.max():.3f}")
    n_clipped = sum(1 for r in rows if r["clip_frac"] >= CLIP_FRAC_DEGRADED)
    print(f"separately flagged as clipped/degraded (clip_frac >= {CLIP_FRAC_DEGRADED}): "
          f"{n_clipped} ({100.0*n_clipped/len(rows):.1f}%)")
    print("\nhow many would be called drawings at each threshold "
          "(clipped images excluded from this count):")
    candidates = [r for r in rows if r["clip_frac"] < CLIP_FRAC_DEGRADED]
    cand_scores = np.array([r["score"] for r in candidates])
    for t in (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        n = int((cand_scores >= t).sum())
        print(f"  threshold {t:.2f} -> {n:5d} flagged as drawn "
              f"({100.0*n/len(rows):.2f}% of all images)")
    print()
    for cls in sorted({r["source_class"] for r in rows}):
        subset = [r for r in rows if r["source_class"] == cls]
        n = sum(1 for r in subset if label_for(r, args.threshold) == "synthetic")
        nc = sum(1 for r in subset if label_for(r, args.threshold) == "clipped")
        print(f"  {cls}: {n}/{len(subset)} drawn, {nc}/{len(subset)} clipped "
              f"(at threshold {args.threshold:.2f})")

    if args.sheets:
        rng = np.random.default_rng(cfg["split"]["seed"])
        flagged = [r for r in rows if label_for(r, args.threshold) == "synthetic"]
        unflagged = [r for r in rows if label_for(r, args.threshold) == "photo"]
        clipped_rows = [r for r in rows if label_for(r, args.threshold) == "clipped"]
        sheets_dir = out_dir / "synthetic_review"
        sheets_dir.mkdir(parents=True, exist_ok=True)

        def sample(pool, n):
            if len(pool) <= n:
                return sorted(pool, key=lambda r: -r["score"])
            idx = rng.choice(len(pool), size=n, replace=False)
            return sorted([pool[i] for i in idx], key=lambda r: -r["score"])

        contact_sheet(sample(flagged, args.n), raw_root,
                      f"FLAGGED as illustration (score >= {args.threshold:.2f}) - "
                      f"{len(flagged)} total", sheets_dir / "flagged.jpg")
        contact_sheet(sample(unflagged, args.n), raw_root,
                      f"NOT flagged - kept as real photos (score < {args.threshold:.2f}) - "
                      f"{len(unflagged)} total", sheets_dir / "unflagged.jpg")
        if clipped_rows:
            contact_sheet(sample(clipped_rows, args.n), raw_root,
                          f"CLIPPED / degraded augmentation (clip_frac >= "
                          f"{CLIP_FRAC_DEGRADED}) - {len(clipped_rows)} total - "
                          f"separate decision", sheets_dir / "clipped.jpg")
        borderline = sorted([r for r in rows
                             if label_for(r, args.threshold) != "clipped"
                             and abs(r["score"] - args.threshold) < 0.12],
                            key=lambda r: -r["score"])
        if borderline:
            contact_sheet(borderline[:args.n], raw_root,
                          f"BORDERLINE around {args.threshold:.2f} - these decide the "
                          f"threshold ({len(borderline)} within 0.12)",
                          sheets_dir / "borderline.jpg")
        print(f"\ncontact sheets: {sheets_dir}")

    print(f"\nCSV: {out_dir / 'synthetic_flags.csv'}")
    print("NOTHING has been excluded - a human approves the threshold first.")


if __name__ == "__main__":
    main()
