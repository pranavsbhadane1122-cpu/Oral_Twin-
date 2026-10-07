"""Are the cavity model's masks usable on MIO images for a 'how much is mouth' figure?

The model scored Dice 0.933 on Piyarathne and we have just measured it making
alignment worse on MIO-derived pairs. That told us it is a poorer ORB region;
it did NOT tell us whether the mask is wrong. Those are different claims - a
mask can be anatomically right and still a bad keypoint region.

This renders both masks side by side so the question can be answered by looking.
Green = cavity model, yellow = HSV heuristic.

Output: data/processed/cavity_spotcheck/ (git-ignored, derivative images)
Usage: python scripts/spotcheck_cavity_mio.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment import cavity_region  # noqa: E402
from src.alignment.features import mouth_mask  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imread_unicode  # noqa: E402

N = 10
SEED = 20251007


def outline(canvas, mask, colour, thickness=2):
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, colour, thickness)


def main():
    cfg = load_config()
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "cavity_spotcheck"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = sorted(p for p in long_root.glob("pair_*") if p.is_dir())
    rng = np.random.default_rng(SEED)
    chosen = [pairs[i] for i in rng.choice(len(pairs), N, replace=False)]
    chosen.sort(key=lambda p: p.name)

    rows = []
    for pair in chosen:
        img = imread_unicode(pair / "visit1.jpg")
        if img is None:
            continue
        gt = json.loads((pair / "ground_truth.json").read_text(encoding="utf-8"))
        cls = gt["source_class"]

        cav = cavity_region.predict_cavity(img, cfg)
        heur = mouth_mask(img, cfg)
        h, w = img.shape[:2]
        cav_frac = float((cav > 127).sum()) / (h * w) if cav is not None else float("nan")
        heur_frac = float((heur > 127).sum()) / (h * w)

        canvas = img.copy()
        if cav is not None:
            outline(canvas, cav, (0, 255, 0), 2)        # green  = model
        outline(canvas, heur, (0, 255, 255), 1)         # yellow = heuristic
        label = f"{pair.name} [{cls}] model {cav_frac:.0%} / heuristic {heur_frac:.0%}"
        cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{pair.name}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append((pair.name, cls, cav_frac, heur_frac))

    print(f"{'pair':<12}{'class':<12}{'model':>8}{'heuristic':>11}")
    for name, cls, c, h in rows:
        print(f"{name:<12}{cls:<12}{c:>7.1%}{h:>10.1%}")
    fr = np.array([r[2] for r in rows])
    hr = np.array([r[3] for r in rows])
    print(f"\nmean model {fr.mean():.1%}  mean heuristic {hr.mean():.1%}")
    print(f"model range {fr.min():.1%} - {fr.max():.1%}")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
