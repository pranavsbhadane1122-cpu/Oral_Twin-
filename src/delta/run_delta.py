"""Run change detection over the longitudinal pairs.

For each pair: align visit2 to visit1 with the Phase 4 module, warp visit2's
lesion mask with the SAME estimated homography, then compare against visit1's
mask. Reports go to data/processed/delta/pair_XXXX/change_report.json.

data/longitudinal and data/processed/delta_masks are inputs only.

Usage:
  python -m src.delta.run_delta                 # all pairs
  python -m src.delta.run_delta --pair pair_0007
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.alignment.align import align_pair
from src.alignment.homography import warp_visit2
from src.delta.compare import compare
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode


def paths_for(pair_name, cfg):
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    return {
        "pair": PROJECT_ROOT / cfg["paths"]["longitudinal"] / pair_name,
        "masks": processed / "delta_masks" / pair_name,
        "out": processed / "delta" / pair_name,
    }


def run_pair(pair_name, cfg=None):
    """Align, warp mask2, compare. Returns (report, extras) where extras carries
    the warped image/mask for debug rendering."""
    cfg = cfg or load_config()
    p = paths_for(pair_name, cfg)

    img1 = imread_unicode(p["pair"] / "visit1.jpg")
    img2 = imread_unicode(p["pair"] / "visit2.jpg")
    mask1 = cv2.imread(str(p["masks"] / "mask1.png"), cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(str(p["masks"] / "mask2.png"), cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in (img1, img2, mask1, mask2)):
        raise FileNotFoundError(f"missing inputs for {pair_name} - "
                                f"run longitudinal_sim and synthetic_masks first")

    alignment = align_pair(p["pair"] / "visit1.jpg", p["pair"] / "visit2.jpg", cfg)
    confidence = alignment["confidence"]

    if alignment["H"] is None:
        report = compare(mask1, np.zeros_like(mask1), img1, img1, 0.0, cfg)
        report["reason"] = f"alignment failed: {alignment['reason']}"
        report["status"] = "unreliable"
        report["summary"] = {"comparable": False}
        extras = {"img2_warped": None, "mask2_warped": None}
    else:
        H = np.asarray(alignment["H"], float)
        img2_warped = warp_visit2(img2, H, img1.shape)
        mask2_warped = cv2.warpPerspective(mask2, H, (img1.shape[1], img1.shape[0]),
                                           flags=cv2.INTER_NEAREST)
        report = compare(mask1, mask2_warped, img1, img2_warped, confidence, cfg)
        extras = {"img2_warped": img2_warped, "mask2_warped": mask2_warped}

    report["pair"] = pair_name
    report["alignment"] = {
        "confidence": confidence,
        "reason": alignment["reason"],
        "n_inliers": alignment["info"]["n_inliers"],
        "H": alignment["H"],
    }
    extras.update({"img1": img1, "mask1": mask1})
    return report, extras


def save_report(report, cfg=None):
    cfg = cfg or load_config()
    out = paths_for(report["pair"], cfg)["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "change_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", help="single pair name, e.g. pair_0007")
    args = parser.parse_args()

    cfg = load_config()
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    names = ([args.pair] if args.pair
             else [p.name for p in sorted(long_root.glob("pair_*")) if p.is_dir()])

    n_notable = n_unreliable = 0
    for i, name in enumerate(names, 1):
        report, _ = run_pair(name, cfg)
        save_report(report, cfg)
        if report["status"] == "unreliable":
            n_unreliable += 1
        elif report["summary"].get("any_notable_change"):
            n_notable += 1
        if args.pair or i % 25 == 0:
            print(f"  {i}/{len(names)} {name}: {report['summary'].get('headline', report['reason'])}")

    out_root = PROJECT_ROOT / cfg["paths"]["processed"] / "delta"
    print(f"\nprocessed {len(names)} pairs -> {out_root}")
    print(f"  notable change reported: {n_notable}")
    print(f"  refused as unreliable:   {n_unreliable}")


if __name__ == "__main__":
    main()
