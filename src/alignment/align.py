"""Align a visit pair: warp visit2 into visit1's viewpoint with a confidence score.

data/longitudinal is an INPUT ONLY - every output is written under
data/processed/alignment/pair_XXXX/ (warped.jpg + alignment.json).

CLI:
  python -m src.alignment.align --pair data/longitudinal/pair_0001
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.alignment.confidence import pair_blur_metric, score
from src.alignment.features import detect_and_match
from src.alignment.homography import estimate, warp_visit2
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode, imwrite_jpg


def align_pair(path_visit1, path_visit2, cfg=None):
    """Return dict: warped, H, confidence, breakdown, reason, info."""
    cfg = cfg or load_config()
    img1 = imread_unicode(path_visit1)
    img2 = imread_unicode(path_visit2)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"could not read {path_visit1} or {path_visit2}")

    pts1, pts2, match_info = detect_and_match(img1, img2, cfg)
    H, hom_info = estimate(pts1, pts2, cfg)
    blur = pair_blur_metric(img1, img2)

    rejected = H is None
    conf, breakdown = score(
        hom_info["inlier_ratio"], hom_info["n_inliers"], hom_info["reproj_error"],
        blur, cfg=cfg, rejected=rejected,
    )
    warped = None if rejected else warp_visit2(img2, H, img1.shape)

    return {
        "visit1": str(path_visit1),
        "visit2": str(path_visit2),
        "warped": warped,
        "H": None if rejected else np.asarray(H, float).tolist(),
        "confidence": round(float(conf), 4),
        "breakdown": breakdown,
        "reason": hom_info["reason"],
        "info": {**match_info, **{k: v for k, v in hom_info.items() if k != "reason"}},
        "image_shape": [int(img1.shape[0]), int(img1.shape[1])],
    }


def output_dir_for(pair_dir, cfg=None):
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"]["processed"] / "alignment" / Path(pair_dir).name


def save_result(result, out_dir):
    """Write warped.jpg (when aligned) + alignment.json into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if result["warped"] is not None:
        imwrite_jpg(out_dir / "warped.jpg", result["warped"])
    payload = {k: v for k, v in result.items() if k != "warped"}
    payload["warped_image"] = "warped.jpg" if result["warped"] is not None else None
    (out_dir / "alignment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True,
                        help="path to a data/longitudinal/pair_XXXX folder")
    args = parser.parse_args()

    cfg = load_config()
    pair_dir = Path(args.pair)
    if not pair_dir.is_absolute():
        pair_dir = PROJECT_ROOT / pair_dir

    result = align_pair(pair_dir / "visit1.jpg", pair_dir / "visit2.jpg", cfg)
    out_dir = save_result(result, output_dir_for(pair_dir, cfg))

    if result["reason"]:
        print(f"{pair_dir.name}: REJECTED - {result['reason']} (confidence 0.0)")
    else:
        print(f"{pair_dir.name}: aligned, confidence {result['confidence']:.3f} "
              f"({result['info']['n_inliers']} inliers, "
              f"reproj {result['info']['reproj_error']:.2f}px)")
    print(f"outputs written to {out_dir}")


if __name__ == "__main__":
    main()
