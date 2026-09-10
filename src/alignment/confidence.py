"""Alignment confidence scoring.

score() maps four signals into a single [0, 1] number plus a breakdown dict:

  inlier_ratio     - fraction of ratio-test matches RANSAC kept; used directly
  n_inliers        - absolute support, saturating at inliers_saturation
  reprojection_err - exp(-err / reproj_halflife), so sub-pixel fits score ~1
  blur             - variance of Laplacian of the SHARPER-limiting image
                     (min of the two visits), saturating at blur_saturation

Each term is in [0, 1] and they are combined with the weights in
configs/config.yaml (alignment.confidence). A rejected homography scores 0.
"""

import cv2
import numpy as np

from src.utils.config import load_config


def blur_metric(img):
    """Variance of the Laplacian - higher means sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pair_blur_metric(img1, img2):
    """The pair is only as sharp as its blurrier image."""
    return min(blur_metric(img1), blur_metric(img2))


def score(inlier_ratio, n_inliers, reprojection_error, blur, cfg=None, rejected=False):
    """Return (score in [0, 1], breakdown dict)."""
    ccfg = (cfg or load_config())["alignment"]["confidence"]

    if rejected:
        breakdown = {"inlier_ratio": 0.0, "n_inliers": 0.0, "reprojection": 0.0,
                     "blur": 0.0, "rejected": True}
        return 0.0, breakdown

    t_ratio = float(np.clip(inlier_ratio, 0.0, 1.0))
    t_inliers = float(np.clip(n_inliers / ccfg["inliers_saturation"], 0.0, 1.0))
    err = float(reprojection_error)
    t_reproj = 0.0 if not np.isfinite(err) else float(np.exp(-max(err, 0.0) / ccfg["reproj_halflife"]))
    t_blur = float(np.clip(blur / ccfg["blur_saturation"], 0.0, 1.0))

    total = (
        ccfg["w_inlier_ratio"] * t_ratio
        + ccfg["w_inliers"] * t_inliers
        + ccfg["w_reproj"] * t_reproj
        + ccfg["w_blur"] * t_blur
    )
    weight_sum = (ccfg["w_inlier_ratio"] + ccfg["w_inliers"]
                  + ccfg["w_reproj"] + ccfg["w_blur"])
    total = float(np.clip(total / weight_sum, 0.0, 1.0))

    breakdown = {
        "inlier_ratio": round(t_ratio, 4),
        "n_inliers": round(t_inliers, 4),
        "reprojection": round(t_reproj, 4),
        "blur": round(t_blur, 4),
        "rejected": False,
        "raw": {
            "inlier_ratio": round(float(inlier_ratio), 4),
            "n_inliers": int(n_inliers),
            "reprojection_error_px": None if not np.isfinite(err) else round(err, 4),
            "blur_var_laplacian": round(float(blur), 2),
        },
    }
    return total, breakdown
