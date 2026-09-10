"""RANSAC homography estimation with plausibility checks.

estimate() maps points in image2 onto image1 (i.e. the returned H warps visit2
into visit1's viewpoint). Degenerate or implausible results are rejected with a
human-readable reason instead of silently returning garbage.

Rejection criteria (thresholds from configs/config.yaml alignment section):
  - too few matched points to even attempt a fit
  - findHomography returned nothing
  - fewer than min_inliers RANSAC inliers
  - |det(H)| below min_abs_det (near-singular)
  - implied scale outside [min_scale, max_scale]
  - perspective terms |h31|/|h32| above max_perspective (implausible tilt for
    two phone photos of the same mouth)
"""

import cv2
import numpy as np

from src.utils.config import load_config


def normalize_h(H):
    """Scale H so h33 = 1, making entries comparable across estimates."""
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) < 1e-12:
        return H
    return H / H[2, 2]


def check_plausible(H, cfg=None):
    """Return None if H looks physically plausible, else a reason string."""
    acfg = (cfg or load_config())["alignment"]
    Hn = normalize_h(H)
    det = float(np.linalg.det(Hn))
    if not np.isfinite(det) or abs(det) < acfg["min_abs_det"]:
        return f"near-singular homography (|det|={abs(det):.2e})"

    affine_det = float(np.linalg.det(Hn[:2, :2]))
    if not np.isfinite(affine_det) or affine_det <= 0:
        return f"non-positive affine determinant ({affine_det:.3e}) - mirrored/degenerate"
    scale = float(np.sqrt(abs(affine_det)))
    if scale < acfg["min_scale"] or scale > acfg["max_scale"]:
        return f"implausible scale {scale:.3f} (allowed {acfg['min_scale']}-{acfg['max_scale']})"

    persp = max(abs(float(Hn[2, 0])), abs(float(Hn[2, 1])))
    if persp > acfg["max_perspective"]:
        return f"heavy perspective term {persp:.5f} (max {acfg['max_perspective']})"
    return None


def estimate(points1, points2, cfg=None):
    """Fit H mapping points2 -> points1.

    Returns (H, info). H is None when rejected; info always carries
    'reason' (None on success), 'n_inliers', 'inlier_ratio', 'reproj_error'.
    """
    cfg = cfg or load_config()
    acfg = cfg["alignment"]
    info = {"reason": None, "n_inliers": 0, "inlier_ratio": 0.0,
            "reproj_error": float("inf"), "n_matches": int(len(points1))}

    if len(points1) < 4 or len(points2) < 4:
        info["reason"] = f"too few matches ({len(points1)}) to fit a homography"
        return None, info

    H, mask = cv2.findHomography(
        np.asarray(points2, np.float32), np.asarray(points1, np.float32),
        cv2.RANSAC, acfg["ransac_reproj_thresh"],
    )
    if H is None:
        info["reason"] = "findHomography failed to converge"
        return None, info

    mask = mask.ravel().astype(bool)
    n_in = int(mask.sum())
    info["n_inliers"] = n_in
    info["inlier_ratio"] = float(n_in / max(len(points1), 1))

    if n_in:
        src = np.asarray(points2, np.float32)[mask].reshape(-1, 1, 2)
        dst = np.asarray(points1, np.float32)[mask].reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(src, H)
        info["reproj_error"] = float(np.linalg.norm(proj - dst, axis=2).mean())

    if n_in < acfg["min_inliers"]:
        info["reason"] = f"only {n_in} inliers (min {acfg['min_inliers']})"
        return None, info

    reason = check_plausible(H, cfg)
    if reason:
        info["reason"] = reason
        return None, info

    return normalize_h(H), info


def warp_visit2(img2, H, target_shape):
    """Warp visit2 into visit1's frame. target_shape is (h, w[, c])."""
    h, w = target_shape[:2]
    return cv2.warpPerspective(img2, np.asarray(H, np.float64), (w, h))


def corner_error(H_est, H_true, width, height):
    """Mean L2 distance (px) between the image corners projected through both
    homographies. Scale-invariant: both matrices are normalized first, and
    projection divides by the homogeneous coordinate anyway."""
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    ).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, normalize_h(H_est))
    b = cv2.perspectiveTransform(corners, normalize_h(H_true))
    return float(np.linalg.norm(a - b, axis=2).mean())
