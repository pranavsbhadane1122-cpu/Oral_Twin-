"""Feature detection and matching for visit-pair alignment.

Pipeline: grayscale -> CLAHE (so the two visits' different lighting does not
dominate) -> ORB keypoints restricted to a mouth-region mask -> BFMatcher with
Hamming distance + Lowe ratio test.

Region source
-------------
`alignment.region_source` selects where the ORB mask comes from: the trained
cavity segmenter (`cavity_model`, the default) or the original colour heuristic
(`hsv_heuristic`). The heuristic is always kept as the fallback - if the model
is unavailable or predicts a degenerate near-empty mask, that image silently
degrades to the heuristic rather than to no keypoints at all, and the path
actually taken is recorded in the alignment output.

Mouth-region mask heuristic (the fallback)
-----------------------------------------
Intra-oral photos are dominated by two colour families: teeth (bright, low
saturation, i.e. near-white/grey) and gum/lip/tongue tissue (reddish-pink hues,
which in OpenCV's 0-179 hue scale wrap around 0: roughly H<=20 or H>=160).
The mask is the union of those two HSV bands, cleaned with a morphological
close+open, keeping only the largest connected component (the mouth), then
dilated slightly so keypoints on the lesion/gum border survive. If that
component covers less than mask_min_area_frac of the frame the heuristic is
assumed to have failed and the full frame is used instead - a bad mask is worse
than no mask.

All tunables come from configs/config.yaml (alignment section).
"""

import cv2
import numpy as np

from src.alignment import cavity_region
from src.utils.config import load_config

HSV = "hsv_heuristic"
CAVITY = "cavity_model"
FULL_FRAME = "full_frame"


def _acfg(cfg=None):
    return (cfg or load_config())["alignment"]


def mouth_mask(img_bgr, cfg=None):
    """Binary mask (uint8 0/255) of the dental/oral region. See module docstring."""
    acfg = _acfg(cfg)
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # teeth: bright and weakly saturated
    teeth = cv2.inRange(hsv, (0, 0, 120), (179, 90, 255))
    # tissue: reddish-pink, hue wraps around 0
    tissue_low = cv2.inRange(hsv, (0, 40, 40), (20, 255, 255))
    tissue_high = cv2.inRange(hsv, (160, 40, 40), (179, 255, 255))
    mask = cv2.bitwise_or(teeth, cv2.bitwise_or(tissue_low, tissue_high))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    if mask.sum() / 255.0 < acfg["mask_min_area_frac"] * h * w:
        return np.full((h, w), 255, np.uint8)  # heuristic failed - use whole frame

    return cv2.dilate(mask, k, iterations=2)


def region_mask(img_bgr, cfg=None):
    """(mask, source) for the ORB region of interest.

    source is one of cavity_model | hsv_heuristic | full_frame, and says which
    path was actually taken rather than which was requested - a fallback that
    is not recorded is a fallback nobody notices.
    """
    cfg = cfg or load_config()
    requested = cfg["alignment"].get("region_source", CAVITY)

    if requested == CAVITY:
        mask, reason = cavity_region.cavity_mask(img_bgr, cfg)
        if mask is not None:
            return mask, CAVITY
        # fall through to the heuristic, carrying the reason
        fallback = mouth_mask(img_bgr, cfg)
        source = FULL_FRAME if fallback.all() else HSV
        return fallback, f"{source} (fell back: {reason})"

    if requested != HSV:
        raise ValueError(
            f"unsupported alignment.region_source {requested!r} "
            f"(expected {CAVITY!r} or {HSV!r})")

    mask = mouth_mask(img_bgr, cfg)
    return mask, FULL_FRAME if mask.all() else HSV


def preprocess(img_bgr, cfg=None):
    """Grayscale + CLAHE, for lighting-robust descriptors."""
    acfg = _acfg(cfg)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=acfg["clahe_clip"],
        tileGridSize=(acfg["clahe_grid"], acfg["clahe_grid"]),
    )
    return clahe.apply(gray)


def build_detector(cfg=None):
    acfg = _acfg(cfg)
    name = acfg.get("detector", "orb").lower()
    if name != "orb":
        raise ValueError(f"unsupported detector '{name}' (config alignment.detector)")
    return cv2.ORB_create(nfeatures=acfg["n_features"])


def detect_and_match(img1, img2, cfg=None):
    """Return (pts1, pts2, info).

    pts1/pts2: float32 (N, 2) arrays of matched coordinates in img1 and img2.
    info: dict with keypoint counts, raw match count and the ratio-test survivors.
    """
    cfg = cfg or load_config()
    acfg = cfg["alignment"]
    detector = build_detector(cfg)

    g1, g2 = preprocess(img1, cfg), preprocess(img2, cfg)
    m1, src1 = region_mask(img1, cfg)
    m2, src2 = region_mask(img2, cfg)
    kp1, des1 = detector.detectAndCompute(g1, m1)
    kp2, des2 = detector.detectAndCompute(g2, m2)

    info = {"n_kp1": len(kp1), "n_kp2": len(kp2), "n_raw_matches": 0,
            "n_good_matches": 0,
            "region_source_requested": acfg.get("region_source", CAVITY),
            "region_source_visit1": src1, "region_source_visit2": src2,
            "region_frac_visit1": round(float(m1.sum() / 255.0 / m1.size), 4),
            "region_frac_visit2": round(float(m2.sum() / 255.0 / m2.size), 4)}
    empty = np.zeros((0, 2), np.float32)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return empty, empty, info

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    info["n_raw_matches"] = len(knn)

    ratio = acfg["ratio_test"]
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    info["n_good_matches"] = len(good)
    if not good:
        return empty, empty, info

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, info
