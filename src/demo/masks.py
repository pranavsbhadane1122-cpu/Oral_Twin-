"""Where the demo's lesion masks come from.

The Phase 3 segmenter does not exist yet (its training data is blocked), so the
demo needs a mask source it can be honest about. Two sources exist, and each
carries a label the UI is required to show:

  SYNTHETIC    the simulated masks built from the longitudinal ground truth in
               Phase 5. Real for the sample pairs, meaningless for anything else.
  PLACEHOLDER  a classical region finder for uploaded photographs: the pixels
               whose colour sits furthest from the photograph's dominant tissue
               colour. It is arithmetic on colour, not a trained segmenter and
               not a detector of disease.

Nothing here may be presented as a model output.
"""

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config

SYNTHETIC = "synthetic"
PLACEHOLDER = "placeholder"
NONE = "none"

LABELS = {
    SYNTHETIC: ("Simulated masks from the sample pair's ground truth — "
                "not a trained segmenter"),
    PLACEHOLDER: ("PLACEHOLDER region finder: colour-deviation arithmetic, "
                  "not a trained segmenter and not a detection of disease"),
    NONE: "No region available to compare",
}


def synthetic_masks_path(pair_name, cfg=None):
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"]["processed"] / "delta_masks" / pair_name


def load_synthetic_masks(pair_name, cfg=None):
    """(mask1 in visit1 coords, mask2 in visit2 coords) or None if absent."""
    d = synthetic_masks_path(pair_name, cfg)
    m1, m2 = d / "mask1.png", d / "mask2.png"
    if not (m1.exists() and m2.exists()):
        return None
    return (cv2.imread(str(m1), cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(m2), cv2.IMREAD_GRAYSCALE))


def placeholder_mask(img_bgr, cfg=None, valid_mask=None):
    """Candidate region = pixels furthest from the photograph's dominant colour.

    Deliberately simple and deliberately labelled. Lab space is used so the
    distance is roughly perceptual; the threshold is a percentile, so the
    region's size is stable across photographs instead of depending on an
    absolute colour value that lighting would move.
    """
    cfg = cfg or load_config()
    pcfg = cfg["demo"]["placeholder_mask"]
    min_area = cfg["delta"]["min_lesion_area_px"]

    blurred = cv2.GaussianBlur(img_bgr, (0, 0), float(pcfg["blur_sigma"]))
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = np.ones(lab.shape[:2], bool) if valid_mask is None else (valid_mask > 127)
    if not sel.any():
        return np.zeros(lab.shape[:2], np.uint8)

    dominant = np.median(lab[sel].reshape(-1, 3), axis=0)
    deviation = np.linalg.norm(lab - dominant, axis=2)
    deviation[~sel] = 0.0

    cutoff = float(np.percentile(deviation[sel], float(pcfg["deviation_percentile"])))
    mask = ((deviation >= cutoff) & sel).astype(np.uint8) * 255

    k = int(pcfg["morph_kernel"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = sorted(
        [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
         if stats[i, cv2.CC_STAT_AREA] >= min_area],
        reverse=True,
    )[: int(pcfg["max_regions"])]

    out = np.zeros_like(mask)
    for _, i in keep:
        out[labels == i] = 255
    return out
