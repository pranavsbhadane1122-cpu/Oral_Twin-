"""Describe a single photograph from measurable properties only.

What this module may say: whether an intraoral view is present, how much of the
frame it occupies (only when that can be measured reliably), whether the photo
is in focus, and whether the mouth runs off the edge of the frame.

What it may NEVER say: any disease name, any severity, any lesion count. The
classifier is parked after failing its Grad-CAM gate, and the lesion segmenter
failed its Dice gate. There is no validated model behind any clinical claim, so
the caption layer does not make one. tests/test_caption.py enforces this with a
regex over a condition vocabulary rather than leaving it to good intentions.

Mouth extent is deliberately awkward
------------------------------------
Two independent estimators disagree about it, each failing in its own
direction - see models/logs/cavity_spotcheck_verdict.txt. The cavity model
reports 22-40% on extreme close-ups where the whole frame is already mouth; the
HSV heuristic silently reports 100% when its own mask looks implausible. So the
fraction is printed only when both agree. When they do not, this module says the
extent could not be measured, which is the truthful answer, rather than picking
whichever number sounds plausible.
"""

import cv2
import numpy as np

from src.alignment import cavity_region
from src.alignment.confidence import blur_metric
from src.alignment.features import mouth_mask
from src.utils.config import load_config

# focus bands
POOR, MARGINAL, USABLE = "poor", "marginal", "usable"

VIEW_INTRAORAL = "intraoral"
VIEW_PARTIAL = "partial"
VIEW_NONE = "none"
VIEW_UNCERTAIN = "uncertain"


def _frac(mask):
    if mask is None:
        return None
    return float((mask > 127).sum()) / float(mask.shape[0] * mask.shape[1])


def mouth_extent(img_bgr, cfg=None):
    """Both estimates of the mouth's share of the frame, and whether to trust them.

    Returns a dict with cavity_model, hsv_heuristic, agreed (bool), fraction
    (float or None) and reason (str or None). fraction is None exactly when the
    two sources cannot corroborate each other.
    """
    cfg = cfg or load_config()
    ccfg = cfg["caption"]

    model_frac = _frac(cavity_region.predict_cavity(img_bgr, cfg))
    heuristic_frac = _frac(mouth_mask(img_bgr, cfg))

    out = {"cavity_model": model_frac, "hsv_heuristic": heuristic_frac,
           "agreed": False, "fraction": None, "reason": None}

    if model_frac is None:
        out["reason"] = "the cavity model is unavailable, so only one estimate exists"
        return out
    if heuristic_frac is not None and heuristic_frac >= ccfg["heuristic_fullframe_frac"]:
        # the heuristic failed open and returned the whole frame; it is not a
        # measurement and must not be averaged with one
        out["reason"] = ("the colour heuristic could not find a mouth region and "
                         "fell back to the whole frame, so it cannot corroborate")
        return out

    if abs(model_frac - heuristic_frac) <= ccfg["mouth_fraction_tolerance"]:
        out["agreed"] = True
        out["fraction"] = (model_frac + heuristic_frac) / 2.0
    else:
        out["reason"] = (f"the two estimates disagree "
                         f"({model_frac:.0%} and {heuristic_frac:.0%})")
    return out


def view_type(extent, cfg=None):
    """intraoral | partial | none | uncertain, from the two extent estimates.

    This is a much easier question than the fraction and both sources answer it
    correctly on all ten spot-check images, so it is stated with confidence
    where the fraction is not.
    """
    cfg = cfg or load_config()
    floor = cfg["caption"]["intraoral_min_frac"]
    model, heuristic = extent["cavity_model"], extent["hsv_heuristic"]
    present = [v for v in (model, heuristic) if v is not None]
    if not present:
        return VIEW_UNCERTAIN
    if all(v < floor for v in present):
        return VIEW_NONE
    if extent["fraction"] is not None and extent["fraction"] >= 0.80:
        return VIEW_INTRAORAL
    if max(present) >= floor:
        return VIEW_PARTIAL
    return VIEW_UNCERTAIN


def focus_band(img_bgr, cfg=None):
    """(band, variance). Bands, because a raw Laplacian variance means nothing
    to the person holding the phone."""
    cfg = cfg or load_config()
    fcfg = cfg["caption"]["focus"]
    variance = float(blur_metric(img_bgr))
    if variance < fcfg["poor"]:
        return POOR, variance
    if variance < fcfg["marginal"]:
        return MARGINAL, variance
    return USABLE, variance


def region_clipped(img_bgr, cfg=None):
    """(clipped, share) - does the mouth region run off the edge of the frame?

    Measured on the heuristic mask, which is the more inclusive of the two and
    therefore the conservative choice for an "is anything cut off" question.
    Returns (None, None) when there is no usable region to test.
    """
    cfg = cfg or load_config()
    ccfg = cfg["caption"]
    mask = mouth_mask(img_bgr, cfg)
    binary = (mask > 127).astype(np.uint8)
    h, w = binary.shape
    if binary.all() or not binary.any():
        return None, None          # full-frame fallback or empty: nothing to say

    margin = ccfg["edge_margin_px"]
    border = np.zeros_like(binary)
    border[:margin, :] = 1
    border[-margin:, :] = 1
    border[:, :margin] = 1
    border[:, -margin:] = 1

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    outline = np.zeros_like(binary)
    cv2.drawContours(outline, contours, -1, 1, 1)
    perimeter = int(outline.sum())
    if perimeter == 0:
        return None, None
    on_border = int((outline & border).sum())
    share = on_border / perimeter
    return bool(share >= ccfg["clipped_frac"]), float(share)


def describe(img_bgr, cfg=None):
    """Structured description of one photograph. No clinical content."""
    cfg = cfg or load_config()
    extent = mouth_extent(img_bgr, cfg)
    view = view_type(extent, cfg)
    band, variance = focus_band(img_bgr, cfg)
    clipped, clipped_share = region_clipped(img_bgr, cfg)

    usable = view != VIEW_NONE and band != POOR
    return {
        "view": view,
        "mouth_extent": extent,
        "focus": {"band": band, "variance_of_laplacian": round(variance, 1)},
        "framing": {"clipped": clipped,
                    "border_share": None if clipped_share is None
                    else round(clipped_share, 3)},
        "usable": usable,
        "shape": [int(img_bgr.shape[0]), int(img_bgr.shape[1])],
    }


# --------------------------------------------------------------------------
# sentences
# --------------------------------------------------------------------------

VIEW_SENTENCE = {
    VIEW_INTRAORAL: "This photograph shows the inside of the mouth filling most of the frame.",
    VIEW_PARTIAL: "This photograph shows part of the mouth.",
    VIEW_NONE: ("No clear view of the inside of the mouth could be found in this "
                "photograph."),
    VIEW_UNCERTAIN: "Whether this photograph shows the inside of the mouth is unclear.",
}

FOCUS_SENTENCE = {
    USABLE: "It is in focus.",
    MARGINAL: "It is slightly soft, but still readable.",
    POOR: "It is out of focus. Please retake it in better light, holding still.",
}


def sentences(description):
    """Plain-language lines for one photograph. Describes, never diagnoses."""
    lines = [VIEW_SENTENCE[description["view"]]]

    extent = description["mouth_extent"]
    if extent["fraction"] is not None:
        lines.append(f"The mouth takes up roughly {extent['fraction']:.0%} of the frame.")
    elif description["view"] != VIEW_NONE:
        lines.append("How much of the frame the mouth takes up could not be measured "
                     f"reliably here: {extent['reason']}.")

    lines.append(FOCUS_SENTENCE[description["focus"]["band"]])

    if description["framing"]["clipped"]:
        lines.append("Part of the mouth runs off the edge of the picture. If you can, "
                     "retake it with the whole area inside the frame.")
    return lines
