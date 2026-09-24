"""The demo pipeline: two photographs in, a change report and a risk band out.

This is the function behind the Streamlit UI, kept separate from it so it can be
tested without a browser. It runs only the modules that have been validated:

    align (Phase 4)  ->  confidence  ->  masks  ->  delta (Phase 5)  ->  risk

Refusals are first-class results, not errors. When the two photographs cannot be
lined up, or a region sits partly outside the newer photograph, the pipeline
returns a refusal with a reason and NO measurement. That behaviour is the point
of the system and the UI is required to show it.

The classifier is parked: while classification.surface_predictions is false,
nothing here passes a prediction to the report or the risk rules.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.delta.compare import compare, overlap_valid_mask
from src.demo import masks as mask_src
from src.risk.report import build_report
from src.risk.risk_rules import stratify
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode

NOT_INCLUDED = [
    "Disease classification (caries / calculus / gingivitis) - the model is parked: "
    "it learned image provenance rather than pathology, so it names nothing here.",
    "Lesion segmentation - no trained segmenter yet; regions shown for uploads come "
    "from a labelled placeholder, and sample pairs use simulated masks.",
    "Oral-cancer / OPMD detection - blocked pending access to an external dataset.",
    "Validation on real repeat photographs - change detection was validated on "
    "simulated visit pairs with known ground truth.",
]

SCREENING_LINE = ("OralTwin is a screening aid, not a diagnostic tool. "
                  "It cannot diagnose anything. Have any finding checked by a dentist.")


def _as_path(image, stack):
    """Accept a path or a BGR array; arrays are written to a temp file so the
    validated align_pair() path is used unchanged."""
    if isinstance(image, (str, Path)):
        return str(image)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, image)
    stack.append(tmp.name)
    return tmp.name


def sample_pair_paths(pair_name, cfg=None):
    cfg = cfg or load_config()
    d = PROJECT_ROOT / cfg["paths"]["longitudinal"] / pair_name
    return d / "visit1.jpg", d / "visit2.jpg"


def first_test_image(class_name, cfg=None):
    """Deterministic pick: the first test-split photograph of a class."""
    cfg = cfg or load_config()
    d = PROJECT_ROOT / cfg["paths"]["processed"] / "classification" / "test" / class_name
    files = sorted(d.glob("*.jpg"))
    return files[0] if files else None


def demo_cases(cfg=None):
    """The four cases the demo must be able to show, resolved to file paths."""
    cfg = cfg or load_config()
    d = cfg["demo"]
    cls_a, cls_b = d["low_confidence_classes"]
    return {
        "good": {
            "title": "Sample pair - a clear change",
            "pair_name": d["sample_pairs"]["good"],
            "paths": sample_pair_paths(d["sample_pairs"]["good"], cfg),
            "note": "Two simulated visits of the same mouth, well aligned.",
        },
        "out_of_frame": {
            "title": "Sample pair - region partly out of frame",
            "pair_name": d["sample_pairs"]["out_of_frame"],
            "paths": sample_pair_paths(d["sample_pairs"]["out_of_frame"], cfg),
            "note": "The area of interest is not fully inside the newer photograph.",
        },
        "low_confidence": {
            "title": "Two different mouths",
            "pair_name": None,
            "paths": (first_test_image(cls_a, cfg), first_test_image(cls_b, cfg)),
            "note": "Photographs of different mouths: there is no real alignment to find.",
        },
        "unrelated": {
            "title": "One photograph is not a mouth",
            "pair_name": None,
            "paths": (first_test_image(cls_a, cfg), PROJECT_ROOT / d["unrelated_asset"]),
            "note": "Negative control: the second photograph is not an oral photograph.",
        },
    }


def run_demo(visit1, visit2, pair_name=None, cfg=None):
    """Run the validated pipeline over two photographs. Refusals are results."""
    cfg = cfg or load_config()
    from src.alignment.align import align_pair

    temps = []
    try:
        p1, p2 = _as_path(visit1, temps), _as_path(visit2, temps)
        img1, img2 = imread_unicode(p1), imread_unicode(p2)
        if img1 is None or img2 is None:
            raise ValueError("one of the two images could not be read")
        alignment = align_pair(p1, p2, cfg)
    finally:
        for t in temps:
            Path(t).unlink(missing_ok=True)

    confidence = float(alignment["confidence"])
    H = alignment["H"]
    result = {
        "status": "refused",
        "reason": None,
        "pair_name": pair_name or "uploaded photos",
        "alignment": {
            "confidence": confidence,
            "breakdown": alignment["breakdown"],
            "reason": alignment["reason"],
            "n_inliers": int(alignment["info"].get("n_inliers", 0)),
            "H": H,
        },
        "images": {"visit1": img1, "visit2": img2, "visit2_warped": None, "overlay": None},
        "masks": {"source": mask_src.NONE, "label": mask_src.LABELS[mask_src.NONE],
                  "mask1": None, "mask2_warped": None},
        "not_included": list(NOT_INCLUDED),
        "screening_line": SCREENING_LINE,
    }

    if H is None:
        blank = np.zeros(img1.shape[:2], np.uint8)
        change_report = compare(blank, blank, img1, img1, 0.0, cfg)
        change_report["status"] = "unreliable"
        change_report["reason"] = (
            "the two photographs could not be lined up ("
            + str(alignment["reason"]) + "). Retake the newer photo with similar "
            "framing and lighting."
        )
        change_report["summary"] = {"comparable": False}
    else:
        Hm = np.asarray(H, float)
        warped = alignment["warped"]
        valid = overlap_valid_mask(img1.shape, Hm, img2.shape)
        result["images"]["visit2_warped"] = warped
        result["images"]["overlay"] = cv2.addWeighted(img1, 0.5, warped, 0.5, 0.0)

        loaded = mask_src.load_synthetic_masks(pair_name, cfg) if pair_name else None
        if loaded is not None:
            mask1, mask2 = loaded
            mask2_warped = cv2.warpPerspective(mask2, Hm, (img1.shape[1], img1.shape[0]),
                                               flags=cv2.INTER_NEAREST)
            source = mask_src.SYNTHETIC
        else:
            mask1 = mask_src.placeholder_mask(img1, cfg)
            mask2_warped = mask_src.placeholder_mask(warped, cfg, valid_mask=valid)
            source = mask_src.PLACEHOLDER

        result["masks"] = {"source": source, "label": mask_src.LABELS[source],
                           "mask1": mask1, "mask2_warped": mask2_warped}
        change_report = compare(mask1, mask2_warped, img1, warped, confidence, cfg,
                                valid_mask=valid)

    change_report["pair"] = result["pair_name"]
    risk = stratify(change_report, None, None, confidence, cfg)
    result["change_report"] = change_report
    result["risk"] = risk
    result["status"] = "ok" if change_report.get("status") == "ok" else "refused"
    result["reason"] = change_report.get("reason")
    result["report_text"] = build_report(
        result["pair_name"], change_report, risk, None, None, {},
        {"reason": alignment["reason"]}, cfg=cfg,
    )
    return result
