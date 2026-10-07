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
    "Naming a condition. The classifier that would do this is parked. On its own "
    "held-out photographs it correctly flagged only about 49 of every 100 cases it "
    "should have flagged; on photographs from a different clinic that fell to about "
    "7 in every 100. It had also learned to tell images apart by where they came "
    "from rather than by what was in them. This demo names no condition at all.",

    "Outlining an area of concern. The lesion segmentation model failed its "
    "acceptance test. Rather than finding the area of concern, it learned to draw "
    "around the inside of the mouth: its outlines matched the mouth more closely "
    "than they matched the area they were supposed to find, and on photographs of "
    "healthy mouths it still painted an outline over roughly a sixth of the "
    "picture. Nothing shown here is an area found by a model.",

    "Measuring how much of the picture is mouth. The oral-cavity outline is "
    "accurate on the data it was trained on, but measurably worse on photographs "
    "from elsewhere, so it is never trusted on its own. It is used only as one half "
    "of an agreement check against a second, independent estimate. When the two "
    "disagree, the figure is withheld and the app says so instead of picking one.",

    "Change detection on real repeat photographs. The change figures here are "
    "measured on simulated areas with known answers, not on areas produced by any "
    "model, and on simulated visit pairs rather than two real visits by one person.",

    "OPMD detection - blocked pending access to an external dataset.",
]

SIMULATED = ("Simulated visit pair: one real photograph, warped by a known "
             "transform, with a simulated area of interest. Not two real visits.")
CONSTRUCTED = ("Constructed input: these two photographs were deliberately paired to "
               "show what the app does when a comparison cannot be trusted.")
INDUCED_NOTE = (
    "INDUCED, not natural. These two photographs line up perfectly well. The "
    "reliability threshold was raised to {floor} - above this pair's real alignment "
    "score - purely to show the refusal. No pair in the sample set fails alignment "
    "on its own: 200 of 200 succeed, so this path has no natural example and is "
    "shown here by changing a setting rather than by finding a genuine failure."
)

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
            "provenance": SIMULATED,
            "induced": None,
        },
        "out_of_frame": {
            "title": "Sample pair - region partly out of frame",
            "pair_name": d["sample_pairs"]["out_of_frame"],
            "paths": sample_pair_paths(d["sample_pairs"]["out_of_frame"], cfg),
            "note": "The area of interest is not fully inside the newer photograph.",
            "provenance": SIMULATED,
            "induced": None,
        },
        "unusable": {
            "title": "Sample pair - the photograph is not good enough",
            "pair_name": d["sample_pairs"]["unusable"],
            "paths": sample_pair_paths(d["sample_pairs"]["unusable"], cfg),
            "note": "This photograph is genuinely out of focus. Nothing was changed "
                    "to make it so.",
            "provenance": SIMULATED,
            "induced": None,
        },
        "low_confidence": {
            "title": "Two different mouths",
            "pair_name": None,
            "paths": (first_test_image(cls_a, cfg), first_test_image(cls_b, cfg)),
            "note": "Photographs of different mouths: there is no real alignment to find.",
            "provenance": CONSTRUCTED,
            "induced": None,
        },
        "induced_low_confidence": {
            "title": "Reliability threshold raised (induced)",
            "pair_name": d["sample_pairs"]["induced_low_confidence"],
            "paths": sample_pair_paths(d["sample_pairs"]["induced_low_confidence"], cfg),
            "note": "What the app does when a comparison is judged too unreliable "
                    "to report.",
            "provenance": CONSTRUCTED,
            "induced": INDUCED_NOTE.format(floor=d["induced_confidence_floor"]),
            "cfg_overrides": {"delta": {"confidence_floor":
                                        d["induced_confidence_floor"]}},
        },
        "unrelated": {
            "title": "One photograph is not a mouth",
            "pair_name": None,
            "paths": (first_test_image(cls_a, cfg), PROJECT_ROOT / d["unrelated_asset"]),
            "note": "Negative control: the second photograph is not an oral photograph.",
            "provenance": CONSTRUCTED,
            "induced": None,
        },
    }


def lesion_table_rows(lesions):
    """Rows for the UI table.

    A region that is not comparable shows no figures at all - not even its raw
    pixel areas. Printing 755 and 1004 next to "partly out of frame" invites the
    reader to divide them and conclude the spot grew by a third, which is exactly
    the inference the comparability label exists to prevent.
    """
    from src.delta.compare import FULL, PARTIAL

    rows = []
    for les in lesions:
        comparable = les["comparability"] == FULL
        if les["comparability"] == PARTIAL:
            label = "partly out of frame"
        elif comparable:
            label = "comparable"
        else:
            label = "low confidence"
        if comparable:
            change = (format(les["area_change_pct"], "+.0f") + "%"
                      if les["area_change_pct"] is not None else "-")
            area1 = les["area_visit1_px"] or "-"
            area2 = les["area_visit2_px"] or "-"
            colour = "notable" if (les.get("color_shift") or {}).get("notable") else "no"
        else:
            change = area1 = area2 = colour = "-"
        rows.append({
            "region": les["id"],
            "state": les["state"],
            "comparability": label,
            "area, previous visit": area1,
            "area, this visit": area2,
            "area change": change,
            "colour change": colour,
        })
    return rows


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

    # The caption layer: describes the photographs, never their condition.
    # Both visits are described, because "is this one usable" is a per-photo
    # question and the user took two.
    from src.caption import compare as caption_compare
    from src.caption import describe as caption_describe
    from src.caption.render import LIMITATIONS

    result["caption"] = {
        "visit1": caption_describe.describe(img1, cfg),
        "visit2": caption_describe.describe(img2, cfg),
        "comparison": caption_compare.summarise(change_report, cfg),
        "limitations": LIMITATIONS,
    }
    result["caption"]["lines"] = {
        "visit1": caption_describe.sentences(result["caption"]["visit1"]),
        "visit2": caption_describe.sentences(result["caption"]["visit2"]),
        "comparison": caption_compare.sentences(result["caption"]["comparison"]),
    }
    return result
