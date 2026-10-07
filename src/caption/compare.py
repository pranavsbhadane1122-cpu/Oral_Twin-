"""Describe what changed between two visits, from the existing delta engine.

This module adds no measurement of its own. It translates the change report
that src/delta already produces into plain language, and - more importantly -
it refuses in plain language when the delta engine says the comparison is not
reliable. A refusal that reads like a result is the failure mode worth guarding
against here: the user must be able to tell "nothing changed" apart from "this
could not be checked".

No disease names. The delta engine measures area, colour and edge regularity of
regions that were marked in both photographs; it does not know what any of them
is, and neither does this module.
"""

from src.utils.config import load_config

RELIABLE, UNRELIABLE, FAILED = "reliable", "unreliable", "failed"

RETAKE = ("Please retake the photograph in similar lighting and framing to the "
          "last one, and try again.")


def _confidence_word(confidence):
    if confidence >= 0.65:
        return "a good match"
    if confidence >= 0.45:
        return "a moderate match"
    return "a weak match"


def summarise(change_report, cfg=None):
    """Structured comparison summary. Mirrors the delta report, adds no claims."""
    cfg = cfg or load_config()
    status = change_report.get("status", "ok")
    confidence = float(change_report.get("alignment_confidence", 0.0))
    summary = change_report.get("summary", {}) or {}

    if status != "ok" or not summary.get("comparable", False):
        state = FAILED if change_report.get("reason") else UNRELIABLE
        return {
            "state": state,
            "confidence": confidence,
            "reason": change_report.get("reason")
                      or "the two photographs could not be lined up reliably",
            "regions": [],
            "excluded": [],
        }

    regions, excluded = [], []
    for lesion in change_report.get("lesions", []):
        entry = {
            "id": lesion.get("id"),
            "comparability": lesion.get("comparability"),
            "state": lesion.get("state"),
            "confidence": lesion.get("confidence"),
            "reliability": lesion.get("reliability"),
        }
        if lesion.get("comparability") == "full":
            entry["area_change_pct"] = lesion.get("area_change_pct")
            entry["notable"] = bool(lesion.get("notable_area_change")
                                    or (lesion.get("color_shift") or {}).get("notable")
                                    or (lesion.get("edge_irregularity") or {}).get("notable"))
            regions.append(entry)
        else:
            # keep both: comparability is the stable code the wording keys on,
            # reason is the engine's own free text and is the better fallback
            entry["reason"] = lesion.get("reason")
            excluded.append(entry)

    return {
        "state": RELIABLE,
        "confidence": confidence,
        "reason": None,
        "regions": regions,
        "excluded": excluded,
        "any_notable_change": bool(summary.get("any_notable_change")),
    }


def _region_sentence(region):
    change = region.get("area_change_pct")
    confidence = region.get("confidence")
    tail = (f" (alignment confidence {confidence:.2f})"
            if isinstance(confidence, (int, float)) else "")
    if change is None:
        return f"One marked area could not be measured{tail}."
    direction = "larger" if change > 0 else "smaller"
    if abs(change) < 1.0:
        return f"One marked area is about the same size as last time{tail}."
    return (f"One marked area is about {abs(change):.0f}% {direction} than last "
            f"time{tail}.")


EXCLUDED_REASON = {
    "partial_out_of_frame": ("part of it is outside one of the two photographs, so "
                             "its size cannot be compared fairly"),
    "unreliable_low_confidence": ("the two photographs did not line up well enough "
                                  "to compare it"),
}


def sentences(comparison):
    """Plain-language lines for a visit pair. Refusals are stated in full."""
    if comparison["state"] in (FAILED, UNRELIABLE):
        return [
            "This photograph could not be compared with your previous one reliably.",
            f"Reason: {comparison['reason']}",
            RETAKE,
        ]

    lines = [f"The two photographs line up with {_confidence_word(comparison['confidence'])} "
             f"(confidence {comparison['confidence']:.2f})."]

    if comparison["regions"]:
        lines += [_region_sentence(r) for r in comparison["regions"]]
        if not comparison["any_notable_change"]:
            lines.append("No change large enough to be worth noting was measured.")
    else:
        lines.append("No area could be compared between the two photographs.")

    for region in comparison["excluded"]:
        reason = (EXCLUDED_REASON.get(region.get("comparability"))
                  or region.get("reason")
                  or "it could not be compared reliably")
        lines.append(f"One marked area was left out of the comparison because {reason}.")

    return lines
