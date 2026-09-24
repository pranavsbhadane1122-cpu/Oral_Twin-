"""Assemble one human-readable visit report.

Combines the classifier's read of each photo, the per-lesion change lines from
the Phase 5 ChangeReport, the rule-based risk band, and pointers to the
Grad-CAM images, into text a patient can read.

Language rules enforced here:
  - never states or implies a diagnosis; the classifier's output is phrased as
    "signs consistent with X" with its confidence shown
  - anything above the Low band carries "Have this checked by a dentist."
  - measurements that could not be trusted are said to be untrusted, not hidden

OralTwin is a screening aid, not a diagnostic tool.
"""

from src.delta.compare import FULL, PARTIAL
from src.risk.risk_rules import LOW
from src.utils.config import load_config

HEADER = "ORALTWIN VISIT COMPARISON"
FOOTER = ("OralTwin is a screening aid, not a diagnostic tool. It cannot diagnose anything. "
          "Have any finding checked by a dentist.")


PARKED_NOTICE = [
    "  Condition screening is switched off in this build.",
    "  (The condition model is parked pending better training data, so this report",
    "   describes CHANGE between the two photos only, and names no condition.)",
]


def condition_block(prediction_visit1, prediction_visit2, cfg=None):
    """Condition lines, or a parked notice when classification.surface_predictions
    is false. Nothing derived from the classifier may cross this boundary."""
    cfg = cfg or load_config()
    if not cfg["classification"].get("surface_predictions", False):
        return list(PARKED_NOTICE)
    return [
        condition_line(prediction_visit1, "previous visit"),
        condition_line(prediction_visit2, "this visit"),
        "  (These are pattern matches from a screening model, not a diagnosis.)",
    ]


def condition_line(prediction, label):
    if not prediction:
        return f"  {label}: not assessed"
    return (f"  {label}: signs consistent with {prediction['class_name']} "
            f"(model confidence {prediction['probability']:.2f})")


def lesion_lines(change_report):
    lesions = change_report.get("lesions", [])
    if not lesions:
        return ["  no distinct spots were tracked in these photos"]
    lines = []
    for lesion in lesions:
        tag = {FULL: "", PARTIAL: " [not comparable]"}.get(lesion.get("comparability"), "")
        lines.append(f"  - spot {lesion['id']}{tag}: {lesion['message']}")
    return lines


def build_report(pair_name, change_report, risk, prediction_visit1=None,
                 prediction_visit2=None, image_paths=None, alignment=None, cfg=None):
    """Return the report as a plain-text string."""
    image_paths = image_paths or {}
    alignment = alignment or {}

    lines = [
        "=" * 68,
        f"{HEADER} - {pair_name}",
        "=" * 68,
        "",
        "WHAT THE PHOTOS SHOW",
        *condition_block(prediction_visit1, prediction_visit2, cfg),
        "",
        "WHAT CHANGED SINCE LAST VISIT",
    ]

    if change_report.get("status") != "ok":
        lines.append(f"  Could not compare: {change_report.get('reason')}")
    else:
        lines.append(f"  {change_report['summary']['headline']}")
        lines += lesion_lines(change_report)
        conf = change_report.get("alignment_confidence")
        if conf is not None:
            lines.append(f"  (photo alignment confidence {conf:.2f}"
                         f"{', ' + alignment['reason'] if alignment.get('reason') else ''})")

    lines += ["", f"SCREENING BAND: {risk['band']}", ""]
    lines += [f"  {line}" for line in risk["text"].splitlines()]

    if risk["rules_fired"]:
        lines += ["", "  Why this band (explicit rules):"]
        for rule in risk["rules_fired"]:
            lines.append(f"    [{rule['rule']}] ({rule['level']}) {rule['detail']}")
    else:
        lines += ["", "  Why this band: no change rule was triggered."]

    if image_paths:
        lines += ["", "WHERE THE MODEL WAS LOOKING"]
        for label, path in image_paths.items():
            lines.append(f"  {label}: {path}")
        lines.append("  Warm colours mark the regions that most influenced the model. "
                     "On the change map, red gained attention and blue lost it.")

    lines += ["", "BACKGROUND", f"  {risk['context_note']}"]
    if risk["band"] != LOW:
        lines += ["", "NEXT STEP", "  Book a check-up and show these photos to your dentist."]

    lines += ["", "-" * 68, FOOTER, ""]
    return "\n".join(lines)
