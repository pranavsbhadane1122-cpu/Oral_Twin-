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
from src.risk.risk_rules import NOT_ASSESSED, LOW
from src.utils.config import load_config

HEADER = "ORALTWIN VISIT COMPARISON"
FOOTER = ("OralTwin is a screening aid, not a diagnostic tool. It cannot diagnose anything. "
          "Have any finding checked by a dentist.")


PARKED_NOTICE = [
    "  Condition screening is switched off in this build.",
    "  (The condition model is parked pending better training data, so this report",
    "   describes CHANGE between the two photos only, and names no condition.)",
]


# "not assessed" was the wording here until 2026-11-04. It collided
# with the Not assessed BAND name, so a report with no prediction
# appeared to announce its band in the header - which the ordering
# test caught. Two different meanings must not share a phrase.
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
    """A SCORE, never a condition name.

    Until 2026-11-04 this read "signs consistent with <class> (model
    confidence 0.88)", which was unreachable while the classifier was parked.
    Surfacing the classifier for Tracks A and C made it reachable, and it was
    wrong twice over: it names a condition, which this project does not do,
    and "signs consistent with" is a diagnostic phrasing - in a report whose
    own closing section says "this does not identify a condition".

    What the track brief actually asks for is the SCORE with the measured
    performance beside it. The score is what the model computes; the name is
    an interpretation it has not earned, from a model that failed its
    attention gate at 6/10.
    """
    if not prediction:
        return f"  {label}: no screening score for this photograph"
    return (f"  {label}: screening score {prediction['probability']:.2f} "
            f"(see the measured performance below; the model does not name "
            f"what it reacted to)")


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

    # ----------------------------------------------------------------
    # THE ORDER OF THIS REPORT IS THE POINT.
    #
    # The band appears once, in THEREFORE, as the conclusion of reasoning the
    # reader has already seen. It used to appear in a "SCREENING BAND: X"
    # header two thirds of the way up, which made everything below it
    # commentary on a verdict already delivered. The words were the same; the
    # object was a prediction with justification attached.
    #
    # Nothing above THEREFORE may name the band. The old header, the risk
    # text, the per-rule detail lines and the headline summary all did, so
    # they are gone - their content is in the explanation sections, generated
    # from RULE_SPECS rather than restated here.
    # ----------------------------------------------------------------
    lines = [
        "=" * 68,
        f"{HEADER} - {pair_name}",
        "=" * 68,
        "",
        "WHAT THE PHOTOS SHOW",
        *condition_block(prediction_visit1, prediction_visit2, cfg),
        "",
    ]

    explanation_failed = None
    try:
        from src.explain.explain import explain as build_explanation
        from src.explain.render import render as render_explanation

        explanation = build_explanation(change_report, risk, cfg)
        lines += render_explanation(explanation).splitlines()
    except Exception as exc:                            # noqa: BLE001
        # A broken explanation must not suppress the report, but it must be
        # visible rather than silently absent.
        explanation_failed = f"{type(exc).__name__}: {exc}"
        lines += [f"[explanation unavailable: {explanation_failed}]"]

    if image_paths:
        lines += ["", "WHERE THE MEASUREMENTS CAME FROM"]
        for label, path in image_paths.items():
            lines.append(f"  {label}: {path}")
        lines.append("  These images mark the regions the numbers above were "
                     "computed from.")

    lines += ["", "BACKGROUND", f"  {risk['context_note']}"]
    if risk["band"] not in (LOW, NOT_ASSESSED):
        lines += ["", "NEXT STEP", "  Book a check-up and show these photos to your dentist."]

    lines += ["", "-" * 68, FOOTER, ""]
    return "\n".join(lines)
