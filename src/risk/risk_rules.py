"""Rule-based risk stratification. NOT a learned risk model.

Every band below is produced by explicit, inspectable rules over quantities the
system actually measured: area change on a comparable lesion, a new lesion
appearing, relative colour change past the calibrated threshold, and a
confidently-predicted gum condition persisting across visits. There is no
trained risk classifier, no personal probability, and no diagnosis.

Deliberate exclusion
--------------------
The edge/compactness metric is NOT used by any rule. Its validation showed a
35% false-flag rate because perimeter^2/area is sensitive to mask resampling
rather than to real border change, so it must not influence user-facing output
until it is reformulated. Nothing in this module reads `edge_irregularity`.

Confidence discipline
---------------------
Low alignment confidence and non-comparable (partly out-of-frame) lesions can
never RAISE a band - a measurement we do not trust is not evidence of risk.
They only add an advisory asking for a better photo.

General context, not personal prediction
----------------------------------------
Population-level research associates gum disease with cardiovascular disease
and with diabetes; these are associations reported across populations, not
predictions about any individual, and they are surfaced as background reading
only:

  Sanz M, Marco del Castillo A, Jepsen S, et al. "Periodontitis and
  cardiovascular diseases: Consensus report." Journal of Clinical
  Periodontology, 2020;47(3):268-288.

  Preshaw PM, Alba AL, Herrera D, Jepsen S, Konstantinidis A, Makrilakis K,
  Taylor R. "Periodontitis and diabetes: a two-way relationship."
  Diabetologia, 2012;55(1):21-31.

OralTwin is a screening aid, not a diagnostic tool.
"""

from src.delta.compare import FULL, PARTIAL
from src.utils.config import load_config

LOW, MODERATE, ELEVATED = "Low", "Moderate", "Elevated"
ORDER = {LOW: 0, MODERATE: 1, ELEVATED: 2}

DENTIST_LINE = "Have this checked by a dentist."
RETAKE_LINE = ("Some of this could not be compared reliably - retake the photo with the "
               "same framing and lighting, and the whole area in view.")

CONTEXT_NOTE = (
    "General context, not a prediction about you: population studies report an "
    "association between gum disease and cardiovascular disease and diabetes "
    "(Sanz et al., J Clin Periodontol 2020; Preshaw et al., Diabetologia 2012). "
    "Looking after your gums is worthwhile regardless of what this screening shows."
)

BAND_TEXT = {
    LOW: "Nothing in these two photos stood out as changed.",
    MODERATE: "Something changed between these two photos that is worth a professional look.",
    ELEVATED: "A clear change stands out between these two photos.",
}


def _comparable(change_report):
    return [l for l in change_report.get("lesions", []) if l.get("comparability") == FULL]


def _rule(rule_id, level, detail):
    return {"rule": rule_id, "level": level, "detail": detail}


def evaluate_rules(change_report, prediction_visit1=None, prediction_visit2=None,
                   alignment_confidence=None, cfg=None):
    """Return (fired_rules, advisories). Pure function over measurements."""
    cfg = cfg or load_config()
    rcfg = cfg["risk"]
    fired, advisories = [], []

    confidence = (alignment_confidence
                  if alignment_confidence is not None
                  else change_report.get("alignment_confidence", 0.0))

    if change_report.get("status") != "ok":
        advisories.append(change_report.get("reason") or "the two photos could not be compared")
        return fired, advisories

    lesions = _comparable(change_report)
    partials = [l for l in change_report.get("lesions", []) if l.get("comparability") == PARTIAL]

    for lesion in lesions:
        change = lesion.get("area_change_pct")
        if change is not None and change >= rcfg["growth_elevated_pct"]:
            fired.append(_rule(
                "AREA_GROWTH_LARGE", ELEVATED,
                f"a spot grew about {change:.0f}% in area since the last visit "
                f"(threshold {rcfg['growth_elevated_pct']:.0f}%)",
            ))
        elif change is not None and change >= rcfg["growth_moderate_pct"]:
            fired.append(_rule(
                "AREA_GROWTH", MODERATE,
                f"a spot grew about {change:.0f}% in area since the last visit "
                f"(threshold {rcfg['growth_moderate_pct']:.0f}%)",
            ))

        if lesion.get("state") == "new":
            fired.append(_rule("NEW_LESION", MODERATE,
                               "a spot is visible now that was not visible last visit"))

        color = lesion.get("color_shift") or {}
        if color.get("measurable") and color.get("notable"):
            fired.append(_rule(
                "COLOUR_CHANGE", MODERATE,
                f"a spot's colour changed relative to the tissue around it "
                f"(magnitude {color.get('magnitude')}, calibrated threshold "
                f"{cfg['delta']['relative_color_thresh']})",
            ))

    # the classifier is parked: while surface_predictions is false no rule may
    # depend on its output, so PERSISTENT_CONDITION cannot fire at all
    surface = cfg["classification"].get("surface_predictions", False)
    p1, p2 = (prediction_visit1 or {}, prediction_visit2 or {}) if surface else ({}, {})
    persistent = [c.lower() for c in rcfg["persistent_conditions"]]
    threshold = rcfg["condition_prob_high"]
    same = (p1.get("class_name") and p1.get("class_name") == p2.get("class_name"))
    if (same and p1["class_name"].lower() in persistent
            and float(p1.get("probability", 0)) >= threshold
            and float(p2.get("probability", 0)) >= threshold):
        fired.append(_rule(
            "PERSISTENT_CONDITION", MODERATE,
            f"signs consistent with {p1['class_name']} were picked up at both visits "
            f"with high model confidence ({p1['probability']:.2f}, {p2['probability']:.2f})",
        ))

    if partials:
        advisories.append(
            f"{len(partials)} spot(s) were only partly inside the newer photo, so their "
            f"size could not be compared. " + RETAKE_LINE
        )
    if change_report.get("reliability") == "low" or float(confidence) < 0.5:
        advisories.append(
            f"Alignment confidence was low ({float(confidence):.2f}), so these measurements "
            f"are less certain than usual. " + RETAKE_LINE
        )
    return fired, advisories


def stratify(change_report, prediction_visit1=None, prediction_visit2=None,
             alignment_confidence=None, cfg=None):
    """Assign a Low / Moderate / Elevated band with the rules that produced it."""
    cfg = cfg or load_config()
    rcfg = cfg["risk"]
    fired, advisories = evaluate_rules(change_report, prediction_visit1, prediction_visit2,
                                       alignment_confidence, cfg)

    band = LOW
    for rule in fired:
        if ORDER[rule["level"]] > ORDER[band]:
            band = rule["level"]

    escalated = False
    n_moderate = sum(1 for r in fired if r["level"] == MODERATE)
    if band == MODERATE and n_moderate >= rcfg["escalate_on_rule_count"]:
        band = ELEVATED
        escalated = True

    lines = [BAND_TEXT[band]]
    lines += [f"- {r['detail']}" for r in fired]
    if escalated:
        lines.append(f"- {n_moderate} separate signals changed at once, which is why this is "
                     f"flagged higher than any single one of them")
    if advisories:
        lines += [f"- {a}" for a in advisories]
    if band != LOW:
        lines.append(DENTIST_LINE)
    else:
        lines.append("Keep taking a photo at each visit so changes stay easy to spot.")

    return {
        "band": band,
        "rules_fired": fired,
        "n_rules": len(fired),
        "escalated_by_rule_count": escalated,
        "advisories": advisories,
        "text": "\n".join(lines),
        "context_note": CONTEXT_NOTE,
        "basis": "rule-based; no learned risk model and no personal prediction",
        "excluded_metrics": ["edge_irregularity (resampling-sensitive, known unreliable)"],
        "disclaimer": "OralTwin is a screening aid, not a diagnostic tool.",
    }
