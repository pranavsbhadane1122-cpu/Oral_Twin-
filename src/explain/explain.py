"""Why a comparison landed where it did.

Satisfies the requirement that the system explain its reasoning rather than
only produce an output. Every comparable result carries a complete account:
what was measured, which measurement decided, the rule that decided it, where
that threshold came from, what would have changed the answer, and where in the
image the number was computed.

TWO DISCIPLINES THIS MODULE KEEPS
---------------------------------

It recomputes NOTHING. Every value comes from the change report and the risk
result it is handed - the same ones the UI renders. A module that recomputed
could describe a different result than the one on screen, and the user would
have no way to tell which was which.

A REFUSAL GETS A QUALITATIVE EXPLANATION ONLY. No numbers, not even the
alignment confidence. "The two photos could not be lined up well enough" is a
complete explanation. A confidence value printed beside a refusal is the Phase 5
failure again: a reader will treat any number as a result.
"""

from src.explain import claims
from src.risk.risk_rules import (
    BAND_TEXT,
    ELEVATED,
    LOW,
    MODERATE,
    NOT_ASSESSED,
    ORDER,
    RULE_SPECS,
    _fmt,
    rule_spec,
)
from src.utils.config import load_config

DISCLAIMER = "Have this checked by a dentist."

# Plain-language names for every measurement that can appear, so no variable
# name reaches a reader.
MEASUREMENT_NAMES = {
    "area_change_pct": "how much the marked area grew or shrank since the last visit",
    "area_visit1_px": "the size of the marked area at the earlier visit",
    "area_visit2_px": "the size of the marked area at this visit",
    "color_magnitude": "how much the colour of the marked area changed compared "
                       "with the tissue around it",
    "match_iou": "how well the marked areas in the two photos overlap",
    "state": "whether the marked area is new, matched or no longer visible",
    "comparability": "whether this area could be compared at all",
    "confidence": "how well the two photographs lined up",
}

UNITS = {"area_change_pct": "%", "area_visit1_px": " pixels",
         "area_visit2_px": " pixels"}

# Derived figures are rounded to ONE DECIMAL AT MOST before a reader sees them.
#
# 71.48% claims precision the pipeline does not have. At the median alignment
# corner error of 1.18 px, a coherent one-pixel misplacement of the boundary of
# a median 1,792 px region (radius ~24 px, circumference ~150 px) moves the
# measured area by roughly 177 px - about 10% of the region. Two decimal places
# assert a hundredth of a percent on a quantity whose uncertainty is nearer ten
# percent.
#
# That bound is pessimistic: it assumes the whole boundary shifts coherently,
# whereas homography error is usually smallest near the image centre where
# these regions sit. It is an upper bound, not a point estimate. Even so, the
# second decimal is indefensible and the first is generous.
# Percentages and magnitudes: one decimal. Ratios in [0, 1]: two.
#
# A ratio needs the second decimal because the first destroys it - an overlap
# of 0.9723 rounded to one decimal is 1.0, which _fmt renders as "1" and a
# reader takes as a perfect overlap. That is worse than false precision: it is
# a false claim. Caught by the baseline drift gate on the change that
# introduced it.
DERIVED_ONE_DP = {"area_change_pct", "color_magnitude"}
DERIVED_TWO_DP = {"match_iou"}


def _display(key, value):
    """How a value is shown. Rounds derived quantities; leaves counts alone."""
    if isinstance(value, float):
        if key in DERIVED_ONE_DP:
            value = round(value, 1)
        elif key in DERIVED_TWO_DP:
            value = round(value, 2)
    return f"{_fmt(value)}{UNITS.get(key, '')}"


# --------------------------------------------------------------------------
# The risk statement.
#
# It says what the risk IS, in terms of ACTION rather than condition. "This area
# has changed enough to be worth having looked at" is a statement about what to
# do; "this area shows signs of X" is a statement about what something is, and
# this project cannot support one.
#
# It names a change and a reason to seek review. It never names a condition, a
# category, a likelihood, or a body system - and a test enforces each of those,
# including the probability words, because "likely" is the easiest one to reach
# for when describing a band.
# --------------------------------------------------------------------------

RISK_STATEMENTS = {
    ELEVATED: ("This area has changed enough since {earlier} to be worth having "
               "looked at."),
    MODERATE: ("This area has changed since {earlier} by enough to be worth "
               "mentioning at your next visit."),
    LOW: ("Nothing measured here changed enough since {earlier} to single this "
          "area out."),
    NOT_ASSESSED: ("No comparison was made, so there is nothing to act on either "
                   "way."),
}


def risk_statement(band, earlier_date=None):
    """The band as a reason to act, with no condition and no likelihood."""
    template = RISK_STATEMENTS.get(band, RISK_STATEMENTS[NOT_ASSESSED])
    # "{earlier}" must never survive into output: when no date is known the
    # sentence is rewritten rather than left with a hole in it. An unfilled
    # placeholder in a report about someone is worse than a vaguer sentence.
    if "{earlier}" not in template:
        return template
    if earlier_date:
        return template.format(earlier=earlier_date)
    return template.replace("since {earlier} ", "since the earlier photo ") \
                   .replace("since {earlier}", "since the earlier photo")


def _measurements(lesion, confidence):
    """Every value that fed the decision, named in plain language."""
    colour = lesion.get("color_shift") or {}
    raw = {
        "area_change_pct": lesion.get("area_change_pct"),
        "area_visit1_px": lesion.get("area_visit1_px"),
        "area_visit2_px": lesion.get("area_visit2_px"),
        "color_magnitude": colour.get("magnitude") if colour.get("measurable") else None,
        "match_iou": lesion.get("match_iou"),
        "state": lesion.get("state"),
        "confidence": round(float(confidence), 2),
    }
    out = []
    for key, value in raw.items():
        if value is None:
            continue
        out.append({
            "key": key,
            "name": MEASUREMENT_NAMES.get(key, key),
            "value": value,
            "display": _display(key, value),
        })
    return out


def _region(lesion, spec):
    """Where the deciding measurement was computed.

    The honest counterpart to a Grad-CAM heatmap. This is NOT a claim about
    where a model attended - it is a record of which pixels the number came
    from, which is a different and much weaker claim, and the only one this
    project can support.

    No Grad-CAM output in this project has been scored by a clinician
    (docs/CLAIMS.md, "What I expected to find and could not", finding 4), so
    attention claims are not available to us. A computation record is.
    """
    box = lesion.get("bbox_visit1") or lesion.get("bbox") or None
    return {
        "source": spec.region_source if spec else "lesion_bbox",
        "bbox": box,
        "describes": "where this measurement was computed, not where a model looked",
    }


def _counterfactual(spec, measured, cfg):
    """The smallest change that would have produced a different band.

    Computed by INVERTING the rule, not by re-running it in a loop. A loop
    would be slower, would need a step size nobody chose, and would answer a
    slightly different question - "which of the values I happened to try" -
    rather than "where is the boundary".
    """
    if spec is None or spec.threshold_key is None:
        return None
    threshold = spec.threshold(cfg)
    if threshold is None or measured is None:
        return None

    unit = spec.unit
    prov = claims.provenance(spec.threshold_key)
    caveat = ""
    if prov["state"] in (claims.PROVISIONAL, claims.ARTEFACT_ONLY,
                         claims.UNREGISTERED):
        caveat = (f", where {_fmt(threshold)}{unit} is "
                  f"{prov['phrasing']}")

    return {
        "threshold": threshold,
        "measured": measured,
        "margin": round(float(measured) - float(threshold), 1),
        "text": (f"This would have been one band lower if "
                 f"{spec.measurement_name} had been below "
                 f"{_fmt(threshold)}{unit}{caveat}."),
    }


def _deciding(fired, cfg):
    """Which fired rule(s) set the band. Several can tie."""
    if not fired:
        return []
    top = max(ORDER[r["level"]] for r in fired)
    return [r for r in fired if ORDER[r["level"]] == top]


def explain_comparable(change_report, risk, cfg=None,
                       earlier_date=None, later_date=None):
    """Full explanation for a comparison that was actually made."""
    cfg = cfg or load_config()
    confidence = float(change_report.get("alignment_confidence", 0.0))
    lesions = change_report.get("lesions", []) or []
    fired = risk.get("rules_fired", []) or []
    band = risk.get("band", LOW)

    deciding = _deciding(fired, cfg)
    primary = lesions[0] if lesions else {}

    rules = []
    for entry in fired:
        spec = rule_spec(entry["rule"])
        prov = claims.provenance(spec.threshold_key if spec else None)
        rules.append({
            "rule": entry["rule"],
            "level": entry["level"],
            # generated from the spec, never written beside it
            "sentence": spec.sentence(cfg) if spec else entry.get("detail", ""),
            "threshold_key": spec.threshold_key if spec else None,
            "threshold": spec.threshold(cfg) if spec else None,
            "provenance_state": prov["state"],
            "provenance_note": prov["note"],
            "provenance_phrasing": prov["phrasing"],
            "deciding": entry in deciding,
        })

    # the deciding measurement, and its counterfactual
    counterfactual = None
    region = None
    if deciding:
        spec = rule_spec(deciding[0]["rule"])
        measured = None
        if spec:
            if spec.measurement == "color_magnitude":
                measured = (primary.get("color_shift") or {}).get("magnitude")
            else:
                measured = primary.get(spec.measurement)
        counterfactual = _counterfactual(spec, measured, cfg)
        region = _region(primary, spec)

    if deciding:
        names = [rule_spec(d["rule"]).measurement_name
                 if rule_spec(d["rule"]) else d["rule"] for d in deciding]
        if len(deciding) == 1:
            deciding_text = f"The band was set by {names[0]}."
        else:
            deciding_text = ("Two measurements crossed their thresholds at the "
                             "same level: " + "; ".join(names) + ".")
    else:
        # A band assigned because nothing crossed is still a decision.
        deciding_text = ("No measurement crossed its threshold. This band was "
                         "assigned because nothing stood out, which is a "
                         "decision in its own right rather than an absence of "
                         "one.")

    return {
        "kind": "comparable",
        "band": band,
        "band_text": BAND_TEXT.get(band, ""),
        "risk_statement": risk_statement(band, earlier_date),
        "earlier_date": earlier_date,
        "later_date": later_date,
        "measurements": _measurements(primary, confidence),
        "deciding_text": deciding_text,
        "deciding_rules": [d["rule"] for d in deciding],
        "rules": rules,
        "counterfactual": counterfactual,
        "region": region,
        "advisories": list(risk.get("advisories", []) or []),
        "disclaimer": DISCLAIMER,
    }


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

# Which check failed, and what to do - in words, with no numbers at all. The
# alignment confidence is deliberately absent: it is the number a reader would
# most readily mistake for a result.
REFUSAL_CHECKS = {
    "alignment_failed": (
        "The app could not match the two photographs to each other at all.",
        "Retake the newer photo from the same angle as the older one, with the "
        "same area in view.",
    ),
    "low_confidence": (
        "The two photographs could not be lined up well enough to compare them.",
        "Retake the newer photo in better, even light, holding the phone still.",
    ),
    "partial_out_of_frame": (
        "Too little of the older photograph appears in the newer one to compare "
        "them fairly.",
        "Retake the newer photo with more of the same area in frame.",
    ),
    "not_comparable": (
        "These two photographs could not be compared reliably.",
        "Retake the newer photo in similar lighting and framing to the older one.",
    ),
}


def explain_refused(check="not_comparable", region=None,
                    earlier_date=None, later_date=None):
    """Qualitative explanation only. No numbers, by construction.

    There is nothing to interpolate a measurement into: the two strings are
    constants, selected by a key. That is the same structural defence the
    Android ComparisonResult.Refused uses - if there is no number on the
    object, no screen can print one.
    """
    failed, action = REFUSAL_CHECKS.get(check, REFUSAL_CHECKS["not_comparable"])
    return {
        "kind": "refused",
        "band": NOT_ASSESSED,
        "band_text": BAND_TEXT[NOT_ASSESSED],
        "check_failed": failed,
        "what_to_do": action,
        # section 1 renders on refusals too, so the reader can see WHAT was
        # attempted rather than only that it did not work
        "region": region,
        "earlier_date": earlier_date,
        "later_date": later_date,
        "disclaimer": DISCLAIMER,
    }


def explain(change_report, risk, cfg=None, earlier_date=None, later_date=None):
    """Dispatch on the result, exactly as the UI does."""
    if change_report.get("status") != "ok" or not (
            change_report.get("summary", {}) or {}).get("comparable", False):
        reason = (change_report.get("reason") or "").lower()
        lesions = change_report.get("lesions") or []
        region = _region(lesions[0], None) if lesions else None
        if "could not be lined up" in reason or "below the floor" in reason:
            check = "low_confidence"
        elif "align" in reason:
            check = "alignment_failed"
        else:
            check = "not_comparable"
        return explain_refused(check, region=region,
                               earlier_date=earlier_date, later_date=later_date)
    return explain_comparable(change_report, risk, cfg,
                              earlier_date=earlier_date, later_date=later_date)


def unregistered_thresholds():
    """Thresholds a rule depends on that the register does not list.

    An unregistered threshold deciding a band is the exact thing CLAIMS.md was
    built to catch, so this is surfaced as data for a test rather than left to
    be noticed by eye.
    """
    missing = []
    for rule_id, spec in RULE_SPECS.items():
        if spec.threshold_key is None:
            continue
        if claims.provenance(spec.threshold_key)["state"] == claims.UNREGISTERED:
            missing.append((rule_id, spec.threshold_key))
    return missing
