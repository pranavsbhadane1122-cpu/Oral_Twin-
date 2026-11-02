"""Render an explanation as readable text.

THE ORDER IS THE POINT.

"Explain why it identifies a particular risk, rather than simply giving a
prediction" governs the sequence, not just the content. An output that states a
band and then justifies it is a prediction with commentary attached: the reader
has already taken the verdict and the reasoning arrives as decoration. An output
where the band appears only as the conclusion of stated reasoning is a different
object, even with identical words.

So the band appears in section 5 and NOWHERE ABOVE IT. A test asserts the band
string occurs exactly once in a rendered report and that its first occurrence is
inside THEREFORE.

Sections 1 and 7 render on every report, comparable or refused. Section 1 is the
"particular" in "a particular risk": the risk belongs to an area of a
photograph, not to the photograph and not to the person. Section 7 is what the
output does not say, generated from one constant so it cannot drift between
report types.
"""

from src.explain import claims

# --------------------------------------------------------------------------
# Section 7, from a single constant.
#
# Repeating these by hand per report type is how two report types end up
# disclaiming different things, and the shorter list is always the one on the
# report that needed the longer one.
# --------------------------------------------------------------------------
DOES_NOT_SAY = (
    "This does not identify a condition.",
    "This does not say how likely anything is.",
    "This measures change between two photographs and nothing else.",
)

HEADINGS = {
    "looked_at": "WHAT WAS LOOKED AT",
    "measured": "WHAT WAS MEASURED",
    "judged": "HOW IT WAS JUDGED",
    "decided": "WHAT DECIDED IT",
    "therefore": "THEREFORE",
    "changed": "WHAT WOULD HAVE CHANGED IT",
    "not_say": "WHAT THIS DOES NOT SAY",
    "refused": "WHY NO COMPARISON WAS MADE",
}


def render(explanation):
    if explanation["kind"] == "refused":
        return _render_refused(explanation)
    return _render_comparable(explanation)


def _region_lines(region):
    """Section 1. Never invents a location it does not have."""
    if not region:
        return ["  The marked region in the earlier photo."]
    bbox = region.get("bbox")
    if not bbox:
        # A stored report predating the bbox field. Say what is known and no
        # more: a fabricated coordinate would be worse than a vague one.
        return ["  The marked region in the earlier photo."]
    left, top, width, height = bbox
    return [f"  A region {width} by {height} pixels, {left} across and {top} "
            f"down from the top-left of the earlier photo."]


def _dates_line(explanation):
    earlier = explanation.get("earlier_date")
    later = explanation.get("later_date")
    if earlier and later:
        return [f"  Comparing the photo from {earlier} with the one from {later}."]
    return ["  Comparing the earlier photo with the later one."]


def _render_comparable(e):
    lines = []

    # 1 ---------------------------------------------------------------
    lines += [HEADINGS["looked_at"], ""]
    lines += _region_lines(e.get("region"))
    lines += _dates_line(e)
    if e.get("region"):
        lines.append(f"  This records {e['region']['describes']}.")
    lines.append("")

    # 2 ---------------------------------------------------------------
    lines += [HEADINGS["measured"], ""]
    for m in e["measurements"]:
        lines.append(f"  - {m['name']}: {m['display']}")
    lines.append("")

    # 3 ---------------------------------------------------------------
    lines += [HEADINGS["judged"], ""]
    if e["rules"]:
        for rule in e["rules"]:
            lines.append(f"  - {rule['sentence']}")
            if rule["threshold_key"]:
                lines.append(f"      That threshold is {rule['provenance_phrasing']}.")
                if rule["provenance_state"] == claims.UNREGISTERED:
                    lines.append("      This is a defect: a threshold that "
                                 "decided an outcome should not be unregistered.")
    else:
        lines.append("  Every measurement above was compared against its "
                     "threshold. None reached it.")
    lines.append("")

    # 4 ---------------------------------------------------------------
    lines += [HEADINGS["decided"], ""]
    lines.append(f"  {e['deciding_text']}")
    lines.append("")

    # 5 --- the band, for the first time in the output --------------------
    lines += [HEADINGS["therefore"], ""]
    lines.append(f"  {e['band']}.")
    lines.append(f"  {e['risk_statement']}")
    lines.append("")

    # 6 ---------------------------------------------------------------
    if e["counterfactual"]:
        lines += [HEADINGS["changed"], ""]
        lines.append(f"  {e['counterfactual']['text']}")
        lines.append("")

    if e["advisories"]:
        lines += ["WORTH KNOWING", ""]
        for advisory in e["advisories"]:
            lines.append(f"  - {advisory}")
        lines.append("")

    # 7 ---------------------------------------------------------------
    lines += _not_say_lines()
    lines.append(e["disclaimer"])
    return "\n".join(lines)


def _render_refused(e):
    lines = []

    # 1 - the same section, so the reader can see what was attempted.
    #
    # BUT WITHOUT COORDINATES, and that is a decision rather than an omission.
    # Coordinates answer "where was this number computed" - and on a refusal no
    # number was computed, so there is nothing to locate. Printing a pixel box
    # beside a refusal would offer precision about an measurement that does not
    # exist. Dates still render: they say which two photographs were compared,
    # which remains true and is what section 1 is for.
    lines += [HEADINGS["looked_at"], ""]
    lines.append("  The marked region in the earlier photo.")
    lines += _dates_line(e)
    lines.append("")

    # 2-6 collapse to this, with no numbers of any kind
    lines += [HEADINGS["refused"], ""]
    lines.append(f"  {e['check_failed']}")
    lines.append(f"  {e['what_to_do']}")
    lines.append("")
    lines.append("  No comparison was made, so there is nothing to report.")
    lines.append("")

    # 7 — identical wording, from the same constant
    lines += _not_say_lines()
    lines.append(e["disclaimer"])
    return "\n".join(lines)


def _not_say_lines():
    out = [HEADINGS["not_say"], ""]
    out += [f"  - {line}" for line in DOES_NOT_SAY]
    out.append("")
    return out


def explanation_strings(explanation):
    """Every string the explanation can put in front of a reader.

    Used by the vocabulary and placeholder tests, so a new field cannot escape
    the guards by being added without anyone remembering to check it.
    """
    out = list(DOES_NOT_SAY)
    if explanation["kind"] == "refused":
        out += [explanation["check_failed"], explanation["what_to_do"],
                explanation["disclaimer"]]
        # no coordinates on a refusal: see _render_refused
        out.append("The marked region in the earlier photo.")
        return [s for s in out if s]

    out += [explanation["deciding_text"], explanation["risk_statement"],
            explanation["disclaimer"]]
    out += [m["name"] for m in explanation["measurements"]]
    out += _region_lines(explanation.get("region"))
    for rule in explanation["rules"]:
        out.append(rule["sentence"])
        if rule["provenance_phrasing"]:
            out.append(rule["provenance_phrasing"])
    if explanation["counterfactual"]:
        out.append(explanation["counterfactual"]["text"])
    out += list(explanation["advisories"])
    return [s for s in out if s]
