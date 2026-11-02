"""Render an explanation as readable text for the Python report.

Separate from explain.py so the structure and its wording can be tested apart.
Every string produced here goes through the project's single vocabulary guard
(tests/test_caption.py); there is no second guard in this module.
"""

from src.explain import claims


def render(explanation):
    """The explanation as plain text."""
    if explanation["kind"] == "refused":
        return _render_refused(explanation)
    return _render_comparable(explanation)


def _render_refused(e):
    return "\n".join([
        "WHY YOU ARE NOT SEEING A RESULT",
        "",
        e["check_failed"],
        e["what_to_do"],
        "",
        # no numbers, not even the confidence: see explain.explain_refused
        "No comparison was made, so there is nothing to report.",
        "",
        e["disclaimer"],
    ])


def _render_comparable(e):
    lines = ["WHY THIS RESULT", "", e["band_text"], ""]

    lines.append("What was measured")
    for m in e["measurements"]:
        lines.append(f"  - {m['name']}: {m['display']}")
    lines.append("")

    lines.append("What decided it")
    lines.append(f"  {e['deciding_text']}")
    lines.append("")

    if e["rules"]:
        lines.append("The rules that applied")
        for rule in e["rules"]:
            marker = " (this one set the band)" if rule["deciding"] else ""
            lines.append(f"  - {rule['sentence']}{marker}")
            if rule["threshold_key"]:
                lines.append(f"      That threshold is {rule['provenance_phrasing']}.")
                if rule["provenance_state"] == claims.UNREGISTERED:
                    lines.append("      This is a defect: a threshold that decided "
                                 "a band should not be unregistered.")
        lines.append("")

    if e["counterfactual"]:
        lines.append("What would have changed it")
        lines.append(f"  {e['counterfactual']['text']}")
        lines.append("")

    region = e.get("region")
    if region:
        lines.append("Where the number came from")
        if region["bbox"]:
            left, top, width, height = region["bbox"]
            lines.append(f"  A region {width} by {height} pixels, {left} across and "
                         f"{top} down from the top-left of the earlier photo.")
        else:
            lines.append("  The marked region in the earlier photo.")
        lines.append(f"  This records {region['describes']}.")
        lines.append("")

    if e["advisories"]:
        lines.append("Worth knowing")
        for advisory in e["advisories"]:
            lines.append(f"  - {advisory}")
        lines.append("")

    lines.append(e["disclaimer"])
    return "\n".join(lines)


def explanation_strings(explanation):
    """Every string the explanation can put in front of a reader.

    Used by the vocabulary test, so a new field cannot escape the guard by
    being added without anyone remembering to check it.
    """
    out = []
    if explanation["kind"] == "refused":
        out += [explanation["check_failed"], explanation["what_to_do"],
                explanation["band_text"], explanation["disclaimer"]]
        return [s for s in out if s]

    out += [explanation["band_text"], explanation["deciding_text"],
            explanation["disclaimer"]]
    out += [m["name"] for m in explanation["measurements"]]
    for rule in explanation["rules"]:
        out += [rule["sentence"]]
        if rule["provenance_phrasing"]:
            out.append(rule["provenance_phrasing"])
    if explanation["counterfactual"]:
        out.append(explanation["counterfactual"]["text"])
    if explanation.get("region"):
        out.append(explanation["region"]["describes"])
    out += list(explanation["advisories"])
    return [s for s in out if s]
