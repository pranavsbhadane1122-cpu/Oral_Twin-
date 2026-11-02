"""Tests for the explanation layer.

Run with: python -m unittest tests.test_explain -v
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import claims  # noqa: E402
from src.explain.explain import (  # noqa: E402
    explain,
    explain_comparable,
    explain_refused,
    unregistered_thresholds,
)
from src.explain.render import explanation_strings, render  # noqa: E402
from src.risk.risk_rules import (  # noqa: E402
    ELEVATED,
    LOW,
    MODERATE,
    ORDER,
    RULE_SPECS,
    stratify,
)
from src.utils.config import load_config  # noqa: E402

# The single authority. Imported, never retyped - that is the whole point.
from tests.test_caption import CONDITION_RE, CONDITION_WORDS  # noqa: E402

CFG = load_config()


def lesion(area1=1000, area2=1000, state="matched", colour_mag=5.0,
           colour_notable=False, bbox=(10, 20, 30, 40)):
    change = (area2 - area1) / area1 * 100.0 if area1 else 0.0
    return {
        "id": 0, "state": state, "comparability": "full",
        "match_iou": 0.9,
        "bbox_visit1": list(bbox), "bbox_visit2": list(bbox),
        "area_visit1_px": area1, "area_visit2_px": area2,
        "area_change_pct": round(change, 2),
        "notable_area_change": abs(change) >= 10,
        "color_shift": {"measurable": True, "magnitude": colour_mag,
                        "notable": colour_notable},
        "edge_irregularity": {"notable": True},
        "confidence": 0.8, "reliability": "high", "reason": None,
        "message": "",
    }


def report(lesions=None, confidence=0.8, status="ok", comparable=True, reason=None):
    lesions = [lesion()] if lesions is None else lesions
    return {
        "status": status, "reason": reason,
        "alignment_confidence": confidence,
        "reliability": "high",
        "lesions": lesions,
        "summary": {"comparable": comparable, "any_notable_change": False},
    }


def explained(change_report):
    risk = stratify(change_report, None, None,
                    change_report.get("alignment_confidence", 0.0), CFG)
    return explain(change_report, risk, CFG), risk


class TestComparableCompleteness(unittest.TestCase):

    def test_every_field_is_populated_for_a_comparable_result(self):
        e, _ = explained(report([lesion(1000, 1400)]))
        self.assertEqual(e["kind"], "comparable")
        for field in ("band", "band_text", "measurements", "deciding_text",
                      "deciding_rules", "rules", "counterfactual", "region",
                      "disclaimer"):
            self.assertIn(field, e)
            self.assertTrue(e[field] not in (None, "", []),
                            f"{field} must not be empty on a comparable result")

    def test_measurements_use_plain_language_not_variable_names(self):
        e, _ = explained(report([lesion(1000, 1400)]))
        for m in e["measurements"]:
            self.assertNotEqual(m["name"], m["key"],
                                f"{m['key']} has no plain-language name")
            self.assertNotIn("_", m["name"], f"variable name leaked: {m['name']}")
            self.assertTrue(m["name"][0].islower() or m["name"][0].isupper())

    def test_a_band_with_nothing_crossed_still_explains_itself(self):
        # a Low band assigned because nothing fired is still a decision
        e, risk = explained(report([lesion(1000, 1000)]))
        self.assertEqual(risk["band"], LOW)
        self.assertEqual(e["deciding_rules"], [])
        self.assertIn("No measurement crossed", e["deciding_text"])
        self.assertIn("decision in its own right", e["deciding_text"])

    def test_two_rules_at_the_same_level_are_both_named(self):
        # growth past moderate AND a notable colour change: both Moderate
        both = lesion(1000, 1150, colour_mag=40.0, colour_notable=True)
        e, risk = explained(report([both]))
        self.assertGreaterEqual(len(e["deciding_rules"]), 2)
        self.assertIn("Two measurements crossed", e["deciding_text"])

    def test_the_rule_sentence_is_generated_from_the_spec(self):
        e, _ = explained(report([lesion(1000, 1400)]))
        deciding = [r for r in e["rules"] if r["deciding"]][0]
        spec = RULE_SPECS[deciding["rule"]]
        # identical to what the spec generates: no second copy anywhere
        self.assertEqual(deciding["sentence"], spec.sentence(CFG))
        self.assertIn(str(int(spec.threshold(CFG))), deciding["sentence"])


class TestThresholdProvenance(unittest.TestCase):

    def test_every_rule_threshold_is_registered(self):
        """An unregistered threshold deciding a band is what CLAIMS.md exists
        to catch, so this is the test the register was built for."""
        missing = unregistered_thresholds()
        self.assertEqual(
            missing, [],
            f"these rule thresholds have no entry in docs/CLAIMS.md: {missing}")

    def test_provenance_state_is_reported_on_every_rule(self):
        e, _ = explained(report([lesion(1000, 1400)]))
        for rule in e["rules"]:
            if rule["threshold_key"] is None:
                continue
            self.assertIn(rule["provenance_state"],
                          (claims.CALIBRATED, claims.ARTEFACT_ONLY,
                           claims.PROVISIONAL, claims.UNREGISTERED))
            self.assertTrue(rule["provenance_phrasing"])

    def test_relative_color_thresh_is_reported_as_artefact_only(self):
        """It has 97 lesions behind it but was never logged as a run, so the
        explanation must not round it up to calibrated."""
        prov = claims.provenance("delta.relative_color_thresh")
        self.assertEqual(prov["state"], claims.ARTEFACT_ONLY)
        self.assertIn("never written up", prov["phrasing"])

    def test_growth_thresholds_are_reported_as_provisional(self):
        for key in ("risk.growth_moderate_pct", "risk.growth_elevated_pct"):
            self.assertEqual(claims.provenance(key)["state"], claims.PROVISIONAL)

    def test_an_unknown_threshold_is_reported_not_hidden(self):
        prov = claims.provenance("nonexistent.threshold")
        self.assertEqual(prov["state"], claims.UNREGISTERED)
        self.assertIn("NOT REGISTERED", prov["phrasing"])

    def test_provenance_is_not_hard_coded_in_this_module(self):
        """The register is the authority; the module must not restate it."""
        source = (PROJECT_ROOT / "src" / "explain" / "explain.py").read_text("utf-8")
        for word in ("97 no-change", "color_calibration.json", "p95"):
            self.assertNotIn(word, source,
                             f"explain.py restates provenance ({word}) instead "
                             f"of reading docs/CLAIMS.md")


class TestCounterfactual(unittest.TestCase):

    def test_the_counterfactual_applied_actually_flips_the_band(self):
        """Invert the rule, apply it, and confirm the band really changes."""
        original = report([lesion(1000, 1400)])          # +40%, Elevated
        e, risk = explained(original)
        self.assertEqual(risk["band"], ELEVATED)
        threshold = e["counterfactual"]["threshold"]

        # construct the same pair just below the deciding threshold
        just_below = threshold - 1.0
        area2 = int(1000 * (1 + just_below / 100.0))
        flipped_report = report([lesion(1000, area2)])
        _, flipped_risk = explained(flipped_report)

        self.assertLess(ORDER[flipped_risk["band"]], ORDER[risk["band"]],
                        f"going below {threshold} should lower the band, "
                        f"got {flipped_risk['band']}")

    def test_the_counterfactual_names_the_margin(self):
        e, _ = explained(report([lesion(1000, 1400)]))
        cf = e["counterfactual"]
        self.assertAlmostEqual(cf["measured"] - cf["threshold"], cf["margin"], 2)
        self.assertGreater(cf["margin"], 0)

    def test_a_provisional_threshold_says_so_in_the_counterfactual(self):
        e, _ = explained(report([lesion(1000, 1400)]))
        text = e["counterfactual"]["text"]
        self.assertIn("reasoned default", text,
                      "a counterfactual on a provisional threshold must say so")

    def test_no_counterfactual_when_nothing_crossed(self):
        e, _ = explained(report([lesion(1000, 1000)]))
        self.assertIsNone(e["counterfactual"])


class TestRegion(unittest.TestCase):

    def test_the_region_carries_coordinates(self):
        e, _ = explained(report([lesion(1000, 1400, bbox=(11, 22, 33, 44))]))
        self.assertEqual(e["region"]["bbox"], [11, 22, 33, 44])

    def test_the_region_is_described_as_a_computation_not_attention(self):
        e, _ = explained(report([lesion(1000, 1400)]))
        describes = e["region"]["describes"]
        self.assertIn("where this measurement was computed", describes)
        self.assertIn("not where a model looked", describes)


class TestRefusedIsQualitativeOnly(unittest.TestCase):

    def test_a_refusal_explanation_contains_no_digit(self):
        for check in ("alignment_failed", "low_confidence",
                      "partial_out_of_frame", "not_comparable"):
            e = explain_refused(check)
            for text in explanation_strings(e):
                self.assertTrue(all(not c.isdigit() for c in text),
                                f"{check} explanation contains a digit: {text}")

    def test_a_refusal_explanation_has_no_numeric_field(self):
        e = explain_refused("low_confidence")
        for key, value in e.items():
            self.assertNotIsInstance(value, (int, float),
                                     f"refusal explanation field {key} is numeric")

    def test_a_refusal_does_not_leak_the_confidence(self):
        refused = report(status="unreliable", comparable=False,
                         reason="alignment confidence 0.12 is below the floor 0.30",
                         confidence=0.12)
        e, _ = explained(refused)
        self.assertEqual(e["kind"], "refused")
        rendered = render(e)
        self.assertNotIn("0.12", rendered)
        self.assertTrue(all(not c.isdigit() for c in rendered),
                        f"a refusal must print no digits at all: {rendered}")

    def test_a_refusal_says_which_check_failed_and_what_to_do(self):
        e = explain_refused("partial_out_of_frame")
        self.assertIn("Too little", e["check_failed"])
        self.assertIn("Retake", e["what_to_do"])


class TestVocabulary(unittest.TestCase):
    """One guard, imported from the single authority."""

    def test_no_explanation_string_contains_a_forbidden_term(self):
        cases = [report([lesion(1000, 1400)]),
                 report([lesion(1000, 1000)]),
                 report([lesion(1000, 1150, colour_mag=40.0, colour_notable=True)]),
                 report([lesion(1000, 1400, state="new")])]
        for change_report in cases:
            e, _ = explained(change_report)
            for text in explanation_strings(e) + [render(e)]:
                found = CONDITION_RE.search(text)
                self.assertIsNone(
                    found,
                    f"forbidden term {found.group(0) if found else ''!r} in: {text[:140]}")

    def test_refusal_strings_are_clean_too(self):
        for check in ("alignment_failed", "low_confidence",
                      "partial_out_of_frame", "not_comparable"):
            e = explain_refused(check)
            for text in explanation_strings(e) + [render(e)]:
                self.assertIsNone(CONDITION_RE.search(text))

    def test_the_guard_is_imported_not_retyped(self):
        """A second copy of the list would be the drift CLAIMS.md warns about.

        Checked by looking for a DEFINITION of the list rather than for any
        single term: an earlier version of this test searched its own source
        for a term and found the literal it was searching with.
        """
        source = (PROJECT_ROOT / "tests" / "test_explain.py").read_text("utf-8")
        self.assertIn("from tests.test_caption import", source)
        self.assertIsNone(
            re.search(r"CONDITION_WORDS\s*=\s*\[", source),
            "this file defines its own CONDITION_WORDS; it must import the "
            "single authority in tests/test_caption.py instead")


class TestGeneratedKotlinVocabulary(unittest.TestCase):

    GENERATED = (PROJECT_ROOT / "android" / "app" / "src" / "main" / "java"
                 / "com" / "oraltwin" / "app" / "guard" / "ForbiddenVocabulary.kt")

    def test_the_generated_file_exists(self):
        self.assertTrue(self.GENERATED.exists(),
                        "run python scripts/generate_kotlin_vocabulary.py")

    def test_the_generated_list_matches_the_python_source_term_for_term(self):
        text = self.GENERATED.read_text(encoding="utf-8")
        block = text.split("listOf(", 1)[1].split(")", 1)[0]
        kotlin_terms = re.findall(r'"([^"]+)"', block)
        self.assertEqual(
            kotlin_terms, list(CONDITION_WORDS),
            "the generated Kotlin vocabulary has drifted from "
            "tests/test_caption.py; re-run the generator")

    def test_the_generated_file_is_marked_do_not_edit(self):
        text = self.GENERATED.read_text(encoding="utf-8")
        self.assertIn("GENERATED FILE - DO NOT EDIT BY HAND", text)

    def test_the_generator_reports_the_file_as_up_to_date(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts"
                                 / "generate_kotlin_vocabulary.py"), "--check"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(result.returncode, 0,
                         f"generator reports stale: {result.stdout}{result.stderr}")

    def test_no_term_count_is_written_by_hand_anywhere(self):
        """The count was reported as 46 and is 42. A hand-written count rots."""
        actual = len(CONDITION_WORDS)
        for path in (self.GENERATED,
                     PROJECT_ROOT / "scripts" / "generate_kotlin_vocabulary.py",
                     PROJECT_ROOT / "src" / "explain" / "explain.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(f"{actual} terms", text.replace("{len(terms)} terms", ""),
                             f"{path.name} writes the term count by hand")


class TestRendersFromTheSameResult(unittest.TestCase):

    def test_the_explanation_recomputes_nothing(self):
        """It must read the risk result, not re-derive the band."""
        change_report = report([lesion(1000, 1400)])
        risk = stratify(change_report, None, None, 0.8, CFG)
        # hand the explainer a DIFFERENT band than the rules imply; it must
        # report what it was given, because the UI is showing that
        risk_with_odd_band = {**risk, "band": MODERATE}
        e = explain_comparable(change_report, risk_with_odd_band, CFG)
        self.assertEqual(e["band"], MODERATE,
                         "the explanation must report the band it was handed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
