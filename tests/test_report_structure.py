"""Report structure: the order IS the claim, and the numbers must not move.

Two jobs.

ORDERING. The band appears once, as the conclusion of reasoning, and nowhere
above it. An output that states a verdict and then justifies it is a prediction
with commentary; the words can be identical and the object different.

NO SILENT DRIFT. tests/baselines/ holds reports rendered by the code as it stood
before the restructure. Every number, every band and every deciding measurement
must survive a reordering unchanged. A reordering that moves a number is not a
reordering.

Run with: python -m unittest tests.test_report_structure -v
"""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_report_baselines import (  # noqa: E402
    BASELINE_DIR,
    COMPARABLE,
    REFUSED,
    render,
)
from src.explain.render import DOES_NOT_SAY, HEADINGS  # noqa: E402
from src.risk.risk_rules import ELEVATED, LOW, MODERATE, NOT_ASSESSED  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CFG = load_config()
BANDS = (LOW, MODERATE, ELEVATED, NOT_ASSESSED)

MARKER = "# ---8<--- report starts here\n"


def baseline(name):
    text = (BASELINE_DIR / name).read_text(encoding="utf-8")
    return text.split(MARKER, 1)[1] if MARKER in text else text


def rendered_comparable():
    return {pair: render(pair, CFG)[0] for pair in COMPARABLE}


def rendered_refused():
    return render(REFUSED, CFG, induce_refusal=True)[0]


def numbers_in(text):
    """Every numeric token, so a reordering cannot quietly change one."""
    return re.findall(r"-?\d+\.?\d*%?", text)


class TestBandAppearsOnlyAsConclusion(unittest.TestCase):

    def test_the_band_appears_exactly_once(self):
        for pair, text in rendered_comparable().items():
            band = render(pair, CFG)[1]["band"]
            # word boundary and case sensitive: "Low" must not match "lower",
            # and a rule sentence saying "elevated" would be a leak
            hits = re.findall(rf"\b{re.escape(band)}\b", text)
            self.assertEqual(
                len(hits), 1,
                f"{pair}: band {band!r} appears {len(hits)} times; it must "
                f"appear once, as the conclusion")

    def test_the_first_occurrence_of_the_band_is_inside_THEREFORE(self):
        for pair, text in rendered_comparable().items():
            band = render(pair, CFG)[1]["band"]
            therefore = text.index(HEADINGS["therefore"])
            first = re.search(rf"\b{re.escape(band)}\b", text).start()
            self.assertGreater(
                first, therefore,
                f"{pair}: the band appears before {HEADINGS['therefore']}, so "
                f"the report delivers a verdict and then justifies it")

    def test_no_band_name_appears_before_THEREFORE_in_any_case(self):
        """Even lowercase: a rule reading 'counts as elevated' pre-announces."""
        for pair, text in rendered_comparable().items():
            head = text[:text.index(HEADINGS["therefore"])]
            for band in BANDS:
                self.assertIsNone(
                    re.search(rf"\b{re.escape(band)}\b", head, re.IGNORECASE),
                    f"{pair}: {band!r} appears before THEREFORE")

    def test_the_reasoning_sections_precede_the_conclusion(self):
        order = ["looked_at", "measured", "judged", "decided", "therefore"]
        for pair, text in rendered_comparable().items():
            positions = [text.index(HEADINGS[key]) for key in order]
            self.assertEqual(positions, sorted(positions),
                             f"{pair}: sections are out of order")


class TestRequiredSections(unittest.TestCase):

    def test_sections_one_and_seven_are_in_every_comparable_report(self):
        for pair, text in rendered_comparable().items():
            self.assertIn(HEADINGS["looked_at"], text, pair)
            self.assertIn(HEADINGS["not_say"], text, pair)

    def test_sections_one_and_seven_are_in_the_refused_report(self):
        text = rendered_refused()
        self.assertIn(HEADINGS["looked_at"], text)
        self.assertIn(HEADINGS["not_say"], text)

    def test_the_does_not_say_lines_come_from_one_constant(self):
        texts = list(rendered_comparable().values()) + [rendered_refused()]
        for text in texts:
            for line in DOES_NOT_SAY:
                self.assertIn(line, text,
                              "every report type must carry every line")

    def test_a_refused_report_has_no_therefore_section(self):
        # there is no conclusion to draw, so there must be no section claiming one
        self.assertNotIn(HEADINGS["therefore"], rendered_refused())


class TestRiskStatement(unittest.TestCase):

    def statements(self):
        from src.explain.explain import RISK_STATEMENTS, risk_statement
        out = []
        for band in RISK_STATEMENTS:
            out.append(risk_statement(band))
            out.append(risk_statement(band, "3 March 2026"))
        return out

    def test_the_risk_statement_names_no_condition(self):
        from tests.test_caption import CONDITION_RE
        for statement in self.statements():
            found = CONDITION_RE.search(statement)
            self.assertIsNone(
                found,
                f"risk statement names a condition "
                f"({found.group(0) if found else ''}): {statement}")

    def test_the_risk_statement_states_no_likelihood(self):
        probability_words = ["likely", "probable", "probably", "chance",
                             "risk of", "odds", "percent", "probability",
                             "unlikely", "possible", "possibly"]
        for statement in self.statements():
            for word in probability_words:
                self.assertIsNone(
                    re.search(rf"\b{re.escape(word)}\b", statement, re.I),
                    f"risk statement says {word!r}: {statement}")

    def test_the_risk_statement_prescribes_no_clinical_action(self):
        actions = ["biopsy", "treat", "treatment", "monitor", "refer",
                   "referral", "prescribe", "surgery", "test for", "screen for"]
        for statement in self.statements():
            for word in actions:
                self.assertIsNone(
                    re.search(rf"\b{re.escape(word)}\b", statement, re.I),
                    f"risk statement prescribes {word!r}: {statement}")

    def test_the_risk_statement_names_no_body_system(self):
        systems = ["cardiovascular", "diabetes", "heart", "immune", "blood",
                   "systemic", "lymph"]
        for statement in self.statements():
            for word in systems:
                self.assertIsNone(
                    re.search(rf"\b{re.escape(word)}\b", statement, re.I),
                    f"risk statement names {word!r}: {statement}")

    def test_the_word_checks_are_not_vacuous(self):
        self.assertIsNotNone(re.search(r"\blikely\b", "this is likely", re.I))
        self.assertIsNotNone(re.search(r"\bbiopsy\b", "needs a biopsy", re.I))


class TestNoUnfilledPlaceholders(unittest.TestCase):

    # "None" and "nan" are matched only where they indicate a LEAKED
    # VALUE, not as English. An earlier version used \bNone\b and flagged
    # the correct sentence "None reached it." - the pattern was wrong,
    # not the prose.
    PATTERNS = [r"\{[a-zA-Z_]+\}", r"%s\b", r"%\([a-zA-Z_]+\)",
                r"<[a-z_]+>", r"TODO", r"FIXME",
                r":\s*None\b", r"\bNone\s*(?:pixels|px|%)",
                r"\(None\b", r"\bNone\)",
                r":\s*nan\b"]

    def test_no_rendered_line_contains_a_placeholder(self):
        texts = dict(rendered_comparable())
        texts["refused"] = rendered_refused()
        for name, text in texts.items():
            for lineno, line in enumerate(text.splitlines(), 1):
                for pattern in self.PATTERNS:
                    self.assertIsNone(
                        re.search(pattern, line),
                        f"{name} line {lineno} has an unfilled placeholder "
                        f"({pattern}): {line}")

    def test_the_risk_statement_fills_or_rewrites_its_date(self):
        from src.explain.explain import RISK_STATEMENTS, risk_statement
        for band in RISK_STATEMENTS:
            for date in (None, "3 March 2026"):
                statement = risk_statement(band, date)
                self.assertNotIn("{", statement)
                self.assertNotIn("}", statement)


class TestRefusedHasNoDigits(unittest.TestCase):
    """A refusal must carry no number a reader could take as a measurement.

    THE SPEC CONFLICTS WITH ITSELF HERE, and the conflict is resolved in the
    open rather than by quietly broadening the rule.

      section 1 must render on refused reports and must show "the two dates
      compared". A date contains digits.

      so "a refused report contains no digit" cannot hold once dates are
      supplied. It is not a tightenable rule; it is unsatisfiable.

    Three exclusions, each named and each justified:

      the report title     carries a pair identifier (pair_0012). An internal
                           name, not a measurement.
      the BACKGROUND block carries two citation years (2020, 2012). A
                           bibliography, not a measurement, and present on
                           every report type.
      the section 1 dates  required by the spec itself.

    Everything else must be digit-free, and
    test_the_refusal_body_has_no_digit_at_all below holds the strict rule over
    the part the explanation layer actually produces.
    """

    def _auditable_lines(self, text):
        out = []
        in_background = False
        for line in text.splitlines():
            if line.startswith("BACKGROUND"):
                in_background = True
                continue
            if in_background:
                if line.strip() == "":
                    in_background = False
                continue
            if line.startswith("ORALTWIN VISIT COMPARISON"):
                continue
            if line.strip().startswith("Comparing the photo from"):
                continue
            out.append(line)
        return out

    def test_a_refused_report_contains_no_measurement_digit(self):
        text = rendered_refused()
        offenders = [line for line in self._auditable_lines(text)
                     if any(c.isdigit() for c in line)]
        self.assertEqual(
            offenders, [],
            f"a refusal must carry no number a reader could take as a "
            f"measurement. Found: {offenders}")

    def test_the_confidence_value_never_appears_on_a_refusal(self):
        """The specific leak the baseline caught: the pre-restructure report
        printed "confidence 0.62 is below the floor 0.99" twice."""
        text = rendered_refused()
        self.assertNotIn("confidence 0.", text)
        self.assertNotIn("below the floor", text)

    def test_the_refusal_body_has_no_digit_at_all(self):
        """The strict rule, over what src/explain actually produces."""
        from src.explain.explain import explain_refused
        from src.explain.render import render as render_explanation
        for check in ("alignment_failed", "low_confidence",
                      "partial_out_of_frame", "not_comparable"):
            text = render_explanation(explain_refused(check))
            self.assertTrue(all(not c.isdigit() for c in text),
                            f"{check}: {text}")


class TestNoSilentDriftFromBaseline(unittest.TestCase):
    """The baselines were rendered before the restructure.

    A reordering may move a section, add a section, or change wording that was
    deliberately changed. It may NOT change a number, a band, or which
    measurement was named as deciding.
    """

    def test_every_number_survives_the_reordering(self):
        for pair in COMPARABLE:
            before = sorted(numbers_in(baseline(f"report_{pair}.txt")))
            after = sorted(numbers_in(render(pair, CFG)[0]))
            # the restructure removed duplicated restatements of the same
            # values, so after must be a SUBSET of before - never a new number
            new = [n for n in set(after) if n not in set(before)]
            self.assertEqual(
                new, [],
                f"{pair}: the reordering introduced numbers not in the "
                f"baseline: {new}. A reordering that changes a number is a bug.")

    def test_the_band_is_unchanged_for_every_pair(self):
        for pair in COMPARABLE:
            before = baseline(f"report_{pair}.txt")
            band = render(pair, CFG)[1]["band"]
            self.assertIn(
                f"SCREENING BAND: {band}", before,
                f"{pair}: the band changed from the baseline, which is a bug "
                f"rather than a reordering")

    def test_the_deciding_measurement_is_unchanged(self):
        """The baseline named fired rules as [RULE_ID]; the same rules must
        still be the ones that fired."""
        for pair in COMPARABLE:
            before = baseline(f"report_{pair}.txt")
            baseline_rules = set(re.findall(r"\[([A-Z_]+)\]", before))
            fired = {r["rule"] for r in render(pair, CFG)[1]["rules_fired"]}
            self.assertEqual(
                baseline_rules, fired,
                f"{pair}: the rules that fired changed from the baseline "
                f"({baseline_rules} -> {fired})")

    def test_the_baselines_exist(self):
        for pair in COMPARABLE:
            self.assertTrue((BASELINE_DIR / f"report_{pair}.txt").exists())
        self.assertTrue((BASELINE_DIR / f"report_{REFUSED}_refused.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
