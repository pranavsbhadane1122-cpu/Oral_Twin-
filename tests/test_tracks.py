"""Tracks A/B/C, the screening branch, and the rules that hold them honest.

Run with: python -m unittest tests.test_tracks -v
"""

import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracks import screening as screening_mod  # noqa: E402
from src.tracks.performance import (  # noqa: E402
    SMALL_SAMPLE,
    ExternalValidation,
    Measured,
    NoneExists,
    Rate,
    TrackPerformance,
    measurement_path,
    measurements,
)
from src.tracks.pipeline import run as run_pipeline  # noqa: E402
from src.tracks.screening import (  # noqa: E402
    FLAGGED,
    NOT_FLAGGED,
    flagged_statement,
    not_flagged_statement,
    screen,
)
from src.tracks.tracks import (  # noqa: E402
    TRACK_C,
    TRACK_TITLES,
    TrackResult,
    build_tracks,
    track_a_performance,
    track_b_performance,
    track_c_performance,
)
from src.utils.config import load_config  # noqa: E402

# the single vocabulary authority, imported rather than retyped
from tests.source_scan import read_source  # noqa: E402
from tests.test_caption import CONDITION_RE  # noqa: E402

CFG = load_config()


def change_report(comparable=True):
    lesion = {
        "id": 0, "state": "matched", "comparability": "full", "match_iou": 0.9,
        "bbox_visit1": [10, 20, 30, 40], "bbox_visit2": [10, 20, 30, 40],
        "area_visit1_px": 1000, "area_visit2_px": 1400, "area_change_pct": 40.0,
        "notable_area_change": True,
        "color_shift": {"measurable": True, "magnitude": 5.0, "notable": False},
        "edge_irregularity": {"notable": False},
        "confidence": 0.8, "reliability": "high", "reason": None, "message": "",
    }
    return {
        "status": "ok" if comparable else "unreliable",
        "reason": None if comparable else "could not be lined up",
        "alignment_confidence": 0.8, "reliability": "high",
        "lesions": [lesion] if comparable else [],
        "summary": {"comparable": comparable, "any_notable_change": True},
    }


def risk_for(report):
    from src.risk.risk_rules import stratify
    return stratify(report, None, None, report["alignment_confidence"], CFG)


class TestScoreCannotExistWithoutPerformance(unittest.TestCase):

    def test_a_track_result_requires_a_performance_object(self):
        with self.assertRaises(TypeError):
            TrackResult(TRACK_C, "x", 0.5, None)          # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TrackResult(TRACK_C, "x", 0.5, "0.072")       # type: ignore[arg-type]

    def test_performance_is_a_required_positional_argument(self):
        with self.assertRaises(TypeError):
            TrackResult(TRACK_C, "x", 0.5)                # type: ignore[call-arg]

    def test_a_performance_needs_at_least_one_internal_figure(self):
        with self.assertRaises(ValueError):
            TrackPerformance(internal=(), external=NoneExists("because"))

    def test_every_built_track_carries_its_performance(self):
        for track in build_tracks(CFG, score_a=0.5, score_b=0.5, score_c=0.5):
            self.assertIsInstance(track.performance, TrackPerformance)
            self.assertTrue(track.performance.internal)


class TestExternalValidationIsAnEnum(unittest.TestCase):

    def test_the_base_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            ExternalValidation()

    def test_external_cannot_be_none(self):
        with self.assertRaises(TypeError):
            TrackPerformance(internal=(("x", Rate(0.5, 200)),), external=None)

    def test_external_cannot_be_an_empty_string(self):
        with self.assertRaises(TypeError):
            TrackPerformance(internal=(("x", Rate(0.5, 200)),), external="")

    def test_none_exists_requires_a_reason(self):
        for empty in ("", "   "):
            with self.assertRaises(ValueError):
                NoneExists(empty)

    def test_measured_requires_a_set_name_and_a_class_balance(self):
        rate = Rate(0.072, 125, 0.038, 0.131, 9)
        with self.assertRaises(ValueError):
            Measured(sensitivity=rate, set_name="", class_balance="5.5% positive")
        with self.assertRaises(ValueError):
            Measured(sensitivity=rate, set_name="SMART-OM", class_balance="  ")

    def test_none_exists_renders_as_a_sentence(self):
        text = NoneExists("No second dataset shares these labels.").render()
        self.assertTrue(text.endswith("."))
        self.assertIn("No external testing exists", text)

    def test_tracks_a_and_b_use_none_exists(self):
        for performance in (track_a_performance(CFG), track_b_performance(CFG)):
            self.assertIsInstance(performance.external, NoneExists)
            self.assertIn("cannot be run", performance.external.reason)

    def test_track_c_uses_measured(self):
        performance = track_c_performance(CFG)
        self.assertIsInstance(performance.external, Measured)
        self.assertIn("5.5%", performance.external.class_balance)


class TestSmallSamplesCarryIntervals(unittest.TestCase):

    def test_a_small_sample_rate_without_an_interval_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            Rate(value=0.5, denominator=SMALL_SAMPLE - 1)
        self.assertIn("must carry its interval", str(ctx.exception))

    def test_a_large_sample_may_omit_the_interval(self):
        Rate(value=0.5, denominator=SMALL_SAMPLE + 1)      # must not raise

    def test_every_track_a_class_renders_its_interval(self):
        for _, rate in track_a_performance(CFG).internal:
            self.assertLess(rate.denominator, SMALL_SAMPLE)
            self.assertTrue(rate.has_interval)
            self.assertIn("[", rate.render())

    def test_track_c_renders_intervals_on_both_figures(self):
        performance = track_c_performance(CFG)
        self.assertIn("[", performance.internal[0][1].render(always_interval=True))
        self.assertIn("[", performance.external.render())

    def test_a_rate_never_invents_a_count(self):
        """A mean is not a proportion of successes."""
        overlap = track_b_performance(CFG).internal[0][1]
        self.assertIsNone(overlap.numerator)
        self.assertIn("mean over", overlap.render())
        self.assertNotRegex(overlap.render(), r"\d+ of \d+")


class TestTrackBNeverFlattersItself(unittest.TestCase):

    def test_the_reported_figure_is_dice(self):
        label, rate = track_b_performance(CFG).internal[0]
        self.assertIn("Dice", label)
        self.assertAlmostEqual(rate.value, 0.362, places=2)

    def test_per_pixel_sensitivity_never_appears_without_the_ppv(self):
        """0.689 is the friendly number; 0.376 is the one that qualifies it."""
        performance = track_b_performance(CFG)
        text = " ".join(performance.render_lines())
        for form in ("0.689", "69%"):
            if form in text:
                self.assertTrue(
                    "0.376" in text or "38%" in text,
                    f"{form} appears without the PPV beside it: {text}")

    def test_the_note_states_the_failed_gate(self):
        note = track_b_performance(CFG).note
        self.assertIn("failed its acceptance gate", note)
        self.assertIn("0.70", note)


class TestBranchStatements(unittest.TestCase):

    def test_the_labels_name_a_threshold_crossing_not_a_disease(self):
        self.assertEqual(FLAGGED, "Flagged for review")
        self.assertEqual(NOT_FLAGGED, "Not flagged")
        for label in (FLAGGED, NOT_FLAGGED):
            self.assertIsNone(CONDITION_RE.search(label))
            for banned in ("cancer", "cancerous", "benign", "malignant"):
                self.assertNotIn(banned, label.lower())

    def test_the_not_flagged_statement_resolves_its_figures(self):
        data = measurements(CFG)["track_c"]["external"]["sensitivity"]
        statement = not_flagged_statement(CFG)
        self.assertIn(f"{data['numerator']} of {data['denominator']}", statement)
        self.assertIn("not evidence that nothing is present", statement)

    def test_the_flagged_statement_resolves_its_figures(self):
        ppv = measurements(CFG)["track_c"]["external"]["ppv"]["value"]
        self.assertIn(f"1 in {round(1.0 / ppv)}", flagged_statement(CFG))

    def test_the_figures_are_not_hard_coded_in_the_module(self):
        """If they were typed, retraining would leave the sentence lying."""
        source = read_source(PROJECT_ROOT / "src" / "tracks" / "screening.py")
        body = re.sub(r'"""all.*?"""', "", source, flags=re.S)
        for literal in ("9 of 125", "0.072", "1 in 10", "0.103"):
            self.assertNotIn(literal, body,
                             f"{literal} is typed into screening.py; it must "
                             f"resolve from the measurement file")

    def test_the_statements_change_when_the_measurement_changes(self):
        """The real proof that nothing is typed: move the file, move the text."""
        from src.tracks import performance as perf
        real = json.loads(measurement_path(CFG).read_text(encoding="utf-8"))
        fake = json.loads(json.dumps(real))
        fake["track_c"]["external"]["sensitivity"].update(
            {"numerator": 42, "denominator": 999})
        perf._load.cache_clear()
        original = perf._load
        perf._load = lambda _p: fake
        try:
            self.assertIn("42 of 999", not_flagged_statement(CFG))
        finally:
            perf._load = original
            perf._load.cache_clear()

    def test_no_branch_string_names_a_condition(self):
        for text in (flagged_statement(CFG), not_flagged_statement(CFG),
                     screening_mod.QUESTION, screening_mod.REFERRAL_LINE,
                     screening_mod.CLINICIAN_LINE, FLAGGED, NOT_FLAGGED):
            found = CONDITION_RE.search(text)
            self.assertIsNone(found,
                              f"{found.group(0) if found else ''!r} in: {text}")


class TestScreeningBranch(unittest.TestCase):

    def performance(self):
        return track_c_performance(CFG)

    def result(self, score):
        track = TrackResult(TRACK_C, TRACK_TITLES[TRACK_C], score,
                            self.performance())
        return screen(CFG, score, track)

    def test_the_threshold_comes_from_config(self):
        self.assertEqual(self.result(0.9).threshold,
                         CFG["screening"]["flag_threshold"])

    def test_the_threshold_is_not_a_literal_in_the_module(self):
        source = read_source(PROJECT_ROOT / "src" / "tracks" / "screening.py")
        self.assertIn('cfg["screening"]["flag_threshold"]', source)

    def test_above_the_threshold_is_flagged(self):
        self.assertTrue(self.result(0.99).flagged)
        self.assertEqual(self.result(0.99).label, FLAGGED)

    def test_below_the_threshold_is_not_flagged(self):
        self.assertFalse(self.result(0.01).flagged)
        self.assertEqual(self.result(0.01).label, NOT_FLAGGED)

    def test_the_non_informative_statement_renders_on_every_not_flagged(self):
        for score in (0.0, 0.1, 0.25, CFG["screening"]["flag_threshold"] - 1e-9):
            text = "\n".join(self.result(score).render_lines())
            self.assertIn("not evidence that nothing is present", text,
                          f"missing on score {score}")

    def test_both_branches_carry_the_referral_line(self):
        for score in (0.01, 0.99):
            self.assertIn(screening_mod.REFERRAL_LINE,
                          "\n".join(self.result(score).render_lines()))

    def test_both_branches_carry_the_clinician_line(self):
        for score in (0.01, 0.99):
            self.assertIn("A dentist should look at the photographs",
                          "\n".join(self.result(score).render_lines()))


class TestBothBranchesContinueIntoTheChangePipeline(unittest.TestCase):

    def test_the_explanation_is_present_on_a_flagged_result(self):
        report = change_report()
        combined = run_pipeline(CFG, report, risk_for(report), score_c=0.99)
        self.assertTrue(combined.screening.flagged)
        self.assertEqual(combined.explanation["kind"], "comparable")
        self.assertIn("WHAT WAS MEASURED", "\n".join(combined.render_lines()))

    def test_the_explanation_is_present_on_a_not_flagged_result(self):
        report = change_report()
        combined = run_pipeline(CFG, report, risk_for(report), score_c=0.01)
        self.assertFalse(combined.screening.flagged)
        self.assertEqual(combined.explanation["kind"], "comparable")
        self.assertIn("WHAT WAS MEASURED", "\n".join(combined.render_lines()))

    def test_the_explanation_is_identical_on_both_branches(self):
        """Screening must not reach the change measurement at all."""
        report = change_report()
        risk = risk_for(report)
        flagged = run_pipeline(CFG, report, risk, score_c=0.99).explanation
        not_flagged = run_pipeline(CFG, report, risk, score_c=0.01).explanation
        self.assertEqual(flagged, not_flagged)

    def test_the_change_pipeline_runs_with_no_screening_score_at_all(self):
        report = change_report()
        combined = run_pipeline(CFG, report, risk_for(report))
        self.assertIsNone(combined.screening)
        self.assertEqual(combined.explanation["kind"], "comparable")

    def test_a_refusal_still_refuses_on_both_branches(self):
        report = change_report(comparable=False)
        for score in (0.01, 0.99):
            combined = run_pipeline(CFG, report, risk_for(report), score_c=score)
            self.assertEqual(combined.explanation["kind"], "refused")


class TestVocabularyOnEveryNewString(unittest.TestCase):

    def all_strings(self):
        out = [screening_mod.QUESTION, screening_mod.REFERRAL_LINE,
               screening_mod.CLINICIAN_LINE, FLAGGED, NOT_FLAGGED,
               flagged_statement(CFG), not_flagged_statement(CFG)]
        for performance in (track_a_performance(CFG), track_b_performance(CFG),
                            track_c_performance(CFG)):
            out += performance.render_lines()
        for track in build_tracks(CFG, score_a=0.5, score_b=0.5, score_c=0.5):
            out += track.render_lines()
        report = change_report()
        for score in (0.01, 0.99):
            out += run_pipeline(CFG, report, risk_for(report),
                                score_a=0.5, score_b=0.5,
                                score_c=score).render_lines()
        return [s for s in out if s and s.strip()]

    def test_no_track_or_branch_string_contains_a_forbidden_term(self):
        for text in self.all_strings():
            found = CONDITION_RE.search(text)
            self.assertIsNone(
                found,
                f"forbidden term {found.group(0) if found else ''!r} in: {text[:140]}")

    def test_the_guard_is_not_vacuous_here(self):
        self.assertIsNotNone(CONDITION_RE.search("this shows leukoplakia"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
