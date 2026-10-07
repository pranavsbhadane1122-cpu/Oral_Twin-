"""Tests for the caption layer.

The central test is the negative one: no disease name may appear in anything
this layer generates. The classifier failed its Grad-CAM gate and the lesion
segmenter failed its Dice gate, so there is no validated model behind any
clinical claim. A caption that names a condition would be inventing authority
the project does not have, and the regex below is what stops that happening by
accident during a later edit.

The vocabulary is deliberately wider than the datasets' own labels: it includes
conditions nobody has trained on, because the risk is someone adding a helpful
phrase, not someone wiring up a classifier.

Run with: python -m unittest tests.test_caption -v
"""

import copy
import re
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.caption import compare as compare_caption  # noqa: E402
from src.caption import describe as describe_photo  # noqa: E402
from src.caption.render import LIMITATIONS, render  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CONDITION_WORDS = [
    # the datasets' own labels
    "caries", "calculus", "gingivitis", "leukoplakia", "erythroplakia",
    "lichen planus", "oral cancer", "carcinoma", "opmd", "malignant",
    "premalignant", "dysplasia", "tumour", "tumor", "neoplasm",
    # conditions nobody trained on - the realistic failure is a helpful phrase
    "ulcer", "abscess", "cavity decay", "decay", "periodontitis", "gum disease",
    "infection", "inflammation", "lesion", "plaque", "tartar", "stomatitis",
    "candidiasis", "thrush", "cyst", "polyp", "melanoma", "biopsy",
    # severity and diagnostic verbs
    "benign", "diagnos", "disease", "healthy mouth", "suspicious", "severe",
    "mild case", "stage ", "grade ",
]
CONDITION_RE = re.compile("|".join(re.escape(w) for w in CONDITION_WORDS),
                          re.IGNORECASE)


def fake_change_report(confidence=0.75, status="ok", comparable=True,
                       lesions=None, reason=None):
    return {
        "status": status,
        "reason": reason,
        "alignment_confidence": confidence,
        "reliability": "moderate",
        "lesions": lesions if lesions is not None else [],
        "summary": {"comparable": comparable, "any_notable_change": False},
    }


def full_lesion(**overrides):
    lesion = {
        "id": 0, "state": "matched", "comparability": "full",
        "area_change_pct": -1.45, "notable_area_change": False,
        "color_shift": {"notable": False}, "edge_irregularity": {"notable": False},
        "confidence": 0.63, "reliability": "moderate", "reason": None,
    }
    lesion.update(overrides)
    return lesion


def blank_description(**overrides):
    description = {
        "view": describe_photo.VIEW_INTRAORAL,
        "mouth_extent": {"cavity_model": 0.8, "hsv_heuristic": 0.85,
                         "agreed": True, "fraction": 0.825, "reason": None},
        "focus": {"band": describe_photo.USABLE, "variance_of_laplacian": 300.0},
        "framing": {"clipped": False, "border_share": 0.01},
        "usable": True,
        "shape": [256, 256],
    }
    description.update(overrides)
    return description


class TestNoDiseaseNames(unittest.TestCase):
    """Nothing this layer generates may name a condition."""

    def assert_clean(self, text, context=""):
        found = CONDITION_RE.search(text)
        if found is not None:
            self.fail(f"condition word {found.group(0)!r} appeared in "
                      f"{context}: {text!r}")

    def test_limitations_sentence_itself_is_clean(self):
        # "dental conditions" is allowed as a denial; nothing names a condition
        self.assert_clean(LIMITATIONS, "LIMITATIONS")

    def test_every_photograph_sentence_is_clean(self):
        for view in (describe_photo.VIEW_INTRAORAL, describe_photo.VIEW_PARTIAL,
                     describe_photo.VIEW_NONE, describe_photo.VIEW_UNCERTAIN):
            for band in (describe_photo.USABLE, describe_photo.MARGINAL,
                         describe_photo.POOR):
                for clipped in (True, False):
                    description = blank_description(
                        view=view, focus={"band": band, "variance_of_laplacian": 1.0},
                        framing={"clipped": clipped, "border_share": 0.5})
                    for line in describe_photo.sentences(description):
                        self.assert_clean(line, f"view={view} band={band}")

    def test_every_comparison_sentence_is_clean(self):
        cases = [
            fake_change_report(lesions=[full_lesion()]),
            fake_change_report(lesions=[full_lesion(area_change_pct=37.0,
                                                    notable_area_change=True)]),
            fake_change_report(lesions=[full_lesion(
                comparability="partial_out_of_frame",
                reason="partial_out_of_frame")]),
            fake_change_report(confidence=0.1, status="unreliable", comparable=False,
                               reason="alignment confidence 0.10 is below the floor"),
            fake_change_report(comparable=True, lesions=[]),
        ]
        for report in cases:
            comparison = compare_caption.summarise(report)
            for line in compare_caption.sentences(comparison):
                self.assert_clean(line, "comparison")

    def test_full_rendered_report_is_clean(self):
        comparison = compare_caption.summarise(
            fake_change_report(lesions=[full_lesion()]))
        self.assert_clean(render(blank_description(), comparison), "full report")

    def test_the_guard_actually_catches_something(self):
        """Non-vacuity: the regex must fire on a sentence that names a condition."""
        self.assertIsNotNone(CONDITION_RE.search("This looks like early caries."))
        self.assertIsNotNone(CONDITION_RE.search("Possible gingivitis detected."))
        with self.assertRaises(AssertionError):
            self.assert_clean("signs of leukoplakia", "deliberate")


class TestLimitationsAlwaysPresent(unittest.TestCase):

    def test_present_with_a_comparison(self):
        comparison = compare_caption.summarise(
            fake_change_report(lesions=[full_lesion()]))
        self.assertIn(LIMITATIONS, render(blank_description(), comparison))

    def test_present_without_a_comparison(self):
        self.assertIn(LIMITATIONS, render(blank_description(), None))

    def test_present_on_the_refusal_path(self):
        comparison = compare_caption.summarise(
            fake_change_report(confidence=0.05, status="unreliable",
                               comparable=False, reason="below the floor"))
        self.assertIn(LIMITATIONS, render(blank_description(), comparison))

    def test_present_when_the_photograph_is_unusable(self):
        description = blank_description(
            view=describe_photo.VIEW_NONE, usable=False,
            focus={"band": describe_photo.POOR, "variance_of_laplacian": 3.0})
        self.assertIn(LIMITATIONS, render(description, None))

    def test_it_is_the_last_thing_said(self):
        text = render(blank_description(), None)
        self.assertTrue(text.rstrip().endswith(LIMITATIONS))


class TestLowConfidenceRefusal(unittest.TestCase):

    def setUp(self):
        self.comparison = compare_caption.summarise(
            fake_change_report(confidence=0.08, status="unreliable", comparable=False,
                               reason="alignment confidence 0.08 is below the floor "
                                      "0.35"))
        self.lines = compare_caption.sentences(self.comparison)
        self.text = "\n".join(self.lines)

    def test_it_refuses(self):
        self.assertEqual(self.comparison["state"], compare_caption.FAILED)
        self.assertIn("could not be compared", self.text)

    def test_it_gives_a_reason(self):
        self.assertIn("Reason:", self.text)
        self.assertIn("below the floor", self.text)

    def test_it_asks_for_a_retake(self):
        self.assertIn("retake", self.text.lower())

    def test_it_reports_no_change_figures(self):
        self.assertEqual(self.comparison["regions"], [])
        self.assertNotIn("%", self.text)
        self.assertNotIn("larger", self.text)
        self.assertNotIn("smaller", self.text)

    def test_a_refusal_does_not_read_like_no_change(self):
        """The failure mode worth guarding: 'could not check' vs 'nothing changed'."""
        self.assertNotIn("No change large enough", self.text)


class TestUnusablePhotograph(unittest.TestCase):

    def test_out_of_focus_is_called_out_and_flagged(self):
        description = blank_description(
            usable=False,
            focus={"band": describe_photo.POOR, "variance_of_laplacian": 5.0})
        text = render(description, None)
        self.assertIn("out of focus", text)
        self.assertIn("may not be good enough to rely on", text)

    def test_no_intraoral_view_is_stated(self):
        description = blank_description(view=describe_photo.VIEW_NONE, usable=False)
        text = render(description, None)
        self.assertIn("No clear view", text)
        self.assertIn("may not be good enough to rely on", text)

    def test_a_usable_photo_carries_no_warning(self):
        self.assertNotIn("may not be good enough",
                         render(blank_description(), None))

    def test_disagreeing_estimates_refuse_the_figure(self):
        description = blank_description(
            mouth_extent={"cavity_model": 0.22, "hsv_heuristic": 0.70,
                          "agreed": False, "fraction": None,
                          "reason": "the two estimates disagree (22% and 70%)"})
        text = render(description, None)
        self.assertIn("could not be measured reliably", text)
        self.assertNotIn("takes up roughly", text)

    def test_agreeing_estimates_give_the_figure(self):
        text = render(blank_description(), None)
        self.assertIn("takes up roughly", text)


class TestMouthExtentPolicy(unittest.TestCase):
    """The spot-check decided this policy; these tests pin it down."""

    def setUp(self):
        self.cfg = copy.deepcopy(load_config())

    def test_heuristic_full_frame_is_not_treated_as_a_measurement(self):
        img = np.full((64, 64, 3), 30, np.uint8)
        original_model = describe_photo.cavity_region.predict_cavity
        original_heur = describe_photo.mouth_mask
        describe_photo.cavity_region.predict_cavity = \
            lambda *a, **k: np.full((64, 64), 255, np.uint8)
        describe_photo.mouth_mask = lambda *a, **k: np.full((64, 64), 255, np.uint8)
        try:
            extent = describe_photo.mouth_extent(img, self.cfg)
        finally:
            describe_photo.cavity_region.predict_cavity = original_model
            describe_photo.mouth_mask = original_heur
        self.assertIsNone(extent["fraction"])
        self.assertIn("fell back to the whole frame", extent["reason"])

    def test_agreement_within_tolerance_yields_a_figure(self):
        img = np.full((100, 100, 3), 30, np.uint8)
        model = np.zeros((100, 100), np.uint8); model[:50, :] = 255   # 50%
        heur = np.zeros((100, 100), np.uint8); heur[:55, :] = 255     # 55%
        original_model = describe_photo.cavity_region.predict_cavity
        original_heur = describe_photo.mouth_mask
        describe_photo.cavity_region.predict_cavity = lambda *a, **k: model
        describe_photo.mouth_mask = lambda *a, **k: heur
        try:
            extent = describe_photo.mouth_extent(img, self.cfg)
        finally:
            describe_photo.cavity_region.predict_cavity = original_model
            describe_photo.mouth_mask = original_heur
        self.assertTrue(extent["agreed"])
        self.assertAlmostEqual(extent["fraction"], 0.525, places=2)

    def test_disagreement_beyond_tolerance_withholds_the_figure(self):
        img = np.full((100, 100, 3), 30, np.uint8)
        model = np.zeros((100, 100), np.uint8); model[:22, :] = 255   # 22%
        heur = np.zeros((100, 100), np.uint8); heur[:70, :] = 255     # 70%
        original_model = describe_photo.cavity_region.predict_cavity
        original_heur = describe_photo.mouth_mask
        describe_photo.cavity_region.predict_cavity = lambda *a, **k: model
        describe_photo.mouth_mask = lambda *a, **k: heur
        try:
            extent = describe_photo.mouth_extent(img, self.cfg)
        finally:
            describe_photo.cavity_region.predict_cavity = original_model
            describe_photo.mouth_mask = original_heur
        self.assertFalse(extent["agreed"])
        self.assertIsNone(extent["fraction"])
        self.assertIn("disagree", extent["reason"])


class TestExcludedRegions(unittest.TestCase):

    def test_out_of_frame_region_is_excluded_with_a_reason(self):
        comparison = compare_caption.summarise(fake_change_report(
            lesions=[full_lesion(comparability="partial_out_of_frame",
                                 reason="partial_out_of_frame")]))
        self.assertEqual(comparison["regions"], [])
        self.assertEqual(len(comparison["excluded"]), 1)
        text = "\n".join(compare_caption.sentences(comparison))
        self.assertIn("left out of the comparison", text)
        self.assertIn("outside one of the two photographs", text)

    def test_excluded_region_reports_no_size_change(self):
        comparison = compare_caption.summarise(fake_change_report(
            lesions=[full_lesion(comparability="partial_out_of_frame",
                                 reason="partial_out_of_frame",
                                 area_change_pct=58.0)]))
        text = "\n".join(compare_caption.sentences(comparison))
        self.assertNotIn("58", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
