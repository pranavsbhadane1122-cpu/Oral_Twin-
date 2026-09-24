"""Phase 8 checks for the demo pipeline behind the UI.

  - the four cases the demo must handle return a well-formed result
  - refusals are results, not exceptions, and carry no measurement
  - no classifier output and no diagnostic language reaches any UI string while
    classification.surface_predictions is false

Run with: python -m unittest tests.test_app -v
"""

import re
import unittest

import numpy as np

from src.delta.compare import FULL, PARTIAL
from src.demo import masks as mask_src
from src.demo.pipeline import (
    NOT_INCLUDED,
    SCREENING_LINE,
    demo_cases,
    lesion_table_rows,
    run_demo,
)
from src.utils.config import load_config
from src.utils.prep_common import imread_unicode

CFG = load_config()

DIAGNOSTIC_PATTERNS = [
    r"\bdiagnos(is|e|ed|tic)\b",
    r"\byou have\b",
    r"\bcancer\b",
    r"\bmalignan(t|cy)\b",
    r"\bconfirms?\b",
    r"\bprescrib",
]
ALLOWED = ["not a diagnostic tool", "cannot diagnose", "not a diagnosis",
           "oral-cancer / opmd detection"]

RESULT_KEYS = {"status", "reason", "pair_name", "alignment", "images", "masks",
               "change_report", "risk", "report_text", "not_included", "screening_line"}


def ui_strings(result):
    """Every string the UI can put on screen for this result."""
    out = [result["report_text"], result["risk"]["text"], result["screening_line"],
           result["masks"]["label"], str(result["reason"] or ""),
           str(result["alignment"]["reason"] or "")]
    out += list(result["not_included"])
    out += [result["risk"]["context_note"]]
    out += [r["detail"] for r in result["risk"]["rules_fired"]]
    out += list(result["risk"]["advisories"])
    report = result["change_report"]
    out += [str(report.get("reason") or ""), report.get("summary", {}).get("headline", "")]
    for lesion in report.get("lesions", []):
        out.append(lesion["message"])
    return [s for s in out if s]


class DemoCaseMixin:
    case_key = None

    @classmethod
    def setUpClass(cls):
        case = demo_cases(CFG)[cls.case_key]
        cls.case = case
        cls.result = run_demo(case["paths"][0], case["paths"][1],
                              pair_name=case["pair_name"], cfg=CFG)

    def test_result_shape(self):
        self.assertTrue(RESULT_KEYS.issubset(self.result), RESULT_KEYS - set(self.result))
        self.assertIn(self.result["status"], ("ok", "refused"))
        for key in ("confidence", "breakdown", "reason", "n_inliers", "H"):
            self.assertIn(key, self.result["alignment"])
        self.assertIsNotNone(self.result["images"]["visit1"])
        self.assertIsNotNone(self.result["images"]["visit2"])
        self.assertIsInstance(self.result["report_text"], str)
        self.assertTrue(self.result["not_included"])

    def test_no_diagnostic_or_classifier_language(self):
        for text in ui_strings(self.result):
            haystack = text.lower()
            for allowed in ALLOWED:
                haystack = haystack.replace(allowed, "")
            for pattern in DIAGNOSTIC_PATTERNS:
                self.assertIsNone(re.search(pattern, haystack),
                                  f"{pattern!r} in UI string: {text[:160]}")
            for banned in CFG["classification"]["classes"]:
                self.assertNotIn(banned, haystack, f"class name in UI string: {text[:160]}")
            self.assertNotIn("signs consistent with", haystack)
            self.assertNotIn("model confidence", haystack)


class TestGoodPair(DemoCaseMixin, unittest.TestCase):
    case_key = "good"

    def test_compares_and_reports_change(self):
        self.assertEqual(self.result["status"], "ok")
        self.assertEqual(self.result["masks"]["source"], mask_src.SYNTHETIC)
        self.assertEqual(self.result["change_report"]["status"], "ok")
        self.assertTrue(self.result["change_report"]["lesions"])
        self.assertIsNotNone(self.result["images"]["overlay"])
        self.assertIn(self.result["risk"]["band"], ("Low", "Moderate", "Elevated"))

    def test_overlay_is_a_blend_of_both_photographs(self):
        overlay = self.result["images"]["overlay"]
        self.assertEqual(overlay.shape, self.result["images"]["visit1"].shape)
        self.assertFalse(np.array_equal(overlay, self.result["images"]["visit1"]))


class TestOutOfFramePair(DemoCaseMixin, unittest.TestCase):
    case_key = "out_of_frame"

    def test_partial_region_carries_no_area_number(self):
        lesions = self.result["change_report"]["lesions"]
        partial = [l for l in lesions if l["comparability"] == PARTIAL]
        self.assertTrue(partial, "this sample pair should have a partly out-of-frame region")
        for lesion in partial:
            self.assertIsNone(lesion["area_change_pct"])
            self.assertIsNone(lesion["color_shift"])
            self.assertFalse(lesion["notable_area_change"])
            self.assertIn("partly outside the newer photo", lesion["message"])

    def test_ui_table_shows_no_figures_for_a_non_comparable_region(self):
        """Raw areas must not be shown either: a reader could divide them."""
        lesions = self.result["change_report"]["lesions"]
        rows = lesion_table_rows(lesions)
        for lesion, row in zip(lesions, rows):
            if lesion["comparability"] == FULL:
                continue
            self.assertEqual(row["comparability"], "partly out of frame")
            for column in ("area, previous visit", "area, this visit",
                           "area change", "colour change"):
                self.assertEqual(row[column], "-", column)

    def test_partial_region_never_raises_the_band(self):
        comparable = [l for l in self.result["change_report"]["lesions"]
                      if l["comparability"] == FULL and l["notable_area_change"]]
        if not comparable:
            self.assertEqual(self.result["risk"]["band"], "Low")


class TestLowConfidencePair(DemoCaseMixin, unittest.TestCase):
    """Two photographs of different mouths: there is no alignment to find."""

    case_key = "low_confidence"

    def test_refuses_with_a_reason_and_no_numbers(self):
        self.assertEqual(self.result["status"], "refused")
        self.assertTrue(self.result["reason"])
        self.assertFalse(self.result["change_report"].get("comparable", False))
        self.assertEqual(self.result["change_report"].get("lesions", []), [])
        self.assertFalse(self.result["change_report"]["summary"].get("comparable"))

    def test_refusal_asks_for_a_retake(self):
        self.assertRegex(self.result["reason"].lower(), r"retake|line[d]? up")


class TestUnrelatedPhotos(DemoCaseMixin, unittest.TestCase):
    """A photograph that is not a mouth must fail gracefully."""

    case_key = "unrelated"

    def test_does_not_crash_and_refuses(self):
        self.assertEqual(self.result["status"], "refused")
        self.assertIsNone(self.result["alignment"]["H"])
        self.assertEqual(self.result["change_report"].get("lesions", []), [])
        self.assertIsNone(self.result["images"]["overlay"])


class TestUploadPath(unittest.TestCase):
    """Uploads arrive as arrays and must use the labelled placeholder masks."""

    @classmethod
    def setUpClass(cls):
        case = demo_cases(CFG)["good"]
        cls.img1 = imread_unicode(case["paths"][0])
        cls.img2 = imread_unicode(case["paths"][1])
        cls.result = run_demo(cls.img1, cls.img2, pair_name=None, cfg=CFG)

    def test_arrays_accepted_and_masks_labelled_placeholder(self):
        self.assertIn(self.result["status"], ("ok", "refused"))
        self.assertEqual(self.result["masks"]["source"], mask_src.PLACEHOLDER)
        label = self.result["masks"]["label"].lower()
        self.assertIn("placeholder", label)
        self.assertIn("not a trained segmenter", label)

    def test_pair_name_is_generic(self):
        self.assertEqual(self.result["pair_name"], "uploaded photos")


class TestStaticUiText(unittest.TestCase):
    def test_switch_is_off(self):
        self.assertFalse(CFG["classification"]["surface_predictions"])

    def test_not_included_panel_names_the_gaps(self):
        text = " ".join(NOT_INCLUDED).lower()
        for gap in ("classifier", "segmentation", "opmd"):
            self.assertIn(gap, text)

    def test_screening_line_present(self):
        self.assertIn("screening aid", SCREENING_LINE.lower())
        self.assertIn("dentist", SCREENING_LINE.lower())

    def test_app_source_mentions_no_condition_names(self):
        from src.utils.config import PROJECT_ROOT
        source = (PROJECT_ROOT / "app" / "oraltwin_app.py").read_text(encoding="utf-8")
        for banned in CFG["classification"]["classes"]:
            self.assertNotIn(banned, source.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
