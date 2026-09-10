"""Phase 7 checks (run with: python -m unittest tests.test_explain_risk -v).

  - Grad-CAM heatmap shape and range are correct
  - the flat-CAM warning fires when the CAM carries no localisation
  - change-CAM refuses below the confidence floor, and is exactly zero outside
    the valid overlap
  - risk rules: no-change -> Low; injected growth -> Moderate/Elevated citing
    the right rule; a non-comparable lesion never raises the band
  - the edge/compactness metric appears in no rule output
  - no diagnostic language anywhere in generated text
"""

import json
import re
import unittest

import cv2
import numpy as np
import tensorflow as tf

from src.delta.compare import FULL, PARTIAL, compare
from src.explainability.change_cam import change_cam
from src.explainability.gradcam import (
    build_grad_model,
    compute,
    finalize_heatmap,
    load_classifier,
    overlay,
    resolve_layer_name,
)
from src.risk.report import build_report
from src.risk.risk_rules import ELEVATED, LOW, MODERATE, stratify
from src.utils.config import load_config

CFG = load_config()
SIZE = CFG["classification"]["img_size"]

# words a screening aid must never use about the person in front of it
DIAGNOSTIC_PATTERNS = [
    r"\bdiagnos(is|e|ed|tic)\b",
    r"\byou have\b",
    r"\byou are suffering\b",
    r"\bcancer\b",
    r"\bmalignan(t|cy)\b",
    r"\bbenign\b",
    r"\bconfirms?\b",
    r"\btreatment plan\b",
    r"\bprescrib",
]
# negated disclaimers are the point, not a violation
ALLOWED = ["not a diagnostic tool", "cannot diagnose", "not a diagnosis",
           "never a diagnosis", "no diagnosis"]


def assert_no_diagnostic_language(testcase, text):
    haystack = text.lower()
    for allowed in ALLOWED:
        haystack = haystack.replace(allowed, "")
    for pattern in DIAGNOSTIC_PATTERNS:
        testcase.assertIsNone(re.search(pattern, haystack),
                              f"diagnostic language matching {pattern!r} in: {text[:300]}")


def disc(radius, center=(112, 112), size=SIZE):
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, center, int(radius), 255, -1)
    return mask


def flat_image(bgr=(120, 140, 200), size=SIZE):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = bgr
    return img


def stub_model(zero_head=False):
    """Model shaped like the classifier (nested base + gap/dropout/probs).

    With zero_head the classifier weights are zero, so the class score has no
    gradient, the CAM collapses to a constant, and the flat guard must fire.
    """
    base_in = tf.keras.Input(shape=(SIZE, SIZE, 3))
    x = tf.keras.layers.Conv2D(4, 3, strides=8, padding="same", activation="relu")(base_in)
    base = tf.keras.Model(base_in, x, name="mobilenetv2_1.00_224")

    inputs = tf.keras.Input(shape=(SIZE, SIZE, 3), name="image")
    feat = base(inputs)
    gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")(feat)
    drop = tf.keras.layers.Dropout(0.0, name="dropout")(gap)
    probs = tf.keras.layers.Dense(len(CFG["classification"]["classes"]),
                                  activation="softmax", name="probs")(drop)
    model = tf.keras.Model(inputs, probs)
    if zero_head:
        dense = model.get_layer("probs")
        dense.set_weights([np.zeros_like(w) for w in dense.get_weights()])
    return model


class TestGradCam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_classifier(CFG)
        cls.grad_model = build_grad_model(cls.model, cfg=CFG)

    def test_resolved_layer_is_spatial(self):
        name = resolve_layer_name(self.model, CFG)
        layer = self.model.get_layer("mobilenetv2_1.00_224").get_layer(name)
        self.assertEqual(len(layer.output_shape), 4, "Grad-CAM target must be a feature map")

    def test_heatmap_shape_and_range(self):
        img = flat_image()
        img[60:150, 60:150] = (40, 60, 160)
        result = compute(self.model, img, cfg=CFG, grad_model=self.grad_model)

        self.assertEqual(result["heatmap"].shape, img.shape[:2])
        self.assertGreaterEqual(float(result["heatmap"].min()), 0.0)
        self.assertLessEqual(float(result["heatmap"].max()), 1.0)
        self.assertIn(result["class_name"], CFG["classification"]["classes"])
        self.assertTrue(0.0 <= result["probability"] <= 1.0)

    def test_explicit_class_index_is_honoured(self):
        img = flat_image()
        result = compute(self.model, img, class_index=1, cfg=CFG, grad_model=self.grad_model)
        self.assertEqual(result["class_index"], 1)

    def test_overlay_shape_matches_image(self):
        img = flat_image()
        blended = overlay(img, np.zeros(img.shape[:2], np.float32), CFG)
        self.assertEqual(blended.shape, img.shape)
        self.assertEqual(blended.dtype, np.uint8)


class TestFlatCam(unittest.TestCase):
    def test_uniform_cam_is_flagged_flat(self):
        heatmap, std, flat = finalize_heatmap(np.full((7, 7), 3.7, np.float32),
                                              (SIZE, SIZE), CFG)
        self.assertTrue(flat)
        self.assertAlmostEqual(std, 0.0, places=6)
        self.assertAlmostEqual(float(heatmap.max()), 0.0, places=6)

    def test_zero_gradient_model_produces_flat_cam(self):
        model = stub_model(zero_head=True)
        result = compute(model, flat_image(), cfg=CFG)
        self.assertTrue(result["flat"],
                        f"a gradient-free model must yield a flat CAM: {result['heatmap_std']}")

    def test_structured_cam_is_not_flat(self):
        cam = np.zeros((7, 7), np.float32)
        cam[2:5, 2:5] = 1.0
        _, std, flat = finalize_heatmap(cam, (SIZE, SIZE), CFG)
        self.assertFalse(flat)
        self.assertGreater(std, CFG["explain"]["flat_cam_std_eps"])


class TestChangeCam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_classifier(CFG)
        cls.grad_model = build_grad_model(cls.model, cfg=CFG)
        cls.img = flat_image()
        cls.img[50:170, 50:170] = (40, 70, 170)

    def test_refuses_below_confidence_floor(self):
        H = np.eye(3)
        low = CFG["delta"]["confidence_floor"] - 0.05
        result = change_cam(self.model, self.img, self.img, H, low, cfg=CFG,
                            grad_model=self.grad_model)
        self.assertEqual(result["status"], "refused")
        self.assertIsNone(result["overlay"])
        self.assertIsNone(result["difference"])
        self.assertIn("confidence", result["reason"])

    def test_refuses_when_alignment_failed(self):
        result = change_cam(self.model, self.img, self.img, None, 0.9, cfg=CFG,
                            grad_model=self.grad_model)
        self.assertEqual(result["status"], "refused")
        self.assertIn("could not be aligned", result["reason"])

    def test_zero_outside_overlap(self):
        shift = 40
        H = np.array([[1.0, 0.0, float(shift)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = change_cam(self.model, self.img, self.img, H, 0.9, cfg=CFG,
                            grad_model=self.grad_model)
        self.assertEqual(result["status"], "ok")
        outside = result["difference"][:, :shift]
        self.assertTrue(np.all(outside == 0.0),
                        "difference map must be zero where the photos do not overlap")
        self.assertLess(result["valid_fraction"], 1.0)

    def test_identical_visits_have_no_change(self):
        result = change_cam(self.model, self.img, self.img.copy(), np.eye(3), 0.9, cfg=CFG,
                            grad_model=self.grad_model)
        self.assertEqual(result["status"], "ok")
        self.assertLess(float(np.abs(result["difference"]).max()), 1e-5)


def change_report(area1, area2, confidence=0.85, state="matched", comparability=FULL,
                  color_notable=False, edge_notable=True):
    """Hand-built ChangeReport; edge_notable defaults True to prove it is ignored."""
    change = None if comparability == PARTIAL else 100.0 * (area2 - area1) / max(area1, 1)
    return {
        "status": "ok", "reason": None, "alignment_confidence": confidence,
        "reliability": "high" if confidence >= 0.7 else "moderate",
        "lesions": [{
            "id": 0, "state": state, "comparability": comparability, "match_iou": 0.6,
            "area_visit1_px": area1, "area_visit2_px": area2,
            "area_change_pct": change, "notable_area_change": bool(change and change >= 10),
            "color_shift": {"measurable": True, "magnitude": 40.0, "notable": color_notable},
            "edge_irregularity": {"compactness_visit1": 1.1, "compactness_visit2": 1.9,
                                  "relative_change": 0.72, "notable": edge_notable},
            "confidence": confidence, "reliability": "high", "reason": None,
            "message": "test lesion",
        }],
        "summary": {"comparable": True, "n_lesions_visit1": 1, "n_lesions_visit2": 1,
                    "n_matched": 1, "n_new": 0, "n_resolved": 0, "n_comparable": 1,
                    "n_partial_out_of_frame": 0, "n_notable_changes": 0,
                    "any_notable_change": False, "headline": "test"},
        "disclaimer": "test",
    }


class TestRiskRules(unittest.TestCase):
    def test_no_change_is_low(self):
        risk = stratify(change_report(1000, 1000), cfg=CFG)
        self.assertEqual(risk["band"], LOW)
        self.assertEqual(risk["rules_fired"], [])

    def test_moderate_growth_fires_area_rule(self):
        risk = stratify(change_report(1000, 1150), cfg=CFG)  # +15%
        self.assertEqual(risk["band"], MODERATE)
        self.assertEqual([r["rule"] for r in risk["rules_fired"]], ["AREA_GROWTH"])

    def test_large_growth_is_elevated(self):
        risk = stratify(change_report(1000, 1600), cfg=CFG)  # +60%
        self.assertEqual(risk["band"], ELEVATED)
        self.assertIn("AREA_GROWTH_LARGE", [r["rule"] for r in risk["rules_fired"]])

    def test_two_moderate_rules_escalate(self):
        risk = stratify(change_report(1000, 1150, color_notable=True), cfg=CFG)
        self.assertEqual(risk["band"], ELEVATED)
        self.assertTrue(risk["escalated_by_rule_count"])
        self.assertIn("COLOUR_CHANGE", [r["rule"] for r in risk["rules_fired"]])

    def test_persistent_condition_rule(self):
        pred = {"class_name": "gingivitis", "probability": 0.95}
        risk = stratify(change_report(1000, 1000), pred, dict(pred), cfg=CFG)
        self.assertEqual([r["rule"] for r in risk["rules_fired"]], ["PERSISTENT_CONDITION"])
        self.assertEqual(risk["band"], MODERATE)

    def test_non_comparable_lesion_never_raises_band(self):
        report = change_report(1000, 400, comparability=PARTIAL)
        risk = stratify(report, cfg=CFG)
        self.assertEqual(risk["band"], LOW, "an uncomparable lesion must not raise risk")
        self.assertEqual(risk["rules_fired"], [])
        self.assertTrue(any("retake" in a.lower() for a in risk["advisories"]))

    def test_low_confidence_never_raises_band(self):
        report = change_report(1000, 1000, confidence=0.35)
        report["reliability"] = "low"
        risk = stratify(report, alignment_confidence=0.35, cfg=CFG)
        self.assertEqual(risk["band"], LOW)
        self.assertTrue(any("confidence" in a.lower() for a in risk["advisories"]))

    def test_unreliable_report_produces_advisory_only(self):
        risk = stratify({"status": "unreliable", "reason": "could not align",
                         "lesions": [], "summary": {"comparable": False}}, cfg=CFG)
        self.assertEqual(risk["band"], LOW)
        self.assertEqual(risk["rules_fired"], [])
        self.assertTrue(risk["advisories"])

    def test_edge_metric_never_influences_rules(self):
        """edge_notable is True in every fixture; it must never appear anywhere."""
        for report in (change_report(1000, 1000), change_report(1000, 1600),
                       change_report(1000, 1000, comparability=PARTIAL)):
            risk = stratify(report, cfg=CFG)
            blob = json.dumps(risk["rules_fired"]) + risk["text"]
            for banned in ("compactness", "edge_irregularity", "irregular", "border"):
                self.assertNotIn(banned, blob.lower(),
                                 f"edge metric leaked into risk output: {blob[:200]}")
        # and the exclusion is declared
        self.assertTrue(any("edge" in m for m in stratify(change_report(1000, 1000),
                                                          cfg=CFG)["excluded_metrics"]))


class TestReportLanguage(unittest.TestCase):
    def test_dentist_line_present_above_low(self):
        risk = stratify(change_report(1000, 1600), cfg=CFG)
        self.assertIn("checked by a dentist", risk["text"].lower())

    def test_low_band_has_no_alarm(self):
        risk = stratify(change_report(1000, 1000), cfg=CFG)
        self.assertNotIn("checked by a dentist", risk["text"].lower())

    def test_generated_reports_have_no_diagnostic_language(self):
        pred = {"class_name": "gingivitis", "probability": 0.93}
        for report in (change_report(1000, 1000), change_report(1000, 1600),
                       change_report(1000, 400, comparability=PARTIAL)):
            risk = stratify(report, pred, dict(pred), cfg=CFG)
            text = build_report("pair_test", report, risk, pred, dict(pred),
                                {"Grad-CAM": "x.jpg"}, {})
            assert_no_diagnostic_language(self, text)
            assert_no_diagnostic_language(self, risk["text"])
            self.assertIn("screening aid", text.lower())

    def test_condition_is_phrased_as_signs_not_diagnosis(self):
        pred = {"class_name": "calculus", "probability": 0.88}
        risk = stratify(change_report(1000, 1000), cfg=CFG)
        text = build_report("pair_test", change_report(1000, 1000), risk, pred, pred)
        self.assertIn("signs consistent with", text)

    def test_context_note_is_general_not_personal(self):
        risk = stratify(change_report(1000, 1600), cfg=CFG)
        note = risk["context_note"].lower()
        self.assertIn("not a prediction about you", note)
        self.assertNotIn("your risk of", note)


class TestCompareIntegration(unittest.TestCase):
    """The risk module must read a real ChangeReport, not just the fixtures."""

    def test_real_change_report_flows_into_rules(self):
        img = flat_image()
        report = compare(disc(30), disc(45), img, img.copy(), 0.9, CFG)
        risk = stratify(report, cfg=CFG)
        self.assertIn(risk["band"], (MODERATE, ELEVATED))
        self.assertTrue(risk["rules_fired"])
        assert_no_diagnostic_language(self, risk["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
