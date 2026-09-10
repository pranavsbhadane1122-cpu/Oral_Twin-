"""Change-level explainability: what drove the difference BETWEEN two visits.

A plain Grad-CAM explains one photo. For longitudinal screening the useful
question is different - which regions changed the model's attention since the
last visit. change_cam() computes Grad-CAM on visit1 and on the aligned visit2
for the SAME class, then maps their difference:

    difference = CAM(visit2) - CAM(visit1)      (both in [0, 1])

rendered with a diverging colormap so red marks regions that gained the model's
attention and blue marks regions that lost it.

Two guards keep the picture honest:

  - The difference is masked to the valid-overlap region (Phase 5's
    delta.compare.overlap_valid_mask): outside the area photographed at both
    visits there is nothing to compare, so the map is forced to zero there.
  - Below delta.confidence_floor the pair is refused outright - no image, just
    a reason - because a difference map built on a bad alignment mostly shows
    misregistration, not change.

Comparing the same class index at both visits matters: if each CAM used its own
predicted class, the difference would conflate "attention moved" with "the
prediction changed".

OralTwin is a screening aid, not a diagnostic tool.
"""

import cv2
import numpy as np

from src.delta.compare import overlap_valid_mask
from src.explainability.gradcam import build_grad_model, compute, overlay
from src.utils.config import load_config


def diverging_colormap(diff, cmap_name="bwr"):
    """Map a [-1, 1] difference to BGR with matplotlib's diverging colormap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    normalized = (np.clip(diff, -1.0, 1.0) + 1.0) / 2.0
    rgba = cm.get_cmap(cmap_name)(normalized)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def change_cam(model, img1, img2_warped, H, confidence, cfg=None, class_index=None,
               grad_model=None):
    """Difference-of-CAMs between two aligned visits.

    H: homography used to warp visit2 into visit1 (None when alignment failed).
    Returns a dict with status, and on success difference/overlay images.
    """
    cfg = cfg or load_config()
    floor = cfg["delta"]["confidence_floor"]
    result = {
        "status": "ok",
        "reason": None,
        "alignment_confidence": round(float(confidence), 4),
        "difference": None,
        "overlay": None,
        "cam1": None,
        "cam2": None,
        "class_index": None,
        "class_name": None,
        "flat": None,
    }

    if H is None or float(confidence) < floor:
        result["status"] = "refused"
        result["reason"] = (
            f"alignment confidence {float(confidence):.2f} is below the floor {floor:.2f}"
            if H is not None else "the two photos could not be aligned"
        ) + "; a change map would show camera movement rather than real change. "\
            "Retake the photo in similar lighting and framing."
        return result

    grad_model = grad_model or build_grad_model(model, cfg=cfg)
    cam1 = compute(model, img1, class_index=class_index, cfg=cfg, grad_model=grad_model)
    # both visits are explained for the SAME class, so the difference is about
    # attention moving, not about the prediction flipping
    cam2 = compute(model, img2_warped, class_index=cam1["class_index"], cfg=cfg,
                   grad_model=grad_model)

    diff = cam2["heatmap"] - cam1["heatmap"]
    valid = overlap_valid_mask(img1.shape, H, img2_warped.shape)
    diff = np.where(valid > 127, diff, 0.0).astype(np.float32)

    peak = float(np.abs(diff).max())
    normalized = diff / peak if peak > 1e-12 else diff

    colored = diverging_colormap(normalized, cfg["explain"]["diverging_colormap"])
    alpha = float(cfg["explain"]["overlay_alpha"])
    blended = cv2.addWeighted(colored, alpha, img1, 1.0 - alpha, 0.0)
    blended = np.where(valid[..., None] > 127, blended, img1 // 3)  # dim outside overlap

    result.update({
        "difference": normalized,
        "overlay": blended.astype(np.uint8),
        "cam1": cam1,
        "cam2": cam2,
        "class_index": cam1["class_index"],
        "class_name": cam1["class_name"],
        "flat": bool(cam1["flat"] or cam2["flat"]),
        "peak_abs_difference": round(peak, 4),
        "valid_fraction": round(float((valid > 127).mean()), 4),
        "increased_fraction": round(float((normalized > 0.25).mean()), 4),
        "decreased_fraction": round(float((normalized < -0.25).mean()), 4),
    })
    return result


def annotate(image, text_lines):
    """Caption strip above an image."""
    banner = np.zeros((18 * len(text_lines) + 8, image.shape[1], 3), np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(banner, line[:110], (5, 15 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, image])


def side_by_side(img1, img2_warped, result, cfg=None):
    """visit1 CAM | visit2 CAM | change map, captioned."""
    cfg = cfg or load_config()
    if result["status"] != "ok":
        return None
    panels = [
        overlay(img1, result["cam1"]["heatmap"], cfg),
        overlay(img2_warped, result["cam2"]["heatmap"], cfg),
        result["overlay"],
    ]
    labels = ["visit1 Grad-CAM", "visit2 Grad-CAM", "change (red=more, blue=less)"]
    stamped = []
    for panel, label in zip(panels, labels):
        p = panel.copy()
        cv2.rectangle(p, (0, 0), (p.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(p, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                    cv2.LINE_AA)
        stamped.append(p)
    strip = np.hstack(stamped)
    caption = [
        f"class explained: {result['class_name']}  "
        f"(visit1 p={result['cam1']['probability']:.2f}, "
        f"visit2 p={result['cam2']['probability']:.2f})  "
        f"alignment confidence {result['alignment_confidence']:.2f}",
        "Screening aid, not a diagnostic tool - have any finding checked by a dentist.",
    ]
    if result["flat"]:
        caption.insert(1, "WARNING: a heatmap is flat - this explanation is not informative")
    return annotate(strip, caption)
