"""Compare two visits' lesion masks and describe what changed.

compare(mask1, mask2_warped, img1, img2_warped, confidence) -> ChangeReport dict
(JSON-serializable). Both masks and both images must already be in VISIT1's
coordinate frame - run_delta.py warps visit2 with the homography estimated by
the Phase 4 alignment module before calling this.

The masks are plain binary/instance masks, so the real Phase 3 U-Net output
plugs in here unchanged.

Per lesion (connected component above delta.min_lesion_area_px):
  - area change %, matched between visits by IoU; unmatched components are
    reported as "new" (only in visit2) or "resolved" (only in visit1)
  - colour shift: mean HSV inside the lesion in each visit, hue compared
    circularly
  - edge irregularity: compactness = perimeter^2 / (4*pi*area), 1.0 for a
    perfect disc; the relative change between visits is reported

Alignment confidence never scales the measurements - a warped-in measurement
error is not made smaller by multiplying it. Confidence is instead attached to
every lesion as an explicit reliability annotation, and below
delta.confidence_floor the whole comparison is refused rather than reported
with numbers that cannot be trusted.

OralTwin is a screening aid, not a diagnostic tool: every user-facing line says
to have findings checked by a dentist and never names a condition.
"""

import cv2
import numpy as np

from src.utils.config import load_config

DISCLAIMER = ("OralTwin is a screening aid, not a diagnostic tool. "
              "Have any finding checked by a dentist.")


def _binary(mask):
    m = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (m > 127).astype(np.uint8)


def components(mask, min_area):
    """List of dicts (label, area, mask) for components at or above min_area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(_binary(mask), connectivity=8)
    out = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out.append({"label": label, "area": area,
                        "mask": (labels == label).astype(np.uint8)})
    return out


def compactness(component_mask):
    """perimeter^2 / (4*pi*area); 1.0 for a disc, higher = more irregular."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0:
        return None
    return float(perimeter ** 2 / (4.0 * np.pi * area))


def mean_hsv(img, component_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sel = component_mask.astype(bool)
    if not sel.any():
        return None
    return [float(hsv[..., c][sel].mean()) for c in range(3)]


def hue_delta(h1, h2):
    """Circular difference on OpenCV's 0-179 hue scale."""
    d = abs(h1 - h2) % 180.0
    return float(min(d, 180.0 - d))


def match_components(comps1, comps2, iou_thresh):
    """Greedy IoU matching. Returns (pairs, unmatched1, unmatched2)."""
    scores = []
    for i, c1 in enumerate(comps1):
        for j, c2 in enumerate(comps2):
            inter = int(np.logical_and(c1["mask"], c2["mask"]).sum())
            if not inter:
                continue
            union = int(np.logical_or(c1["mask"], c2["mask"]).sum())
            iou = inter / union if union else 0.0
            if iou >= iou_thresh:
                scores.append((iou, i, j))

    scores.sort(reverse=True)
    used1, used2, pairs = set(), set(), []
    for iou, i, j in scores:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        pairs.append((i, j, iou))
    return (pairs,
            [i for i in range(len(comps1)) if i not in used1],
            [j for j in range(len(comps2)) if j not in used2])


def reliability_label(confidence):
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.5:
        return "moderate"
    return "low"


def _message(state, area_change_pct, notable, confidence, color_notable, edge_notable):
    conf_txt = f"confidence {confidence:.2f}"
    check = "have this checked by a dentist"
    if state == "new":
        return f"a new area appeared since the last visit ({conf_txt}) - {check}"
    if state == "resolved":
        return (f"an area seen at the last visit is no longer visible ({conf_txt}) - "
                f"mention it to your dentist at your next appointment")
    direction = "grew" if area_change_pct >= 0 else "shrank"
    magnitude = abs(area_change_pct)
    if notable:
        return f"area {direction} ~{magnitude:.0f}% since the last visit, {conf_txt} - {check}"
    extras = []
    if color_notable:
        extras.append("colour looks different")
    if edge_notable:
        extras.append("the border looks less regular")
    if extras:
        return (f"area is about the same (~{magnitude:.0f}% {direction}) but "
                f"{' and '.join(extras)}, {conf_txt} - {check}")
    return f"no notable change (~{magnitude:.0f}% {direction}), {conf_txt}"


def compare(mask1, mask2_warped, img1, img2_warped, confidence, cfg=None):
    """Return a JSON-serializable ChangeReport dict."""
    cfg = cfg or load_config()
    dcfg = cfg["delta"]
    confidence = float(confidence)

    report = {
        "status": "ok",
        "reason": None,
        "alignment_confidence": round(confidence, 4),
        "reliability": reliability_label(confidence),
        "lesions": [],
        "summary": {},
        "disclaimer": DISCLAIMER,
    }

    if confidence < dcfg["confidence_floor"]:
        report["status"] = "unreliable"
        report["reason"] = (
            f"alignment confidence {confidence:.2f} is below the floor "
            f"{dcfg['confidence_floor']:.2f}; the two photos could not be lined up well "
            f"enough to compare them. Retake the photo in similar lighting and framing."
        )
        report["summary"] = {"comparable": False}
        return report

    min_area = dcfg["min_lesion_area_px"]
    comps1 = components(mask1, min_area)
    comps2 = components(mask2_warped, min_area)
    pairs, only1, only2 = match_components(comps1, comps2, dcfg["match_iou_thresh"])

    lesions = []
    for i, j, iou in pairs:
        c1, c2 = comps1[i], comps2[j]
        change_pct = 100.0 * (c2["area"] - c1["area"]) / c1["area"]
        notable_area = abs(change_pct) >= dcfg["change_area_thresh"] * 100.0

        hsv1, hsv2 = mean_hsv(img1, c1["mask"]), mean_hsv(img2_warped, c2["mask"])
        color = None
        color_notable = False
        if hsv1 and hsv2:
            dh = hue_delta(hsv1[0], hsv2[0])
            ds, dv = hsv2[1] - hsv1[1], hsv2[2] - hsv1[2]
            magnitude = float(np.sqrt(dh ** 2 + ds ** 2 + dv ** 2))
            color_notable = magnitude >= dcfg["color_shift_thresh"]
            color = {"delta_hue": round(dh, 2), "delta_saturation": round(ds, 2),
                     "delta_value": round(dv, 2), "magnitude": round(magnitude, 2),
                     "notable": color_notable}

        k1, k2 = compactness(c1["mask"]), compactness(c2["mask"])
        edge = None
        edge_notable = False
        if k1 and k2:
            rel = (k2 - k1) / k1
            edge_notable = abs(rel) >= dcfg["compactness_change_thresh"]
            edge = {"compactness_visit1": round(k1, 4), "compactness_visit2": round(k2, 4),
                    "relative_change": round(rel, 4), "notable": edge_notable}

        lesions.append({
            "id": len(lesions),
            "state": "matched",
            "match_iou": round(float(iou), 4),
            "area_visit1_px": c1["area"],
            "area_visit2_px": c2["area"],
            "area_change_pct": round(change_pct, 2),
            "notable_area_change": notable_area,
            "color_shift": color,
            "edge_irregularity": edge,
            "confidence": round(confidence, 4),
            "reliability": reliability_label(confidence),
            "message": _message("matched", change_pct, notable_area, confidence,
                                color_notable, edge_notable),
        })

    for i in only1:
        lesions.append({
            "id": len(lesions), "state": "resolved", "match_iou": None,
            "area_visit1_px": comps1[i]["area"], "area_visit2_px": 0,
            "area_change_pct": -100.0, "notable_area_change": True,
            "color_shift": None, "edge_irregularity": None,
            "confidence": round(confidence, 4), "reliability": reliability_label(confidence),
            "message": _message("resolved", -100.0, True, confidence, False, False),
        })
    for j in only2:
        lesions.append({
            "id": len(lesions), "state": "new", "match_iou": None,
            "area_visit1_px": 0, "area_visit2_px": comps2[j]["area"],
            "area_change_pct": None, "notable_area_change": True,
            "color_shift": None, "edge_irregularity": None,
            "confidence": round(confidence, 4), "reliability": reliability_label(confidence),
            "message": _message("new", 0.0, True, confidence, False, False),
        })

    report["lesions"] = lesions
    notable = [l for l in lesions if l["notable_area_change"]
               or (l["color_shift"] or {}).get("notable")
               or (l["edge_irregularity"] or {}).get("notable")]
    report["summary"] = {
        "comparable": True,
        "n_lesions_visit1": len(comps1),
        "n_lesions_visit2": len(comps2),
        "n_matched": len(pairs),
        "n_new": len(only2),
        "n_resolved": len(only1),
        "n_notable_changes": len(notable),
        "any_notable_change": bool(notable),
        "headline": (
            f"{len(notable)} area(s) changed noticeably since the last visit - {DISCLAIMER}"
            if notable else
            "No notable change since the last visit. Keep taking photos at each visit."
        ),
    }
    return report
