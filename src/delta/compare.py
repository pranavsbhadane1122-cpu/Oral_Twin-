"""Compare two visits' lesion masks and describe what changed.

compare(mask1, mask2_warped, img1, img2_warped, confidence, valid_mask=...)
    -> ChangeReport dict (JSON-serializable)

Both masks and both images must already be in VISIT1's coordinate frame -
run_delta.py warps visit2 with the homography estimated by the Phase 4
alignment module before calling this. The masks are plain binary/instance
masks, so the real Phase 3 U-Net output plugs in here unchanged.

Per lesion (connected component above delta.min_lesion_area_px):

  area change %
      matched between visits by IoU; unmatched components are reported as
      "new" (only in visit2) or "resolved" (only in visit1).

  colour change, measured RELATIVELY
      A global lighting or white-balance difference between two phone photos
      shifts the whole frame, so an absolute lesion-HSV comparison mostly
      measures the camera. Instead each image is measured on its own terms:
      lesion mean HSV minus the mean HSV of a ring of surrounding tissue
      (delta.color_ring_px thick, excluding every other lesion and anything
      outside the comparable region). That lesion-vs-neighbourhood contrast is
      then compared between visits, so any shift common to the whole frame
      cancels. When the ring is too small to sample (delta.color_ring_min_px)
      the colour metric reports "not measurable" rather than a number.

  edge irregularity
      compactness = perimeter^2 / (4*pi*area), 1.0 for a perfect disc; the
      relative change between visits is reported.

Comparability
    valid_mask marks the region visible in BOTH photos (visit1's frame
    intersected with visit2's frame warped into visit1 coordinates). A lesion
    lying within delta.edge_margin_px of that region's boundary is only partly
    inside the newer photo, so its apparent area shrinks for framing reasons
    alone. Such a lesion is labelled "partial_out_of_frame", carries no area or
    colour numbers, and is excluded from change math. Every lesion carries a
    "comparability" field: full | partial_out_of_frame | unreliable_low_confidence.

Alignment confidence never scales the measurements - a warped-in measurement
error is not made smaller by multiplying it. Confidence is attached as an
explicit reliability annotation, and below delta.confidence_floor the whole
comparison is refused rather than reported with numbers that cannot be trusted.

OralTwin is a screening aid, not a diagnostic tool: every user-facing line says
to have findings checked by a dentist and never names a condition.
"""

import cv2
import numpy as np

from src.utils.config import load_config

DISCLAIMER = ("OralTwin is a screening aid, not a diagnostic tool. "
              "Have any finding checked by a dentist.")

FULL = "full"
PARTIAL = "partial_out_of_frame"
LOW_CONF = "unreliable_low_confidence"


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


def overlap_valid_mask(shape, H, visit2_shape=None):
    """Region visible in BOTH photos, in visit1 coordinates.

    visit1's frame is the whole image; visit2's frame is a full-white image of
    visit2's size warped through H (visit2 -> visit1). Their intersection is
    the only place an area comparison is fair.
    """
    h, w = shape[:2]
    h2, w2 = (visit2_shape or shape)[:2]
    warped_frame = cv2.warpPerspective(
        np.full((h2, w2), 255, np.uint8), np.asarray(H, float), (w, h),
        flags=cv2.INTER_NEAREST,
    )
    return (warped_frame > 127).astype(np.uint8) * 255


def touches_boundary(component_mask, valid_mask, edge_margin_px):
    """True if the component reaches within edge_margin_px of the comparable
    region's edge (so part of it may be missing from the newer photo)."""
    if valid_mask is None:
        return False
    valid = _binary(valid_mask)
    margin = max(int(edge_margin_px), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    safe = cv2.erode(valid, k)
    return bool(np.logical_and(component_mask.astype(bool), safe == 0).any())


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


def circular_signed_delta(a, b):
    """a - b on OpenCV's 0-179 hue circle, result in [-90, 90)."""
    return float((a - b + 90.0) % 180.0 - 90.0)


def ring_region(component_mask, all_lesions, valid_mask, ring_px):
    """Ring of surrounding tissue: the lesion dilated by ring_px, minus every
    lesion pixel in that image, clipped to the comparable region (so the black
    padding left by the warp is never sampled)."""
    margin = max(int(ring_px), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    ring = cv2.dilate(component_mask, k)
    ring = np.logical_and(ring.astype(bool), ~_binary(all_lesions).astype(bool))
    if valid_mask is not None:
        ring = np.logical_and(ring, _binary(valid_mask).astype(bool))
    return ring.astype(np.uint8)


def relative_hsv(img, component_mask, ring_mask, min_ring_px):
    """Lesion HSV minus surrounding-ring HSV, or None if the ring is too small."""
    lesion_sel = component_mask.astype(bool)
    ring_sel = ring_mask.astype(bool)
    if not lesion_sel.any() or int(ring_sel.sum()) < min_ring_px:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    lesion = [float(hsv[..., c][lesion_sel].mean()) for c in range(3)]
    ring = [float(hsv[..., c][ring_sel].mean()) for c in range(3)]
    return {
        "hue": circular_signed_delta(lesion[0], ring[0]),
        "saturation": lesion[1] - ring[1],
        "value": lesion[2] - ring[2],
        "ring_px": int(ring_sel.sum()),
    }


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


def reliability_label(confidence, cfg=None):
    dcfg = (cfg or load_config())["delta"]
    if confidence >= dcfg["reliability_high"]:
        return "high"
    if confidence >= dcfg["reliability_moderate"]:
        return "moderate"
    return "low"


RETAKE = ("this spot is partly outside the newer photo - retake with the area fully "
          "visible for a fair comparison")


def _message(state, comparability, area_change_pct, notable, confidence,
             color_notable, edge_notable):
    conf_txt = f"confidence {confidence:.2f}"
    check = "have this checked by a dentist"
    if comparability == PARTIAL:
        return RETAKE
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


def _blank_lesion(idx, state, comparability, confidence, cfg, area1=0, area2=0, reason=None):
    return {
        "id": idx, "state": state, "comparability": comparability, "match_iou": None,
        "area_visit1_px": int(area1), "area_visit2_px": int(area2),
        "area_change_pct": None, "notable_area_change": False,
        "color_shift": None, "edge_irregularity": None,
        "confidence": round(confidence, 4), "reliability": reliability_label(confidence, cfg),
        "reason": reason,
        "message": _message(state, comparability, 0.0, False, confidence, False, False),
    }


def compare(mask1, mask2_warped, img1, img2_warped, confidence, cfg=None, valid_mask=None):
    """Return a JSON-serializable ChangeReport dict.

    valid_mask: region visible in both photos, in visit1 coordinates (see
    overlap_valid_mask). When None the whole frame is assumed comparable.
    """
    cfg = cfg or load_config()
    dcfg = cfg["delta"]
    confidence = float(confidence)

    report = {
        "status": "ok",
        "reason": None,
        "alignment_confidence": round(confidence, 4),
        "reliability": reliability_label(confidence, cfg),
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

    base_comparability = LOW_CONF if reliability_label(confidence, cfg) == "low" else FULL
    margin = dcfg["edge_margin_px"]
    lesions = []

    for i, j, iou in pairs:
        c1, c2 = comps1[i], comps2[j]
        partial = (touches_boundary(c1["mask"], valid_mask, margin)
                   or touches_boundary(c2["mask"], valid_mask, margin))
        if partial:
            lesion = _blank_lesion(len(lesions), "matched", PARTIAL, confidence, cfg,
                                   c1["area"], c2["area"],
                                   reason="lesion reaches the edge of the region visible "
                                          "in both photos")
            lesion["match_iou"] = round(float(iou), 4)
            lesions.append(lesion)
            continue

        change_pct = 100.0 * (c2["area"] - c1["area"]) / c1["area"]
        notable_area = abs(change_pct) >= dcfg["change_area_thresh"] * 100.0

        ring1 = ring_region(c1["mask"], mask1, valid_mask, dcfg["color_ring_px"])
        ring2 = ring_region(c2["mask"], mask2_warped, valid_mask, dcfg["color_ring_px"])
        rel1 = relative_hsv(img1, c1["mask"], ring1, dcfg["color_ring_min_px"])
        rel2 = relative_hsv(img2_warped, c2["mask"], ring2, dcfg["color_ring_min_px"])

        color_notable = False
        if rel1 is None or rel2 is None:
            color = {"measurable": False,
                     "reason": "not enough surrounding tissue to measure colour against"}
        else:
            dh = rel2["hue"] - rel1["hue"]
            ds = rel2["saturation"] - rel1["saturation"]
            dv = rel2["value"] - rel1["value"]
            magnitude = float(np.sqrt(dh ** 2 + ds ** 2 + dv ** 2))
            color_notable = magnitude >= dcfg["relative_color_thresh"]
            color = {
                "measurable": True,
                "relative_visit1": {k: round(v, 2) for k, v in rel1.items() if k != "ring_px"},
                "relative_visit2": {k: round(v, 2) for k, v in rel2.items() if k != "ring_px"},
                "delta_hue": round(dh, 2), "delta_saturation": round(ds, 2),
                "delta_value": round(dv, 2), "magnitude": round(magnitude, 2),
                "ring_px_visit1": rel1["ring_px"], "ring_px_visit2": rel2["ring_px"],
                "notable": color_notable,
            }

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
            "comparability": base_comparability,
            "match_iou": round(float(iou), 4),
            "area_visit1_px": c1["area"],
            "area_visit2_px": c2["area"],
            "area_change_pct": round(change_pct, 2),
            "notable_area_change": notable_area,
            "color_shift": color,
            "edge_irregularity": edge,
            "confidence": round(confidence, 4),
            "reliability": reliability_label(confidence, cfg),
            "reason": None,
            "message": _message("matched", base_comparability, change_pct, notable_area,
                                confidence, color_notable, edge_notable),
        })

    for i in only1:
        partial = touches_boundary(comps1[i]["mask"], valid_mask, margin)
        lesion = _blank_lesion(
            len(lesions), "resolved", PARTIAL if partial else base_comparability,
            confidence, cfg, comps1[i]["area"], 0,
            reason="lesion reaches the edge of the region visible in both photos"
            if partial else None,
        )
        if not partial:
            lesion["area_change_pct"] = -100.0
            lesion["notable_area_change"] = True
        lesions.append(lesion)

    for j in only2:
        partial = touches_boundary(comps2[j]["mask"], valid_mask, margin)
        lesion = _blank_lesion(
            len(lesions), "new", PARTIAL if partial else base_comparability,
            confidence, cfg, 0, comps2[j]["area"],
            reason="lesion reaches the edge of the region visible in both photos"
            if partial else None,
        )
        if not partial:
            lesion["notable_area_change"] = True
        lesions.append(lesion)

    report["lesions"] = lesions
    comparable = [l for l in lesions if l["comparability"] == FULL]
    partials = [l for l in lesions if l["comparability"] == PARTIAL]
    notable = [l for l in comparable
               if l["notable_area_change"]
               or (l["color_shift"] or {}).get("notable")
               or (l["edge_irregularity"] or {}).get("notable")]

    if notable:
        headline = f"{len(notable)} area(s) changed noticeably since the last visit - {DISCLAIMER}"
    elif partials and not comparable:
        headline = ("The area of interest is only partly visible in the newer photo, so it "
                    "could not be compared. Please retake it with the whole area in frame.")
    else:
        headline = "No notable change since the last visit. Keep taking photos at each visit."

    report["summary"] = {
        "comparable": True,
        "n_lesions_visit1": len(comps1),
        "n_lesions_visit2": len(comps2),
        "n_matched": len(pairs),
        "n_new": len(only2),
        "n_resolved": len(only1),
        "n_comparable": len(comparable),
        "n_partial_out_of_frame": len(partials),
        "n_notable_changes": len(notable),
        "any_notable_change": bool(notable),
        "headline": headline,
    }
    return report
