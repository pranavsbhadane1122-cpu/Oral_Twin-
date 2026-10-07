"""Validate change detection against the injected ground truth.

Ground truth (data/processed/delta_masks/pair_XXXX/meta.json) knows which pairs
had lesion growth injected and exactly how much area was added.

Metrics (all computed over COMPARABLE lesions only - lesions labelled
partial_out_of_frame carry no numbers by design):
  - recall: share of injected-change pairs where a notable area change is reported
  - area false-change rate: share of no-change pairs flagged anyway
  - colour false-flag rate: share of no-change pairs whose relative-colour metric
    fires (the amended target, <= 10%)
  - MAE between measured area_change_% and injected growth
  - out-of-frame accounting, including the six pairs known to have lost lesion
    area off the edge of the newer photo, which must now all be labelled partial
  - the same metrics restricted to the lowest-confidence quartile

Outputs (data/processed/delta/):
  report.txt, results.json, measured_vs_injected.png,
  debug/*.jpg - visit1 + mask1 contour | warped visit2 + mask2 contour + annotation

Usage: python -m src.delta.evaluate_delta
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.delta.compare import DISCLAIMER, FULL, PARTIAL
from src.delta.run_delta import paths_for, run_pair, save_report
from src.delta.synthetic_masks import blob_mask
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imwrite_jpg

N_DEBUG = 5
# a lesion counts as genuinely out of frame when this share of it lands outside
# visit2's frame under the GROUND-TRUTH homography
OUT_OF_FRAME_TOLERANCE = 0.02


def ground_truth_out_of_frame(pair_name, cfg):
    """Independent check: does the lesion actually leave visit2's frame?

    Uses the ground-truth homography and the deterministic mask generator, so
    it never consults the module under test. Returns the largest share of the
    lesion (either visit) that falls outside the newer photo.
    """
    p = paths_for(pair_name, cfg)
    gt = json.loads((p["pair"] / "ground_truth.json").read_text("utf-8"))
    meta = json.loads((p["masks"] / "meta.json").read_text("utf-8"))
    mask1 = cv2.imread(str(p["masks"] / "mask1.png"), cv2.IMREAD_GRAYSCALE)
    h, w = mask1.shape

    center = tuple(meta["center_xy"])
    radius = meta["base_radius_px"]
    mask2_v1 = blob_mask(mask1.shape, center, radius * meta["enlarge_factor"],
                         f"{pair_name}-lesion", cfg)
    H_gt = np.asarray(gt["homography"], float)  # visit1 -> visit2

    worst = 0.0
    for mask in (mask1, mask2_v1):
        ys, xs = np.nonzero(mask > 127)
        if not len(xs):
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(pts, H_gt).reshape(-1, 2)
        outside = ((moved[:, 0] < 0) | (moved[:, 0] > w - 1)
                   | (moved[:, 1] < 0) | (moved[:, 1] > h - 1))
        worst = max(worst, float(outside.mean()))
    return worst


def comparable_lesions(report):
    return [l for l in report["lesions"] if l["comparability"] == FULL]


def largest_comparable_lesion(report):
    lesions = [l for l in comparable_lesions(report) if l["state"] == "matched"]
    return max(lesions, key=lambda l: l["area_visit1_px"]) if lesions else None


def annotate(img, mask, color, label):
    out = img.copy()
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def debug_overlay(extras, report, measured, injected, path, tag=""):
    img1, mask1 = extras["img1"], extras["mask1"]
    img2w, mask2w = extras["img2_warped"], extras["mask2_warped"]
    if img2w is None:
        img2w, mask2w = np.zeros_like(img1), np.zeros_like(mask1)

    left = annotate(img1, mask1, (0, 255, 0), "visit1 + mask1")
    right = annotate(img2w, mask2w, (0, 165, 255), "visit2 warped + mask2")
    valid = extras.get("valid_mask")
    if valid is not None:
        contours, _ = cv2.findContours((valid > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, (255, 255, 0), 1)
        cv2.drawContours(right, contours, -1, (255, 255, 0), 1)
    strip = np.hstack([left, right])

    lesions = report["lesions"]
    msg = lesions[0]["message"] if lesions else (report.get("reason") or "no lesion")
    comparability = lesions[0]["comparability"] if lesions else "n/a"
    banner = np.zeros((52, strip.shape[1], 3), np.uint8)
    m_txt = "n/a" if measured is None else f"{measured:+.1f}%"
    i_txt = "n/a" if injected is None else f"{injected:+.1f}%"
    cv2.putText(banner, f"{tag}{report['pair']}  measured {m_txt}  injected {i_txt}  "
                        f"conf {report['alignment_confidence']:.2f}  [{comparability}]",
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, msg[:100], (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1,
                cv2.LINE_AA)
    imwrite_jpg(path, np.vstack([banner, strip]))


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def collect(cfg):
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    names = [p.name for p in sorted(long_root.glob("pair_*")) if p.is_dir()]
    print(f"evaluating {len(names)} pairs ...")

    rows, cache = [], {}
    for i, name in enumerate(names, 1):
        meta = json.loads((paths_for(name, cfg)["masks"] / "meta.json").read_text("utf-8"))
        report, extras = run_pair(name, cfg)
        save_report(report, cfg)

        lesion = largest_comparable_lesion(report)
        comp = comparable_lesions(report)
        partials = [l for l in report["lesions"] if l["comparability"] == PARTIAL]
        color_flag = any((l.get("color_shift") or {}).get("notable") for l in comp)
        color_measurable = any((l.get("color_shift") or {}).get("measurable") for l in comp)

        gt_off = ground_truth_out_of_frame(name, cfg)
        rows.append({
            "pair": name,
            "injected": bool(meta["injected_lesion_change"]),
            "gt_out_of_frame_frac": round(gt_off, 4),
            "gt_out_of_frame": gt_off > OUT_OF_FRAME_TOLERANCE,
            "injected_pct": meta["injected_area_change_pct"],
            "measured_pct": lesion["area_change_pct"] if lesion else None,
            "confidence": report["alignment_confidence"],
            "status": report["status"],
            "n_comparable": len(comp),
            "n_partial": len(partials),
            "any_partial": bool(partials),
            "flagged_area": any(l["notable_area_change"] for l in comp),
            "flagged_color": color_flag,
            "color_measurable": color_measurable,
            "flagged_edge": any((l.get("edge_irregularity") or {}).get("notable") for l in comp),
            "flagged_any": bool(report["summary"].get("any_notable_change")),
        })
        cache[name] = (report, extras)
        if i % 25 == 0:
            print(f"  {i}/{len(names)}")
    return rows, cache


def main(cfg=None, out_root=None, write_artifacts=True):
    """Evaluate the delta module and return the metrics dict.

    cfg/out_root are parameters so the region-source A/B runs this exact
    code rather than a parallel implementation that could drift from it.
    """
    cfg = cfg or load_config()
    out_root = (Path(out_root) if out_root else
                PROJECT_ROOT / cfg["paths"]["processed"] / "delta")
    debug_dir = out_root / "debug"
    out_root.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows, cache = collect(cfg)

    changed = [r for r in rows if r["injected"]]
    unchanged = [r for r in rows if not r["injected"]]
    # recall/false-change are only meaningful where a comparable lesion exists
    changed_cmp = [r for r in changed if r["n_comparable"]]
    unchanged_cmp = [r for r in unchanged if r["n_comparable"]]
    unchanged_color = [r for r in unchanged_cmp if r["color_measurable"]]

    recall = pct(sum(r["flagged_area"] for r in changed_cmp), len(changed_cmp))
    false_area = pct(sum(r["flagged_area"] for r in unchanged_cmp), len(unchanged_cmp))
    false_color = pct(sum(r["flagged_color"] for r in unchanged_color), len(unchanged_color))
    false_edge = pct(sum(r["flagged_edge"] for r in unchanged_cmp), len(unchanged_cmp))
    false_any = pct(sum(r["flagged_any"] for r in unchanged_cmp), len(unchanged_cmp))

    both = [r for r in rows if r["measured_pct"] is not None and r["injected_pct"] is not None]
    errs = np.array([abs(r["measured_pct"] - r["injected_pct"]) for r in both], float)
    mae = float(errs.mean()) if len(errs) else float("nan")

    confs = np.array([r["confidence"] for r in rows], float)
    q1 = float(np.percentile(confs, 25))
    low = [r for r in rows if r["confidence"] <= q1]
    low_changed = [r for r in low if r["injected"] and r["n_comparable"]]
    low_unchanged = [r for r in low if not r["injected"] and r["n_comparable"]]
    low_both = [r for r in low if r["measured_pct"] is not None and r["injected_pct"] is not None]
    low_errs = np.array([abs(r["measured_pct"] - r["injected_pct"]) for r in low_both], float)

    by_name = {r["pair"]: r for r in rows}
    truly_off = [r for r in rows if r["gt_out_of_frame"]]
    caught = [r for r in truly_off if r["any_partial"]]
    missed = [r for r in truly_off if not r["any_partial"]]
    all_known_ok = not missed
    n_partial_pairs = sum(1 for r in rows if r["any_partial"])
    # pairs labelled partial that ground truth says are fully visible
    over_flagged = [r for r in rows if r["any_partial"] and not r["gt_out_of_frame"]]

    calib_path = out_root / "color_calibration.json"
    calib = json.loads(calib_path.read_text("utf-8")) if calib_path.exists() else {}

    lines = [
        "=" * 72, "DELTA / CHANGE-DETECTION VALIDATION REPORT (post-fix)", "=" * 72,
        "Validated against SYNTHETIC masks derived from the longitudinal ground",
        "truth (Phase 3 segmentation is still pending external data).", "",
        f"pairs evaluated:                 {len(rows)}",
        f"  with injected lesion growth:   {len(changed)}",
        f"  with no change injected:       {len(unchanged)}",
        f"  refused as unreliable:         {sum(1 for r in rows if r['status'] == 'unreliable')}",
        "",
        "PRIMARY METRICS (comparable lesions only)",
        f"  recall on injected changes:    {recall:.1f}%   (target >= 90%, n={len(changed_cmp)})",
        f"  area false-change rate:        {false_area:.1f}%   (target <= 10%, n={len(unchanged_cmp)})",
        f"  colour false-flag rate:        {false_color:.1f}%   (AMENDED target <= 10%, "
        f"n={len(unchanged_color)})",
        f"  MAE measured vs injected:      {mae:.2f} pp   (target <= 10 pp, n={len(both)})",
        "",
        "SECONDARY",
        f"  edge false-flag rate:          {false_edge:.1f}%",
        f"  any-signal false-flag rate:    {false_any:.1f}%",
        "",
        "measured-vs-injected error distribution (percentage points):",
        f"  median: {np.median(errs):.2f}   p90: {np.percentile(errs, 90):.2f}   "
        f"max: {errs.max():.2f}" if len(errs) else "  n/a",
        "",
        "OUT-OF-FRAME HANDLING (amended DoD (b))",
        "Ground truth here is independent of the module under test: the lesion is",
        "warped by the GROUND-TRUTH homography and we count what leaves visit2's",
        f"frame (more than {100 * OUT_OF_FRAME_TOLERANCE:.0f}% counts as out of frame).",
        f"  pairs genuinely partly out of frame:    {len(truly_off)}",
        f"  of those, labelled partial by module:   {len(caught)}/{len(truly_off)}"
        f"{'  OK' if all_known_ok else '  *** MISSED ***'}",
        f"  pairs labelled partial in total:        {n_partial_pairs} "
        f"({pct(n_partial_pairs, len(rows)):.1f}%)",
        f"  labelled partial but fully visible:     {len(over_flagged)} "
        f"(conservative: costs a comparison, never a false reading)",
    ]
    if missed:
        lines.append("  MISSED pairs:")
        for r in missed:
            lines.append(f"    {r['pair']}: {100*r['gt_out_of_frame_frac']:.1f}% outside "
                         f"visit2 but labelled comparable")
    worst_off = sorted(truly_off, key=lambda r: -r["gt_out_of_frame_frac"])[:5]
    if worst_off:
        lines.append("  most-clipped pairs (share of lesion outside visit2):")
        for r in worst_off:
            lines.append(f"    {r['pair']}: {100*r['gt_out_of_frame_frac']:.1f}% outside, "
                         f"labelled {'partial' if r['any_partial'] else 'COMPARABLE'}")

    lines += [
        "",
        "COLOUR THRESHOLD CALIBRATION (amended DoD (a))",
        f"  configured relative_color_thresh: {cfg['delta']['relative_color_thresh']}",
    ]
    if calib:
        lines += [
            f"  basis: {calib['basis']}",
            f"  no-change relative-colour magnitudes (n={calib['n_no_change_lesions']}): "
            f"median {calib['median']}, p90 {calib['p90']}, p95 {calib['p95']}, "
            f"max {calib['max']}",
        ]
    lines += [
        "",
        f"LOWEST-CONFIDENCE QUARTILE (confidence <= {q1:.3f}, n={len(low)})",
        f"  recall:                        {pct(sum(r['flagged_area'] for r in low_changed), len(low_changed)):.1f}%",
        f"  area false-change rate:        {pct(sum(r['flagged_area'] for r in low_unchanged), len(low_unchanged)):.1f}%",
        f"  MAE:                           {low_errs.mean():.2f} pp (n={len(low_both)})"
        if len(low_errs) else "  MAE: n/a",
        "",
        f"confidence range: {confs.min():.3f} - {confs.max():.3f} (median {np.median(confs):.3f})",
        "",
        DISCLAIMER, "",
    ]
    report_txt = "\n".join(lines)
    metrics = {
        "recall": recall, "false_area": false_area, "false_color": false_color,
        "false_edge": false_edge, "false_any": false_any, "mae": mae,
        "n_changed_comparable": len(changed_cmp),
        "n_unchanged_color": len(unchanged_color),
        "n_both": len(both),
        "n_truly_out_of_frame": len(truly_off), "n_caught": len(caught),
        "n_missed": len(missed), "n_partial_pairs": n_partial_pairs,
        "n_over_flagged": len(over_flagged),
        "rows": rows, "report": report_txt,
    }
    if not write_artifacts:
        return metrics
    (out_root / "report.txt").write_text(report_txt, encoding="utf-8")
    (out_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # scatter: measured vs injected
    fig, ax = plt.subplots(figsize=(6, 5))
    if both:
        xs = [r["injected_pct"] for r in both]
        ys = [r["measured_pct"] for r in both]
        cs = [r["confidence"] for r in both]
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=22, alpha=0.85, edgecolor="none")
        fig.colorbar(sc, label="alignment confidence")
        lim = [min(xs + ys) - 5, max(xs + ys) + 5]
        ax.plot(lim, lim, ls="--", lw=1, color="grey", label="perfect")
        ax.legend(loc="upper left")
    ax.set_xlabel("injected area growth (%)")
    ax.set_ylabel("measured area change (%)")
    ax.set_title("Measured vs injected lesion growth (comparable lesions)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "measured_vs_injected.png", dpi=150)
    plt.close(fig)

    # debug overlays: worst remaining measurement errors + the out-of-frame saves
    for f in debug_dir.glob("*.jpg"):
        f.unlink()
    worst = sorted(both, key=lambda r: -abs(r["measured_pct"] - r["injected_pct"]))[:N_DEBUG]
    for rank, r in enumerate(worst, 1):
        report, extras = cache[r["pair"]]
        debug_overlay(extras, report, r["measured_pct"], r["injected_pct"],
                      debug_dir / f"worst_{rank:02d}_{r['pair']}.jpg", tag="WORST ")
    for rank, r in enumerate(sorted(truly_off, key=lambda x: -x["gt_out_of_frame_frac"])[:N_DEBUG], 1):
        report, extras = cache[r["pair"]]
        debug_overlay(extras, report, r["measured_pct"], r["injected_pct"],
                      debug_dir / f"outofframe_{rank:02d}_{r['pair']}.jpg",
                      tag="OUT-OF-FRAME ")

    print(report_txt)
    print(f"report:  {out_root / 'report.txt'}")
    print(f"scatter: {out_root / 'measured_vs_injected.png'}")
    return metrics
    print(f"debug overlays: {debug_dir}")


if __name__ == "__main__":
    main()
