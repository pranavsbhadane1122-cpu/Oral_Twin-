"""Validate change detection against the injected ground truth.

Ground truth (data/processed/delta_masks/pair_XXXX/meta.json) knows which pairs
had lesion growth injected and exactly how much area was added.

Metrics
  - recall: share of injected-change pairs where a notable area change is reported
  - false-change rate: share of no-change pairs where a notable area change is
    reported anyway (also reported for "any notable signal", which includes the
    colour and edge flags, since those react to the simulator's lighting jitter)
  - MAE between measured area_change_% and injected growth, over pairs where both
    exist
  - the same three restricted to the lowest-confidence quartile, to show how the
    numbers degrade when alignment is poor

Outputs (data/processed/delta/):
  report.txt, results.json, measured_vs_injected.png,
  debug/*.jpg - visit1 + mask1 contour | warped visit2 + mask2 contour | annotation

Usage: python -m src.delta.evaluate_delta
"""

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.delta.compare import DISCLAIMER
from src.delta.run_delta import paths_for, run_pair, save_report
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imwrite_jpg

N_DEBUG = 5


def largest_matched_lesion(report):
    """The matched lesion with the biggest visit1 area (the injected one)."""
    matched = [l for l in report["lesions"] if l["state"] == "matched"]
    return max(matched, key=lambda l: l["area_visit1_px"]) if matched else None


def notable_area_change(report):
    return any(l["notable_area_change"] for l in report["lesions"])


def any_notable(report):
    return bool(report["summary"].get("any_notable_change"))


def annotate(img, mask, color, label):
    out = img.copy()
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def debug_overlay(extras, report, measured, injected, path):
    img1, mask1 = extras["img1"], extras["mask1"]
    img2w, mask2w = extras["img2_warped"], extras["mask2_warped"]
    if img2w is None:
        img2w, mask2w = np.zeros_like(img1), np.zeros_like(mask1)

    left = annotate(img1, mask1, (0, 255, 0), "visit1 + mask1")
    right = annotate(img2w, mask2w, (0, 165, 255), "visit2 warped + mask2")
    strip = np.hstack([left, right])

    lesion = largest_matched_lesion(report)
    msg = lesion["message"] if lesion else (report.get("reason") or "no matched lesion")
    banner = np.zeros((52, strip.shape[1], 3), np.uint8)
    m_txt = "n/a" if measured is None else f"{measured:+.1f}%"
    i_txt = "n/a" if injected is None else f"{injected:+.1f}%"
    cv2.putText(banner, f"{report['pair']}  measured {m_txt}  injected {i_txt}  "
                        f"conf {report['alignment_confidence']:.2f}",
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, msg[:96], (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1,
                cv2.LINE_AA)
    imwrite_jpg(path, np.vstack([banner, strip]))


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def main():
    cfg = load_config()
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    out_root = PROJECT_ROOT / cfg["paths"]["processed"] / "delta"
    debug_dir = out_root / "debug"
    out_root.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    names = [p.name for p in sorted(long_root.glob("pair_*")) if p.is_dir()]
    print(f"evaluating {len(names)} pairs ...")

    rows, cache = [], {}
    for i, name in enumerate(names, 1):
        meta = json.loads((paths_for(name, cfg)["masks"] / "meta.json").read_text("utf-8"))
        report, extras = run_pair(name, cfg)
        save_report(report, cfg)

        lesion = largest_matched_lesion(report)
        rows.append({
            "pair": name,
            "injected": bool(meta["injected_lesion_change"]),
            "injected_pct": meta["injected_area_change_pct"],
            "measured_pct": lesion["area_change_pct"] if lesion else None,
            "confidence": report["alignment_confidence"],
            "status": report["status"],
            "flagged_area": notable_area_change(report),
            "flagged_any": any_notable(report),
            "n_new": report["summary"].get("n_new", 0),
            "n_resolved": report["summary"].get("n_resolved", 0),
        })
        cache[name] = (report, extras)
        if i % 25 == 0:
            print(f"  {i}/{len(names)}")

    changed = [r for r in rows if r["injected"]]
    unchanged = [r for r in rows if not r["injected"]]
    recall = pct(sum(r["flagged_area"] for r in changed), len(changed))
    false_area = pct(sum(r["flagged_area"] for r in unchanged), len(unchanged))
    false_any = pct(sum(r["flagged_any"] for r in unchanged), len(unchanged))

    both = [r for r in rows if r["measured_pct"] is not None and r["injected_pct"] is not None]
    errs = np.array([abs(r["measured_pct"] - r["injected_pct"]) for r in both], float)
    mae = float(errs.mean()) if len(errs) else float("nan")

    confs = np.array([r["confidence"] for r in rows], float)
    q1 = float(np.percentile(confs, 25))
    low = [r for r in rows if r["confidence"] <= q1]
    low_changed = [r for r in low if r["injected"]]
    low_unchanged = [r for r in low if not r["injected"]]
    low_both = [r for r in low if r["measured_pct"] is not None and r["injected_pct"] is not None]
    low_errs = np.array([abs(r["measured_pct"] - r["injected_pct"]) for r in low_both], float)

    unreliable = [r for r in rows if r["status"] == "unreliable"]
    lines = [
        "=" * 70, "DELTA / CHANGE-DETECTION VALIDATION REPORT", "=" * 70,
        "Validated against SYNTHETIC masks derived from the longitudinal ground",
        "truth (Phase 3 segmentation is still pending external data).", "",
        f"pairs evaluated:               {len(rows)}",
        f"  with injected lesion growth: {len(changed)}",
        f"  with no change injected:     {len(unchanged)}",
        f"  refused as unreliable:       {len(unreliable)}",
        "",
        "PRIMARY METRICS",
        f"  recall on injected changes:  {recall:.1f}%   (target >= 90%)",
        f"  false-change rate (area):    {false_area:.1f}%   (target <= 10%)",
        f"  false-flag rate (any signal):{false_any:.1f}%   (area + colour + edge flags)",
        f"  MAE measured vs injected:    {mae:.2f} pp   (target <= 10 pp, n={len(both)})",
        "",
        "measured-vs-injected error distribution (percentage points):",
        f"  median: {np.median(errs):.2f}   p90: {np.percentile(errs, 90):.2f}   "
        f"max: {errs.max():.2f}" if len(errs) else "  n/a",
        "",
        f"LOWEST-CONFIDENCE QUARTILE (confidence <= {q1:.3f}, n={len(low)})",
        f"  recall:                      {pct(sum(r['flagged_area'] for r in low_changed), len(low_changed)):.1f}%",
        f"  false-change rate (area):    {pct(sum(r['flagged_area'] for r in low_unchanged), len(low_unchanged)):.1f}%",
        f"  MAE:                         {low_errs.mean():.2f} pp (n={len(low_both)})"
        if len(low_errs) else "  MAE: n/a",
        "",
        f"confidence range: {confs.min():.3f} - {confs.max():.3f} (median {np.median(confs):.3f})",
        "",
        DISCLAIMER, "",
    ]
    report_txt = "\n".join(lines)
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
    ax.set_title("Measured vs injected lesion growth")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "measured_vs_injected.png", dpi=150)
    plt.close(fig)

    # debug overlays: the 5 worst measurement errors
    for f in debug_dir.glob("*.jpg"):
        f.unlink()
    worst = sorted(both, key=lambda r: -abs(r["measured_pct"] - r["injected_pct"]))[:N_DEBUG]
    for rank, r in enumerate(worst, 1):
        report, extras = cache[r["pair"]]
        debug_overlay(extras, report, r["measured_pct"], r["injected_pct"],
                      debug_dir / f"worst_{rank:02d}_{r['pair']}.jpg")

    print(report_txt)
    print(f"report:  {out_root / 'report.txt'}")
    print(f"scatter: {out_root / 'measured_vs_injected.png'}")
    print(f"debug overlays: {debug_dir}")


if __name__ == "__main__":
    main()
