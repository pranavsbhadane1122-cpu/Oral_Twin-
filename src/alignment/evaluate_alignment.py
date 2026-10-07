"""Validate alignment against the known ground-truth homographies.

For every data/longitudinal/pair_XXXX: run align_pair, then compare the
estimated homography to ground truth by CORNER ERROR - project the four image
corners through both matrices and take the mean L2 distance in pixels. Matrix
entries are never compared directly (homographies are scale-ambiguous).

Ground truth stores H_gt mapping visit1 -> visit2; alignment estimates H_est
mapping visit2 -> visit1, so the comparison uses inv(H_gt).

Outputs (data/processed/alignment/):
  report.txt                 - aggregate numbers
  confidence_vs_error.png    - scatter of confidence against corner error
  debug/best_XX_pair_YYYY.jpg, debug/worst_XX_pair_YYYY.jpg
                             - visit1 | warped visit2 | 50% blend overlay

Usage: python -m src.alignment.evaluate_alignment
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.alignment.align import align_pair, output_dir_for, save_result
from src.alignment.homography import corner_error
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode, imwrite_jpg

DISCLAIMER = ("NOTE: OralTwin is a screening aid, not a diagnostic tool. Any finding "
              "must be checked by a dentist.")
N_DEBUG = 5


def labelled(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def debug_triptych(img1, warped, path, caption):
    """visit1 | warped visit2 | 50% blend, with a caption strip."""
    blend = cv2.addWeighted(img1, 0.5, warped, 0.5, 0.0)
    strip = np.hstack([
        labelled(img1, "visit1"),
        labelled(warped, "visit2 warped"),
        labelled(blend, "50% blend"),
    ])
    banner = np.zeros((26, strip.shape[1], 3), np.uint8)
    cv2.putText(banner, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    imwrite_jpg(path, np.vstack([banner, strip]))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main(cfg=None, out_root=None, write_artifacts=True):
    """Evaluate alignment and return the metrics dict.

    cfg/out_root are parameters so the region-source A/B can run this exact
    code twice rather than a second implementation that might drift from it.
    """
    cfg = cfg or load_config()
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    out_root = (Path(out_root) if out_root else
                PROJECT_ROOT / cfg["paths"]["processed"] / "alignment")
    debug_dir = out_root / "debug"
    out_root.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    pairs = sorted(p for p in long_root.glob("pair_*") if p.is_dir())
    print(f"evaluating {len(pairs)} pairs ...")

    rows, reasons = [], {}
    for i, pair in enumerate(pairs, 1):
        gt = json.loads((pair / "ground_truth.json").read_text(encoding="utf-8"))
        result = align_pair(pair / "visit1.jpg", pair / "visit2.jpg", cfg)
        save_result(result, output_dir_for(pair, cfg))

        h, w = result["image_shape"]
        # ground truth maps visit1 -> visit2; we estimate visit2 -> visit1
        H_true = np.linalg.inv(np.asarray(gt["homography"], float))
        err = (corner_error(np.asarray(result["H"], float), H_true, w, h)
               if result["H"] is not None else None)

        rows.append({
            "pair": pair.name, "aligned": result["H"] is not None,
            "confidence": result["confidence"], "corner_error": err,
            "reason": result["reason"], "n_inliers": result["info"]["n_inliers"],
            "has_lesion_change": gt["lesion_change"] is not None,
        })
        if result["reason"]:
            key = result["reason"].split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        if i % 25 == 0:
            print(f"  {i}/{len(pairs)}")

    aligned = [r for r in rows if r["aligned"]]
    errors = np.array([r["corner_error"] for r in aligned], float)
    confs = np.array([r["confidence"] for r in aligned], float)

    pct_aligned = 100.0 * len(aligned) / max(len(rows), 1)
    pct_under10 = 100.0 * float((errors < 10).mean()) if len(errors) else 0.0
    corr = pearson(confs, errors)
    # correlation over ALL pairs (rejected = confidence 0, error counted as worst)
    all_conf = np.array([r["confidence"] for r in rows], float)
    worst = float(errors.max()) if len(errors) else 0.0
    all_err = np.array([r["corner_error"] if r["aligned"] else worst for r in rows], float)
    corr_all = pearson(all_conf, all_err)

    lines = [
        "=" * 70, "ALIGNMENT VALIDATION REPORT", "=" * 70,
        f"pairs evaluated:            {len(rows)}",
        f"aligned (not rejected):     {len(aligned)} ({pct_aligned:.1f}%)",
        f"rejected:                   {len(rows) - len(aligned)}",
        "",
        "corner error over aligned pairs (px, mean L2 of 4 projected corners):",
        f"  mean:                     {errors.mean():.2f}" if len(errors) else "  mean: n/a",
        f"  median:                   {np.median(errors):.2f}" if len(errors) else "  median: n/a",
        f"  p90:                      {np.percentile(errors, 90):.2f}" if len(errors) else "  p90: n/a",
        f"  max:                      {errors.max():.2f}" if len(errors) else "  max: n/a",
        f"  under 10px:               {pct_under10:.1f}% of aligned pairs",
        "",
        f"Pearson r(confidence, corner error), aligned pairs: {corr:+.3f}",
        f"Pearson r(confidence, corner error), all pairs:     {corr_all:+.3f}",
        "  (want strongly negative: higher confidence -> lower error)",
        "",
    ]
    if reasons:
        lines.append("rejection reasons:")
        lines += [f"  {k}: {n}" for k, n in sorted(reasons.items(), key=lambda kv: -kv[1])]
        lines.append("")

    lesion = [r for r in aligned if r["has_lesion_change"]]
    plain = [r for r in aligned if not r["has_lesion_change"]]
    if lesion and plain:
        lines += [
            "median corner error by simulated lesion change:",
            f"  with lesion change:       {np.median([r['corner_error'] for r in lesion]):.2f} px "
            f"({len(lesion)} pairs)",
            f"  without:                  {np.median([r['corner_error'] for r in plain]):.2f} px "
            f"({len(plain)} pairs)", "",
        ]

    lines += [DISCLAIMER, ""]
    report = "\n".join(lines)
    metrics = {
        "n_pairs": len(rows), "n_aligned": len(aligned), "pct_aligned": pct_aligned,
        "median_corner_error": float(np.median(errors)) if len(errors) else None,
        "p90_corner_error": float(np.percentile(errors, 90)) if len(errors) else None,
        "mean_corner_error": float(errors.mean()) if len(errors) else None,
        "pct_under_10px": pct_under10,
        "corr_aligned": corr, "corr_all": corr_all,
        "rows": rows, "report": report,
    }
    if not write_artifacts:
        return metrics
    (out_root / "report.txt").write_text(report, encoding="utf-8")
    (out_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if len(errors):
        ax.scatter(confs, errors, s=18, alpha=0.7, edgecolor="none")
    rej = [r for r in rows if not r["aligned"]]
    if rej:
        ax.scatter([0] * len(rej), [worst] * len(rej), s=22, marker="x", color="crimson",
                   label=f"rejected ({len(rej)})")
        ax.legend(loc="upper right")
    ax.set_xlabel("confidence")
    ax.set_ylabel("corner error (px)")
    ax.set_title(f"Confidence vs corner error (r={corr:+.3f})")
    ax.set_yscale("symlog", linthresh=10)
    ax.axhline(10, ls="--", lw=1, color="grey")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_root / "confidence_vs_error.png", dpi=150)
    plt.close(fig)

    # debug triptychs for the 5 best and 5 worst aligned pairs
    for f in debug_dir.glob("*.jpg"):
        f.unlink()
    ranked = sorted(aligned, key=lambda r: r["corner_error"])
    for tag, subset in (("best", ranked[:N_DEBUG]), ("worst", list(reversed(ranked[-N_DEBUG:])))):
        for rank, r in enumerate(subset, 1):
            pair = long_root / r["pair"]
            warped = imread_unicode(output_dir_for(pair, cfg) / "warped.jpg")
            img1 = imread_unicode(pair / "visit1.jpg")
            if warped is None or img1 is None:
                continue
            caption = (f"{tag} #{rank}  {r['pair']}  corner_err={r['corner_error']:.2f}px  "
                       f"conf={r['confidence']:.3f}  inliers={r['n_inliers']}")
            debug_triptych(img1, warped, debug_dir / f"{tag}_{rank:02d}_{r['pair']}.jpg", caption)

    print(report)
    print(f"report:  {out_root / 'report.txt'}")
    print(f"scatter: {out_root / 'confidence_vs_error.png'}")
    print(f"debug images: {debug_dir}")
    return metrics


if __name__ == "__main__":
    main()
