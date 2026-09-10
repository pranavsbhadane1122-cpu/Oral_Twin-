"""Run explainability + risk over a representative sample of longitudinal pairs.

Picks a deliberate mix - injected change, no change, partly out of frame, and
the lowest-confidence alignments - so the outputs exercise every path, then
writes per-pair Grad-CAM overlays, a change map and a text report to
data/processed/explain/pair_XXXX/.

Usage: python scripts/run_explain_demo.py [n_pairs]
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.delta.compare import PARTIAL  # noqa: E402
from src.delta.run_delta import run_pair  # noqa: E402
from src.explainability.change_cam import change_cam, side_by_side  # noqa: E402
from src.explainability.gradcam import (  # noqa: E402
    build_grad_model,
    compute,
    load_classifier,
    overlay,
)
from src.risk.report import build_report  # noqa: E402
from src.risk.risk_rules import stratify  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import imwrite_jpg  # noqa: E402


def choose_pairs(cfg, n=10):
    """A spread across change / no-change / out-of-frame / low-confidence."""
    results = json.loads(
        (PROJECT_ROOT / cfg["paths"]["processed"] / "delta" / "results.json").read_text("utf-8")
    )
    by_conf = sorted(results, key=lambda r: r["confidence"])
    buckets = {
        "low_confidence": [r["pair"] for r in by_conf[:3]],
        "out_of_frame": [r["pair"] for r in results if r["any_partial"]][:3],
        "changed": [r["pair"] for r in results
                    if r["injected"] and r["flagged_area"] and not r["any_partial"]][:3],
        "no_change": [r["pair"] for r in results
                      if not r["injected"] and not r["flagged_area"]
                      and not r["any_partial"]][:3],
    }
    chosen, seen = [], set()
    for label, names in buckets.items():
        for name in names:
            if name not in seen and len(chosen) < n:
                chosen.append((name, label))
                seen.add(name)
    for r in results:  # top up if a bucket was short
        if len(chosen) >= n:
            break
        if r["pair"] not in seen:
            chosen.append((r["pair"], "extra"))
            seen.add(r["pair"])
    return chosen


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = load_config()
    out_root = PROJECT_ROOT / cfg["paths"]["processed"] / "explain"
    out_root.mkdir(parents=True, exist_ok=True)

    model = load_classifier(cfg)
    grad_model = build_grad_model(model, cfg=cfg)
    index = []

    for pair_name, bucket in choose_pairs(cfg, n):
        report, extras = run_pair(pair_name, cfg)
        img1 = extras["img1"]
        img2w = extras["img2_warped"]
        H = report["alignment"]["H"]
        confidence = report["alignment_confidence"]
        out_dir = out_root / pair_name
        out_dir.mkdir(exist_ok=True)
        paths = {}

        cam1 = compute(model, img1, cfg=cfg, grad_model=grad_model)
        imwrite_jpg(out_dir / "gradcam_visit1.jpg", overlay(img1, cam1["heatmap"], cfg))
        paths["Grad-CAM, previous visit"] = str(
            (out_dir / "gradcam_visit1.jpg").relative_to(PROJECT_ROOT))

        cam2 = None
        if img2w is not None:
            cam2 = compute(model, img2w, cfg=cfg, grad_model=grad_model)
            imwrite_jpg(out_dir / "gradcam_visit2.jpg", overlay(img2w, cam2["heatmap"], cfg))
            paths["Grad-CAM, this visit"] = str(
                (out_dir / "gradcam_visit2.jpg").relative_to(PROJECT_ROOT))

        ccam = change_cam(model, img1, img2w if img2w is not None else img1,
                          H, confidence, cfg=cfg, grad_model=grad_model)
        if ccam["status"] == "ok":
            imwrite_jpg(out_dir / "change_cam.jpg", ccam["overlay"])
            panel = side_by_side(img1, img2w, ccam, cfg)
            imwrite_jpg(out_dir / "explain_panel.jpg", panel)
            paths["Change map (red = more attention)"] = str(
                (out_dir / "change_cam.jpg").relative_to(PROJECT_ROOT))
            paths["Side-by-side panel"] = str(
                (out_dir / "explain_panel.jpg").relative_to(PROJECT_ROOT))

        pred1 = {"class_name": cam1["class_name"], "probability": cam1["probability"]}
        pred2 = ({"class_name": cam2["class_name"], "probability": cam2["probability"]}
                 if cam2 else None)
        risk = stratify(report, pred1, pred2, confidence, cfg)
        text = build_report(pair_name, report, risk, pred1, pred2, paths,
                            report.get("alignment"))
        (out_dir / "report.txt").write_text(text, encoding="utf-8")
        (out_dir / "risk.json").write_text(json.dumps(risk, indent=2), encoding="utf-8")

        index.append({
            "pair": pair_name, "bucket": bucket, "band": risk["band"],
            "confidence": confidence, "change_cam": ccam["status"],
            "change_cam_reason": ccam["reason"],
            "cam1_flat": cam1["flat"], "cam1_class": cam1["class_name"],
            "cam1_prob": round(cam1["probability"], 4),
            "n_partial": sum(1 for l in report.get("lesions", [])
                             if l.get("comparability") == PARTIAL),
            "rules": [r["rule"] for r in risk["rules_fired"]],
        })
        print(f"{pair_name:>11} [{bucket:<15}] band={risk['band']:<8} "
              f"conf={confidence:.2f} change_cam={ccam['status']:<7} "
              f"rules={[r['rule'] for r in risk['rules_fired']]}")

    (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nwrote {len(index)} pair folders to {out_root}")


if __name__ == "__main__":
    main()
