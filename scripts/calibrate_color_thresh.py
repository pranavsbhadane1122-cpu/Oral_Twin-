"""Calibrate delta.relative_color_thresh on the no-change longitudinal pairs.

The threshold is the 95th percentile of the relative-colour magnitude measured
on pairs where NO lesion change was injected - i.e. the level that only 5% of
genuinely-unchanged lesions exceed - rounded up to the next whole unit.

Prints the distribution so the choice is auditable, and writes
data/processed/delta/color_calibration.json for the validation report to cite.

Usage: python scripts/calibrate_color_thresh.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.delta.compare import FULL  # noqa: E402
from src.delta.run_delta import paths_for, run_pair  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    names = [p.name for p in sorted(long_root.glob("pair_*")) if p.is_dir()]

    mags = []
    for i, name in enumerate(names, 1):
        meta = json.loads((paths_for(name, cfg)["masks"] / "meta.json").read_text("utf-8"))
        if meta["injected_lesion_change"]:
            continue
        report, _ = run_pair(name, cfg)
        for lesion in report["lesions"]:
            if lesion["comparability"] != FULL:
                continue
            color = lesion.get("color_shift") or {}
            if color.get("measurable"):
                mags.append(color["magnitude"])
        if i % 50 == 0:
            print(f"  {i}/{len(names)}")

    m = np.array(mags, float)
    if not len(m):
        raise SystemExit("no measurable no-change lesions found")

    p95 = float(np.percentile(m, 95))
    chosen = float(np.ceil(p95))
    out = {
        "basis": "95th percentile of relative-colour magnitude on no-change pairs, "
                 "rounded up to the next whole unit",
        "n_no_change_lesions": int(len(m)),
        "median": round(float(np.median(m)), 3),
        "p75": round(float(np.percentile(m, 75)), 3),
        "p90": round(float(np.percentile(m, 90)), 3),
        "p95": round(p95, 3),
        "p99": round(float(np.percentile(m, 99)), 3),
        "max": round(float(m.max()), 3),
        "chosen_threshold": chosen,
    }
    print(json.dumps(out, indent=2))
    for t in (p95, chosen):
        print(f"  at threshold {t:.2f}: {100.0 * (m >= t).mean():.1f}% of no-change "
              f"lesions would be flagged")

    dest = PROJECT_ROOT / cfg["paths"]["processed"] / "delta"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "color_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest / 'color_calibration.json'}")
    print(f"set configs/config.yaml delta.relative_color_thresh: {chosen}")


if __name__ == "__main__":
    main()
