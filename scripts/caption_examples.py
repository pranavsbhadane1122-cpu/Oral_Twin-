"""Render caption reports over 10 pairs chosen to cover the awkward cases.

Not a random sample: the point is to exercise good, low-confidence,
out-of-frame and poor-focus paths, because those are where a describing layer
is most likely to read like a diagnosing one.

Writes data/processed/caption_examples/ (git-ignored) and a summary table.

Usage: python scripts/caption_examples.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.caption.describe import POOR  # noqa: E402
from src.caption.render import report_for_pair  # noqa: E402
from src.utils.config import load_config  # noqa: E402

PER_CASE = 3


def main():
    cfg = load_config()
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    delta_root = PROJECT_ROOT / cfg["paths"]["processed"] / "delta"
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "caption_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # classify every pair from its existing change report
    buckets = {"low_confidence": [], "out_of_frame": [], "good": []}
    for report_path in sorted(delta_root.glob("pair_*/change_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        name = report_path.parent.name
        summary = report.get("summary", {}) or {}
        if report.get("status") != "ok" or not summary.get("comparable"):
            buckets["low_confidence"].append(name)
        elif summary.get("n_partial_out_of_frame", 0) > 0:
            buckets["out_of_frame"].append(name)
        else:
            buckets["good"].append(name)

    chosen, seen = [], set()
    for case in ("good", "out_of_frame", "low_confidence"):
        for name in buckets[case][:PER_CASE]:
            if name not in seen:
                chosen.append((case, name))
                seen.add(name)

    # top up with the blurriest remaining pairs, to exercise the focus path
    rest = [n for n in buckets["good"] + buckets["out_of_frame"] if n not in seen]
    rows = []
    for case, name in chosen:
        rows.append((case, name))
    for name in rest:
        if len(rows) >= 10:
            break
        rows.append(("extra", name))

    print(f"{'case':<16}{'pair':<12}{'view':<12}{'focus':<10}{'usable':<8}compare")
    print("-" * 74)
    summary_lines = []
    for case, name in rows:
        pair_dir = long_root / name
        if not pair_dir.exists():
            continue
        text, payload = report_for_pair(pair_dir, cfg)
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        d = payload["description"]
        c = payload["comparison"] or {}
        line = (f"{case:<16}{name:<12}{d['view']:<12}{d['focus']['band']:<10}"
                f"{str(d['usable']):<8}{c.get('state', 'n/a')}")
        print(line)
        summary_lines.append(line)
        if d["focus"]["band"] == POOR:
            summary_lines[-1] += "   <- poor focus"

    (out_dir / "_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nreports written to {out_dir}")


if __name__ == "__main__":
    main()
