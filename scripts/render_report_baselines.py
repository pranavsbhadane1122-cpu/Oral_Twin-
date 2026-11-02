"""Render reports to tests/baselines/, so a restructure can be diffed against them.

WHY THESE SIX. They are not a sample - they are every distinct (band, fired
rules) combination the stored reports contain, plus one refusal:

  pair_0000  Low        nothing fired      - the "no measurement crossed" path
  pair_0006  Moderate   AREA_GROWTH        - one provisional threshold
  pair_0004  Moderate   COLOUR_CHANGE      - the artefact-only threshold
  pair_0002  Elevated   AREA_GROWTH_LARGE  - the other provisional threshold
  pair_0009  Elevated   two rules          - the deciding-rules tie path
  pair_0012  refused    (induced)          - the qualitative-only path

The M4 brief said "five comparable and one refused" and then "the same seven".
Six are rendered; the count is reported rather than guessed at.

THE REFUSAL IS INDUCED. No pair in data/longitudinal fails alignment naturally -
200 of 200 succeed - so the refusal is produced by raising
delta.confidence_floor above pair_0012's genuine confidence. The baseline file
says so in its own header; a baseline that hid this would be misleading
evidence.

Usage:
  python scripts/render_report_baselines.py            # write baselines
  python scripts/render_report_baselines.py --stdout   # print, write nothing
"""

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk.report import build_report  # noqa: E402
from src.risk.risk_rules import stratify  # noqa: E402
from src.utils.config import load_config  # noqa: E402

BASELINE_DIR = PROJECT_ROOT / "tests" / "baselines"

COMPARABLE = ["pair_0000", "pair_0006", "pair_0004", "pair_0002", "pair_0009"]
REFUSED = "pair_0012"
INDUCED_FLOOR = 0.99   # above every stored confidence, so the refusal is certain


def report_path(pair):
    return (PROJECT_ROOT / "data" / "processed" / "delta" / pair
            / "change_report.json")


def render(pair, cfg, induce_refusal=False):
    change_report = json.loads(report_path(pair).read_text(encoding="utf-8"))
    run_cfg = cfg
    if induce_refusal:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["delta"]["confidence_floor"] = INDUCED_FLOOR
        confidence = float(change_report.get("alignment_confidence", 0.0))
        change_report = {
            **change_report,
            "status": "unreliable",
            "reason": (f"alignment confidence {confidence:.2f} is below the floor "
                       f"{INDUCED_FLOOR:.2f}; the two photos could not be lined up "
                       f"well enough to compare them."),
            "summary": {"comparable": False},
        }
    risk = stratify(change_report, None, None,
                    change_report.get("alignment_confidence", 0.0), run_cfg)
    text = build_report(pair, change_report, risk, None, None, {}, {}, run_cfg)
    return text, risk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for pair in COMPARABLE:
        text, risk = render(pair, cfg)
        name = f"report_{pair}.txt"
        if args.stdout:
            print(f"===== {name} (band {risk['band']}) =====\n{text}\n")
        else:
            (BASELINE_DIR / name).write_text(text, encoding="utf-8")
            written.append((name, risk["band"],
                            [r["rule"] for r in risk["rules_fired"]]))

    text, risk = render(REFUSED, cfg, induce_refusal=True)
    name = f"report_{REFUSED}_refused.txt"
    header = (
        "# INDUCED REFUSAL, not a natural one. No pair in data/longitudinal\n"
        "# fails alignment on its own (200 of 200 succeed), so this baseline was\n"
        f"# produced by raising delta.confidence_floor to {INDUCED_FLOOR} for\n"
        f"# {REFUSED} alone. Everything below the marker is the rendered report.\n"
        "# ---8<--- report starts here\n"
    )
    if args.stdout:
        print(f"===== {name} (band {risk['band']}) =====\n{header}{text}\n")
    else:
        (BASELINE_DIR / name).write_text(header + text, encoding="utf-8")
        written.append((name, risk["band"], []))

    if not args.stdout:
        print(f"wrote {len(written)} baselines to "
              f"{BASELINE_DIR.relative_to(PROJECT_ROOT)}")
        for name, band, rules in written:
            print(f"  {name:<34} band={band:<13} rules={rules}")


if __name__ == "__main__":
    main()
