"""Side-by-side: HSV heuristic vs trained cavity model as the ORB region.

A measured swap, not an assumed improvement. The cavity segmenter scored
Dice 0.933 on the Piyarathne test split, but the longitudinal pairs are built
from the MIO classification test split - a different dataset entirely. An
in-domain Dice says nothing about out-of-domain behaviour, and Phase 8 already
showed a Piyarathne-trained model losing most of its sensitivity elsewhere.

Both arms run the SAME evaluator functions, with only the config key changed,
so the two columns cannot drift apart through duplicated logic.

Also sweeps delta.valid_region_source, since using the cavity as the comparable
region is a separate claim from using it as the keypoint region.

Writes models/logs/region_source_comparison.txt.

Usage: python scripts/compare_region_source.py
"""

import copy
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment import evaluate_alignment  # noqa: E402
from src.delta import evaluate_delta  # noqa: E402
from src.utils.config import load_config  # noqa: E402

ARMS = ("hsv_heuristic", "cavity_model")
VALID_REGIONS = ("frame_intersection", "cavity", "cavity_and_frame")


def variant(base, **alignment_keys):
    cfg = copy.deepcopy(base)
    cfg["alignment"].update(alignment_keys)
    return cfg


def fmt(value, spec=".2f", na="n/a"):
    return na if value is None else format(value, spec)


def delta(new, old, spec="+.2f", lower_is_better=True):
    """The change, with an explicit verdict arrow. No silent improvements."""
    if new is None or old is None:
        return "n/a"
    d = new - old
    if abs(d) < 1e-9:
        return "same"
    better = (d < 0) if lower_is_better else (d > 0)
    return f"{format(d, spec)} {'better' if better else 'WORSE'}"


def main():
    base = load_config()
    out = PROJECT_ROOT / base["paths"]["models"] / "logs" / "region_source_comparison.txt"
    scratch = PROJECT_ROOT / base["paths"]["processed"] / "region_ab"
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("=" * 78)
    log("REGION SOURCE A/B - HSV heuristic vs trained cavity model")
    log("=" * 78)
    log("")
    log("The cavity segmenter was trained on Piyarathne and scored Dice 0.933 there.")
    log("These pairs come from the MIO classification test split: a different")
    log("dataset, different cameras, different framing. This is an OUT-OF-DOMAIN")
    log("application of the model and the numbers below are the only evidence that")
    log("it transfers. Nothing here is assumed from the 0.933.")
    log("")

    align_metrics, delta_metrics, timings = {}, {}, {}

    for arm in ARMS:
        log("-" * 78)
        log(f"ARM: alignment.region_source = {arm}")
        log("-" * 78)
        cfg = variant(base, region_source=arm)
        started = time.time()
        align_metrics[arm] = evaluate_alignment.main(
            cfg, out_root=scratch / arm / "alignment", write_artifacts=False)
        delta_metrics[arm] = evaluate_delta.main(
            cfg, out_root=scratch / arm / "delta", write_artifacts=False)
        timings[arm] = time.time() - started
        log(f"  done in {timings[arm] / 60:.1f} min")
        log("")

    hsv, cav = align_metrics["hsv_heuristic"], align_metrics["cavity_model"]
    dh, dc = delta_metrics["hsv_heuristic"], delta_metrics["cavity_model"]

    # which images actually used which path
    def path_counts(rows_source):
        counts = {}
        for r in rows_source:
            for key in ("region_source_visit1", "region_source_visit2"):
                value = r.get(key)
                if value:
                    counts[value] = counts.get(value, 0) + 1
        return counts

    log("=" * 78)
    log("SIDE BY SIDE")
    log("=" * 78)
    log(f"{'metric':<40}{'heuristic':>13}{'cavity model':>15}   change")
    log("-" * 78)

    rows = [
        ("pairs aligned", cav["n_aligned"], hsv["n_aligned"], "d", False),
        ("  as % of 200", cav["pct_aligned"], hsv["pct_aligned"], ".1f", False),
        ("median corner error (px)", cav["median_corner_error"],
         hsv["median_corner_error"], ".2f", True),
        ("p90 corner error (px)", cav["p90_corner_error"],
         hsv["p90_corner_error"], ".2f", True),
        ("mean corner error (px)", cav["mean_corner_error"],
         hsv["mean_corner_error"], ".2f", True),
        ("under 10px (%)", cav["pct_under_10px"], hsv["pct_under_10px"], ".1f", False),
        ("r(confidence, error) aligned", cav["corr_aligned"],
         hsv["corr_aligned"], ".3f", True),
        ("r(confidence, error) all", cav["corr_all"], hsv["corr_all"], ".3f", True),
        ("delta recall (%)", dc["recall"], dh["recall"], ".1f", False),
        ("colour false-flag rate (%)", dc["false_color"], dh["false_color"], ".1f", True),
        ("area false-change rate (%)", dc["false_area"], dh["false_area"], ".1f", True),
        ("MAE measured vs injected (pp)", dc["mae"], dh["mae"], ".2f", True),
        ("out-of-frame caught", dc["n_caught"], dh["n_caught"], "d", False),
        ("out-of-frame missed", dc["n_missed"], dh["n_missed"], "d", True),
        ("pairs labelled partial", dc["n_partial_pairs"], dh["n_partial_pairs"], "d", True),
    ]
    for name, new, old, spec, lower_better in rows:
        log(f"{name:<40}{fmt(old, spec):>13}{fmt(new, spec):>15}   "
            f"{delta(new, old, '+' + spec, lower_better)}")

    log("")
    log("which mask each image actually used (cavity_model arm):")
    for name, count in sorted(path_counts(cav["rows"]).items(), key=lambda kv: -kv[1]):
        log(f"  {name:<56} {count}")
    log("")
    log(f"runtime: heuristic {timings['hsv_heuristic'] / 60:.1f} min, "
        f"cavity model {timings['cavity_model'] / 60:.1f} min")
    log("")

    # ----------------------------------------------- valid-region sweep
    log("=" * 78)
    log("DELTA VALID-REGION SWEEP (step 4)")
    log("=" * 78)
    log("Does using the cavity as the COMPARABLE region improve out-of-frame")
    log("detection? Run with the better alignment arm held fixed.")
    log("")
    best_arm = ("cavity_model"
                if (cav["median_corner_error"] or 1e9) < (hsv["median_corner_error"] or 1e9)
                else "hsv_heuristic")
    log(f"alignment.region_source held at: {best_arm}")
    log("")
    log(f"{'valid_region_source':<24}{'caught':>8}{'missed':>8}{'partial':>9}"
        f"{'over-flag':>11}{'recall':>9}{'MAE':>8}")
    log("-" * 78)
    for region in VALID_REGIONS:
        cfg = variant(base, region_source=best_arm)
        cfg["delta"]["valid_region_source"] = region
        m = evaluate_delta.main(cfg, out_root=scratch / f"vr_{region}",
                                write_artifacts=False)
        log(f"{region:<24}{m['n_caught']:>8}{m['n_missed']:>8}"
            f"{m['n_partial_pairs']:>9}{m['n_over_flagged']:>11}"
            f"{m['recall']:>8.1f}%{m['mae']:>8.2f}")
        delta_metrics[f"vr_{region}"] = m
    log("")
    log(f"out of {dh['n_truly_out_of_frame']} pairs that ground truth says are "
        f"genuinely partly out of frame.")
    log("")
    log("OralTwin is a screening aid, not a diagnostic tool.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
