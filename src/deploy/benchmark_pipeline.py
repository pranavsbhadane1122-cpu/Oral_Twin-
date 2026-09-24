"""CPU benchmark for the pipeline that actually runs today.

The classifier is parked, so the working system is the non-neural path:
alignment (ORB + CLAHE + RANSAC) followed by change detection. This measures
what that costs per visit pair on this machine - wall-clock per stage and peak
process memory - which is the real on-device feasibility evidence for the
modules that passed their gates.

Unlike the quantisation harness, these numbers describe code that is validated
and in use.

Usage: python -m src.deploy.benchmark_pipeline [--pairs N]
"""

import argparse
import gc
import time
import tracemalloc

import cv2
import numpy as np

from src.delta.compare import compare, overlap_valid_mask
from src.demo import masks as mask_src
from src.risk.report import build_report
from src.risk.risk_rules import stratify
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode

try:
    import psutil
except ImportError:
    psutil = None


def peak_rss_mb():
    if psutil is None:
        return None
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or getattr(info, "rss", 0)
    return peak / 1e6


def summarise(name, values, unit="ms"):
    a = np.asarray(values, dtype=float)
    return (f"  {name:<28} mean {a.mean():7.1f} {unit}   median {np.median(a):7.1f}   "
            f"p95 {np.percentile(a, 95):7.1f}   max {a.max():7.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    from src.alignment.align import align_pair

    n_pairs = args.pairs or cfg["deploy"]["benchmark_pairs"]
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    pairs = [p for p in sorted(long_root.glob("pair_*")) if p.is_dir()][:n_pairs]
    if not pairs:
        raise SystemExit("no longitudinal pairs found")

    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "deploy"
    out_dir.mkdir(parents=True, exist_ok=True)

    timings = {"read": [], "align": [], "masks": [], "delta": [], "risk": [], "total": []}
    refused = 0
    gc.collect()
    tracemalloc.start()
    rss_before = peak_rss_mb()

    for pair in pairs:
        p1, p2 = pair / "visit1.jpg", pair / "visit2.jpg"
        start_total = time.perf_counter()

        t = time.perf_counter()
        img1, img2 = imread_unicode(p1), imread_unicode(p2)
        timings["read"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        alignment = align_pair(p1, p2, cfg)
        timings["align"].append((time.perf_counter() - t) * 1000)

        if alignment["H"] is None:
            refused += 1
            timings["total"].append((time.perf_counter() - start_total) * 1000)
            continue

        H = np.asarray(alignment["H"], float)
        warped = alignment["warped"]

        t = time.perf_counter()
        loaded = mask_src.load_synthetic_masks(pair.name, cfg)
        if loaded is None:
            mask1 = mask_src.placeholder_mask(img1, cfg)
            mask2_warped = mask_src.placeholder_mask(warped, cfg)
        else:
            mask1, mask2 = loaded
            mask2_warped = cv2.warpPerspective(mask2, H, (img1.shape[1], img1.shape[0]),
                                               flags=cv2.INTER_NEAREST)
        valid = overlap_valid_mask(img1.shape, H, img2.shape)
        timings["masks"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        report = compare(mask1, mask2_warped, img1, warped,
                         alignment["confidence"], cfg, valid_mask=valid)
        timings["delta"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        report["pair"] = pair.name
        risk = stratify(report, None, None, alignment["confidence"], cfg)
        build_report(pair.name, report, risk, None, None, {}, alignment, cfg=cfg)
        timings["risk"].append((time.perf_counter() - t) * 1000)

        timings["total"].append((time.perf_counter() - start_total) * 1000)

    traced_peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    rss_after = peak_rss_mb()

    image_shape = imread_unicode(pairs[0] / "visit1.jpg").shape
    lines = [
        "=" * 78,
        "CPU BENCHMARK - ALIGNMENT + CHANGE DETECTION (the working pipeline)",
        "=" * 78,
        "",
        "These modules passed their Phase 4 and Phase 5 gates and are in use. The",
        "numbers below are the on-device feasibility evidence for them. No neural",
        "network runs in this path.",
        "",
        f"pairs processed:   {len(pairs)}  ({refused} refused at alignment)",
        f"image size:        {image_shape[1]}x{image_shape[0]}",
        f"ORB features:      {cfg['alignment']['n_features']}",
        "",
        "per-pair wall clock:",
        summarise("read both photographs", timings["read"]),
        summarise("alignment (ORB+RANSAC)", timings["align"]),
        summarise("masks + overlap region", timings["masks"]),
        summarise("change detection", timings["delta"]),
        summarise("risk band + report text", timings["risk"]),
        summarise("TOTAL per pair", timings["total"]),
        "",
    ]
    total = np.asarray(timings["total"], dtype=float)
    lines += [
        f"throughput:        {1000.0 / total.mean():.1f} pairs/second on one CPU core",
        f"python heap peak:  {traced_peak:.1f} MB (tracemalloc, Python allocations only)",
    ]
    if rss_after is not None:
        lines.append(f"process peak RSS:  {rss_after:.1f} MB "
                     f"(includes the Python interpreter and OpenCV, {rss_before:.1f} MB "
                     f"before the run)")
    else:
        lines.append("process peak RSS:  psutil not installed, not measured")
    lines += [
        "",
        "Read with care: this is desktop CPU, not phone silicon, and the photographs",
        "are the 224x224 processed copies. A phone would need its own measurement.",
        "",
        "OralTwin is a screening aid, not a diagnostic tool.",
        "",
    ]
    report_text = "\n".join(lines)
    (out_dir / "pipeline_benchmark.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"saved {out_dir / 'pipeline_benchmark.txt'}")


if __name__ == "__main__":
    main()
