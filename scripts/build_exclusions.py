"""Record images that must be excluded from every dataset build, with evidence.

The audit found photographs that appear in two sources under contradicting
labels: three MIO images filed as SANO (clinician-labelled healthy) are the same
scenes as Kaggle images filed as Gingivitis. One of the two labels is wrong and
we cannot tell which, so both copies are dropped rather than silently trusting
one source over the other.

Every row carries the evidence that justified it (normalised cross-correlation
and RANSAC inlier count against the counterpart image), so the decision can be
re-checked rather than taken on faith.

Writes data/processed/exclusions.csv. READ-ONLY on data/raw.

Usage: python scripts/build_exclusions.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import hamming_pairs  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from audit_cross_duplicates import verify_same_scene  # noqa: E402

REASON = ("label conflict: clinician-labelled healthy vs Kaggle gingivitis, "
          "same scene")
PAIR = ("mio", "oral_diseases")


def main():
    cfg = load_config()
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    cache = json.loads((processed / "audit_cache" / "hashes.json").read_text("utf-8"))
    threshold = cfg["audit"]["cross_dataset_phash_threshold"]
    out = processed / "exclusions.csv"

    left, right = PAIR
    roots = {name: Path(cache[name]["root"]) for name in PAIR}
    hashes = {name: np.asarray([int(e["phash"]) for e in cache[name]["entries"]],
                               dtype=np.uint64) for name in PAIR}
    paths = {name: [e["path"] for e in cache[name]["entries"]] for name in PAIR}

    print(f"checking {left} against {right} at Hamming <= {threshold} ...")
    best = defaultdict(dict)   # dataset -> path -> (ncc, inliers, counterpart)
    confirmed_pairs = 0
    for i, j in hamming_pairs(hashes[left], hashes[right], threshold):
        p, q = paths[left][i], paths[right][j]
        ncc, inliers, same = verify_same_scene(roots[left] / p, roots[right] / q)
        if not same:
            continue
        confirmed_pairs += 1
        if p not in best[left] or inliers > best[left][p][1]:
            best[left][p] = (ncc, inliers, q)
        if q not in best[right] or inliers > best[right][q][1]:
            best[right][q] = (ncc, inliers, p)

    print(f"confirmed same-scene pairs: {confirmed_pairs}")
    rows = []
    today = date.today().isoformat()
    for dataset, counterpart in ((left, right), (right, left)):
        for path, (ncc, inliers, other) in sorted(best[dataset].items()):
            rows.append({
                "dataset": dataset,
                "path": path,
                "reason": REASON,
                "counterpart_dataset": counterpart,
                "counterpart_path": other,
                "evidence_ncc": f"{ncc:.4f}",
                "evidence_ransac_inliers": inliers,
                "phash_threshold": threshold,
                "verified_on": today,
            })

    fields = ["dataset", "path", "reason", "counterpart_dataset", "counterpart_path",
              "evidence_ncc", "evidence_ransac_inliers", "phash_threshold", "verified_on"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    per_dataset = defaultdict(int)
    for row in rows:
        per_dataset[row["dataset"]] += 1
    print(f"\nexcluded images: {len(rows)}  {dict(per_dataset)}")
    for row in rows:
        print(f"  {row['dataset']:<14} {row['path'][:58]:<58} "
              f"NCC {row['evidence_ncc']}  inliers {row['evidence_ransac_inliers']}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
