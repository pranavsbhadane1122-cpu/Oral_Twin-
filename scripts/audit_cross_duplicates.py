"""Does any image appear in more than one dataset? READ-ONLY on data/raw.

Checks every pair of datasets two ways: identical bytes (MD5) and visually
near-identical (perceptual hash within a small Hamming distance). An image
present in two sources would otherwise be able to cross a train/test boundary
no matter how carefully each source is split.

Usage: python scripts/audit_cross_duplicates.py
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import hamming_pairs, imread_unicode  # noqa: E402



NCC_MIN = 0.90
INLIER_MIN = 30


def verify_same_scene(path_a, path_b):
    """A matching pHash is only a candidate. Confirm with normalised correlation
    and a RANSAC-filtered ORB match, which a coincidental hash collision fails."""
    a, b = imread_unicode(path_a), imread_unicode(path_b)
    if a is None or b is None:
        return None, None, False
    g1 = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (256, 256))
    g2 = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (256, 256))
    ncc = float(cv2.matchTemplate(g1.astype(np.float32), g2.astype(np.float32),
                                  cv2.TM_CCOEFF_NORMED)[0][0])
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    inliers = 0
    if d1 is not None and d2 is not None and len(k1) > 8 and len(k2) > 8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        good = [m for m, n in matcher.knnMatch(d1, d2, k=2)
                if m.distance < 0.75 * n.distance]
        if len(good) >= 8:
            src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            inliers = int(mask.sum()) if mask is not None else 0
    return ncc, inliers, (ncc > NCC_MIN and inliers >= INLIER_MIN)


def main():
    cfg = load_config()
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    cache_path = processed / "audit_cache" / "hashes.json"
    out = processed / "cross_dataset_duplicates.txt"
    threshold = cfg["audit"]["cross_dataset_phash_threshold"]
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("CROSS-DATASET DUPLICATE CHECK (data/raw is read-only)")
    if not cache_path.exists():
        log("hash cache missing - run scripts/audit_hashes.py first")
        out.write_text("\n".join(lines), encoding="utf-8")
        return

    blob = json.loads(cache_path.read_text(encoding="utf-8"))
    datasets = {name: blob[name]["entries"] for name in blob}
    log(f"\ndatasets compared: "
        f"{ {k: len(v) for k, v in datasets.items()} }")
    log(f"near-duplicate threshold: Hamming <= {threshold} on a 64-bit pHash")

    md5_index = {name: defaultdict(list) for name in datasets}
    for name, entries in datasets.items():
        for e in entries:
            md5_index[name][e["md5"]].append(e["path"])

    log(f"\n{'=' * 78}\nEXACT MATCHES (MD5)\n{'=' * 78}")
    any_exact = False
    for a, b in combinations(datasets, 2):
        shared = set(md5_index[a]) & set(md5_index[b])
        n_files = sum(len(md5_index[a][h]) for h in shared)
        status = f"{len(shared)} shared hashes ({n_files} files in {a})"
        log(f"  {a:<16} vs {b:<16} {status}")
        if shared:
            any_exact = True
            for h in list(shared)[:5]:
                log(f"      {h[:10]}  {a}: {md5_index[a][h][0]}")
                log(f"      {' ' * 10}  {b}: {md5_index[b][h][0]}")
    if not any_exact:
        log("  no byte-identical images shared between any two datasets")

    log(f"\n{'=' * 78}\nNEAR-DUPLICATES (pHash)\n{'=' * 78}")
    hashes = {name: np.asarray([int(e["phash"]) for e in entries], dtype=np.uint64)
              for name, entries in datasets.items()}
    paths = {name: [e["path"] for e in entries] for name, entries in datasets.items()}

    for a, b in combinations(datasets, 2):
        pairs = []
        for i, j in hamming_pairs(hashes[a], hashes[b], threshold):
            pairs.append((paths[a][i], paths[b][j]))
            if len(pairs) >= 5000:
                break
        left = len({p for p, _ in pairs})
        right = len({q for _, q in pairs})
        log(f"  {a:<16} vs {b:<16} {len(pairs)} candidate pairs "
            f"({left} images in {a}, {right} in {b})")
        if not pairs:
            continue
        root_a = Path(blob[a]["root"])
        root_b = Path(blob[b]["root"])
        confirmed = []
        log(f"      verifying each candidate (NCC > {NCC_MIN}, "
            f">= {INLIER_MIN} RANSAC inliers):")
        for p, q in pairs[:200]:
            ncc, inliers, same = verify_same_scene(root_a / p, root_b / q)
            flag = "SAME SCENE" if same else "collision"
            if same:
                confirmed.append((p, q))
            log(f"        NCC {ncc:.3f}  inliers {inliers:>4}  {flag}")
            log(f"          {a}: {p}")
            log(f"          {b}: {q}")
        log(f"      CONFIRMED same-scene pairs: {len(confirmed)} of {len(pairs)}")
        if confirmed:
            log(f"      distinct {a} images involved: "
                f"{sorted({p for p, _ in confirmed})}")
            log(f"      distinct {b} images involved: {len({q for _, q in confirmed})}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
