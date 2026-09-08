"""Shared helpers for dataset preparation: hashing, near-duplicate clustering,
leakage-free splitting and image export.

data/raw is READ-ONLY: everything here only reads source images and writes to
data/processed.
"""

import hashlib
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def imread_unicode(path):
    """cv2.imread that tolerates spaces/unicode in Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_jpg(path, img, quality=95):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"failed to encode {path}")
    buf.tofile(str(path))


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash64(gray32):
    """64-bit perceptual hash of a 32x32 float32 grayscale image."""
    dct = cv2.dct(gray32)
    block = dct[:8, :8].flatten()
    block = np.delete(block, 0)  # drop DC term
    bits = block > np.median(block)
    h = np.uint64(0)
    for b in bits:
        h = np.uint64(h << np.uint64(1)) | np.uint64(int(b))
    return h


def dihedral_phashes(img_bgr):
    """pHashes of the image under all 8 flips/rotations, so mirrored or rotated
    augmented copies of the same photo still collide."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    hashes = []
    for flip in (False, True):
        g = np.fliplr(gray) if flip else gray
        for k in range(4):
            hashes.append(phash64(np.ascontiguousarray(np.rot90(g, k))))
    return hashes


def hamming_pairs(hashes_a, hashes_b, threshold):
    """Yield (i, j) index pairs with Hamming distance <= threshold.

    hashes_*: 1-D uint64 arrays. Chunked table-lookup popcount.
    """
    a = np.asarray(hashes_a, dtype=np.uint64)
    b = np.asarray(hashes_b, dtype=np.uint64)
    chunk = 256
    for start in range(0, len(a), chunk):
        xa = a[start:start + chunk]
        x = (xa[:, None] ^ b[None, :]).view(np.uint8).reshape(len(xa), len(b), 8)
        dist = _POPCOUNT[x].sum(axis=2)
        for i, j in zip(*np.nonzero(dist <= threshold)):
            yield start + int(i), int(j)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_near_duplicates(items, threshold, log=print):
    """items: list of dicts with key 'hashes' (list of 8 uint64).

    Returns a list of clusters (lists of item indices). Any two images whose
    dihedral pHashes come within `threshold` Hamming bits end up in the same
    cluster (transitively), so a cluster never straddles two splits.
    """
    all_hashes, owner = [], []
    for idx, item in enumerate(items):
        for h in item["hashes"]:
            all_hashes.append(h)
            owner.append(idx)
    all_hashes = np.asarray(all_hashes, dtype=np.uint64)
    log(f"  clustering {len(items)} images ({len(all_hashes)} hashes) ...")

    uf = UnionFind(len(items))
    for i, j in hamming_pairs(all_hashes, all_hashes, threshold):
        if owner[i] != owner[j]:
            uf.union(owner[i], owner[j])

    groups = defaultdict(list)
    for idx in range(len(items)):
        groups[uf.find(idx)].append(idx)
    return list(groups.values())


def split_clusters(items, clusters, fractions, seed):
    """Assign whole clusters to train/val/test so per-class fractions are met
    as closely as possible. Returns {split_name: [item_idx, ...]}.
    """
    split_names = list(fractions)
    rng = random.Random(seed)
    clusters = sorted(clusters, key=lambda c: (-len(c), min(c)))
    order = list(range(len(clusters)))
    rng.shuffle(order)
    # big clusters first so their bulk can still be balanced by the small ones
    order.sort(key=lambda ci: -len(clusters[ci]))

    class_totals = defaultdict(int)
    for it in items:
        class_totals[it["cls"]] += 1
    placed = {s: defaultdict(int) for s in split_names}
    assignment = {s: [] for s in split_names}

    for ci in order:
        cluster = clusters[ci]
        counts = defaultdict(int)
        for idx in cluster:
            counts[items[idx]["cls"]] += 1
        major = max(counts, key=counts.get)
        # pick the split with the largest relative deficit for the majority class
        best = max(
            split_names,
            key=lambda s: fractions[s] - placed[s][major] / max(class_totals[major], 1),
        )
        assignment[best].extend(cluster)
        for cls, n in counts.items():
            placed[best][cls] += n
    return assignment


def gather_images(source_map, log=print):
    """source_map: {class_name: [source_dir, ...]}.

    Loads every image, drops unreadable files and exact MD5 duplicates,
    computes dihedral pHashes. Returns list of item dicts.
    """
    items, seen_md5 = [], set()
    n_corrupt = n_exact_dup = 0
    for cls, dirs in source_map.items():
        for d in dirs:
            d = Path(d)
            for p in sorted(d.rglob("*")):
                if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
                    continue
                digest = md5_of(p)
                if digest in seen_md5:
                    n_exact_dup += 1
                    continue
                img = imread_unicode(p)
                if img is None or img.size == 0:
                    n_corrupt += 1
                    continue
                seen_md5.add(digest)
                items.append({"path": p, "cls": cls, "hashes": dihedral_phashes(img)})
        log(f"  {cls}: {sum(1 for it in items if it['cls'] == cls)} usable images so far")
    log(f"  dropped: {n_exact_dup} exact duplicates (MD5), {n_corrupt} unreadable")
    return items


def export_split(items, assignment, out_root, img_size, classes, log=print):
    """Resize and save assigned items as jpg; returns per-split per-class counts."""
    out_root = Path(out_root)
    counts = {}
    for split, idxs in assignment.items():
        counts[split] = {c: 0 for c in classes}
        for cls in classes:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)
        for idx in sorted(idxs):
            it = items[idx]
            img = imread_unicode(it["path"])
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            n = counts[split][it["cls"]]
            imwrite_jpg(out_root / split / it["cls"] / f"{it['cls']}_{n:05d}.jpg", img)
            counts[split][it["cls"]] += 1
        log(f"  {split}: " + ", ".join(f"{c}={counts[split][c]}" for c in classes))
    return counts
