"""Build the split manifests for the three sources that are never pooled.

Splits are written as CSV manifests, not as copied image trees: the raw data is
9 GB and data/raw is read-only, so a manifest naming each image and its split is
both cheaper and easier to audit than a second copy of the pixels.

  piyarathne       primary train/val/test, split by PATIENT (714 of them)
  smartom_external external validation only, never trains; Oral Cancer flagged
                   excluded from every performance figure
  mio_flat         Track A, split by pHash CLUSTER because it has no patient ids

Track B (Kaggle) keeps its existing image-copy pipeline in
src/utils/prepare_classification.py and is not rebuilt here.

Writes data/processed/splits/** and data/processed/SPLIT_DESIGN.md.
READ-ONLY on data/raw.

Usage: python scripts/build_splits.py
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import UnionFind, hamming_pairs, split_clusters  # noqa: E402
from src.utils.smartom import (  # noqa: E402
    CONFOUNDED_CATEGORY,
    CONFOUNDED_REASON,
    category_of,
    list_images,
    patient_id_of,
)

SPLITS = ("train", "val", "test")
PIYARATHNE_DIR = "piyarathne_oral"
MIO_FLAT_FOLDERS = ("GINGIVITIS", "PERIODONTITIS", "SANO")
PATIENT_PREFIX = re.compile(r"^([A-Za-z]+-\d+)-\d+")


def load_exclusions(cfg):
    """{dataset: {relative path}} of images ruled out by hand."""
    path = PROJECT_ROOT / cfg["paths"]["processed"] / "exclusions.csv"
    excluded = defaultdict(set)
    if not path.exists():
        return excluded
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            excluded[row["dataset"]].add(row["path"].replace("\\", "/"))
    return excluded


def write_manifest(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- piyarathne

def build_piyarathne(cfg, excluded, log):
    raw = PROJECT_ROOT / cfg["paths"]["raw"] / PIYARATHNE_DIR
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "splits" / "piyarathne"
    fractions = {k: cfg["split"][k] for k in SPLITS}
    seed = cfg["split"]["seed"]

    csv_path = next(raw.rglob("Imagewise*.csv"), None)
    if csv_path is None:
        log("  Imagewise CSV missing; skipped")
        return None
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))

    on_disk = {}
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            on_disk.setdefault(p.stem.lower(), p)

    dropped = 0
    items, patients = [], defaultdict(list)
    for row in meta:
        name = str(row["Image Name"]).strip()
        match = PATIENT_PREFIX.match(name)
        path = on_disk.get(name.lower())
        if path is None or match is None:
            dropped += 1
            continue
        rel = path.relative_to(raw).as_posix()
        if rel in excluded.get(PIYARATHNE_DIR, set()):
            dropped += 1
            continue
        patients[match.group(1)].append(len(items))
        items.append({
            "image_name": name,
            "path": f"{PIYARATHNE_DIR}/{rel}",
            "patient_id": match.group(1),
            "category": str(row["Category"]).strip(),
            "clinical_diagnosis": str(row.get("Clinical Diagnosis", "")).strip(),
            "cls": str(row["Category"]).strip(),
        })

    log(f"  images: {len(items)}   patients: {len(patients)}   "
        f"dropped (unresolved or excluded): {dropped}")

    clusters = list(patients.values())
    assignment = split_clusters(items, clusters, fractions, seed)

    rows_by_split, counts = {}, {}
    for split, indexes in assignment.items():
        rows = []
        for i in sorted(indexes):
            item = dict(items[i])
            item.pop("cls")
            item["split"] = split
            rows.append(item)
        rows_by_split[split] = rows
        counts[split] = Counter(r["category"] for r in rows)
        write_manifest(out_dir / f"{split}.csv", rows,
                       ["image_name", "path", "patient_id", "category",
                        "clinical_diagnosis", "split"])

    patients_per_split = {s: len({r["patient_id"] for r in rows})
                          for s, rows in rows_by_split.items()}
    log("  per split:")
    categories = sorted({i["category"] for i in items})
    for split in SPLITS:
        total = sum(counts[split].values())
        share = total / max(len(items), 1)
        log(f"    {split:<6} {total:>5} images ({share:>5.1%})  "
            f"{patients_per_split[split]:>3} patients  "
            + "  ".join(f"{c}={counts[split].get(c, 0)}" for c in categories))
    return {"items": len(items), "patients": len(patients), "dropped": dropped,
            "counts": {s: dict(counts[s]) for s in SPLITS},
            "patients_per_split": patients_per_split, "categories": categories,
            "out_dir": out_dir}


# ------------------------------------------------------------ smartom external

def build_smartom(cfg, log):
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "splits" / "smartom_external"
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    images = list_images(cfg)

    rows = []
    for path in images:
        category = category_of(path, cfg)
        confounded = category == CONFOUNDED_CATEGORY
        subject = patient_id_of(path) or Path(path).stem
        rows.append({
            "path": path.relative_to(raw_root).as_posix(),
            "category": category,
            "subject_id": subject,
            # coarse labels only: the positional descriptor matching agrees with
            # the region counts just 89-96% of the time, so fine-grained types
            # derived from it would be wrong for a meaningful share of images
            "lesion_present": "" if confounded else int(category != "01. Normal"),
            "excluded_from_metrics": int(confounded),
            "exclusion_reason": CONFOUNDED_REASON if confounded else "",
        })

    write_manifest(out_dir / "validation.csv", rows,
                   ["path", "category", "subject_id", "lesion_present",
                    "excluded_from_metrics", "exclusion_reason"])

    # Subject counting is only exact where a SMITA id exists. For the rest the
    # filename stem stands in, which is an approximation: two Oral Cancer files
    # share the stem "Ca 2" despite being different photographs, so that
    # category's subject count is a lower bound.
    per_category = defaultdict(
        lambda: {"images": 0, "patients": set(), "no_id": 0, "stems": set()})
    for row in rows:
        bucket = per_category[row["category"]]
        bucket["images"] += 1
        full = raw_root / row["path"]
        pid = patient_id_of(full)
        if pid:
            bucket["patients"].add(pid)
        else:
            bucket["no_id"] += 1
            bucket["stems"].add(Path(row["path"]).stem)
    log("  per category:")
    log(f"    {'category':<28} {'images':>7} {'patients':>9} {'no id':>7} {'stems':>7}")
    summary = {}
    for category in sorted(per_category):
        bucket = per_category[category]
        flag = "   EXCLUDED from metrics" if category == CONFOUNDED_CATEGORY else ""
        log(f"    {category:<28} {bucket['images']:>7} {len(bucket['patients']):>9} "
            f"{bucket['no_id']:>7} {len(bucket['stems']):>7}{flag}")
        summary[category] = {"images": bucket["images"],
                             "patients_with_smita_id": len(bucket["patients"]),
                             "files_without_id": bucket["no_id"],
                             "distinct_stems_without_id": len(bucket["stems"]),
                             "subjects": len(bucket["patients"]) + len(bucket["stems"])}
    usable = sum(v["images"] for k, v in summary.items() if k != CONFOUNDED_CATEGORY)
    log(f"  usable for metrics: {usable} images; "
        f"{summary.get(CONFOUNDED_CATEGORY, {}).get('images', 0)} excluded")
    return {"summary": summary, "usable": usable, "out_dir": out_dir,
            "total": len(rows)}


# ----------------------------------------------------------------- mio flat

def build_mio_flat(cfg, excluded, log):
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "splits" / "mio_flat"
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    cache = PROJECT_ROOT / cfg["paths"]["processed"] / "audit_cache" / "hashes.json"
    fractions = {k: cfg["split"][k] for k in SPLITS}
    seed = cfg["split"]["seed"]
    threshold = cfg["audit"]["phash_cluster_threshold"]

    if not cache.exists():
        log("  hash cache missing; run scripts/audit_hashes.py first")
        return None
    entries = json.loads(cache.read_text(encoding="utf-8"))["mio"]["entries"]

    dropped = 0
    items, hashes = [], []
    for entry in entries:
        rel = entry["path"].replace("\\", "/")
        folder = rel.split("/")[0]
        if folder not in MIO_FLAT_FOLDERS:
            continue
        if rel in excluded.get("mio", set()):
            dropped += 1
            continue
        items.append({"path": f"mio/{rel}", "label": folder.lower(), "cls": folder.lower()})
        hashes.append(int(entry["phash"]))

    log(f"  images: {len(items)}   excluded: {dropped}   "
        f"cluster threshold: Hamming <= {threshold}")

    values = np.asarray(hashes, dtype=np.uint64)
    uf = UnionFind(len(items))
    for i, j in hamming_pairs(values, values, threshold):
        if i != j:
            uf.union(i, j)
    groups = defaultdict(list)
    for index in range(len(items)):
        groups[uf.find(index)].append(index)
    clusters = list(groups.values())
    cluster_of = {index: cid for cid, members in enumerate(clusters) for index in members}
    log(f"  clusters: {len(clusters)} "
        f"(largest {sorted((len(c) for c in clusters), reverse=True)[:5]})")

    assignment = split_clusters(items, clusters, fractions, seed)
    counts, cluster_counts = {}, {}
    for split, indexes in assignment.items():
        rows = []
        for i in sorted(indexes):
            rows.append({"path": items[i]["path"], "label": items[i]["label"],
                         "cluster_id": cluster_of[i], "split": split})
        write_manifest(out_dir / f"{split}.csv", rows,
                       ["path", "label", "cluster_id", "split"])
        counts[split] = Counter(r["label"] for r in rows)
        cluster_counts[split] = len({r["cluster_id"] for r in rows})

    labels = sorted({i["label"] for i in items})
    log("  per split:")
    for split in SPLITS:
        total = sum(counts[split].values())
        log(f"    {split:<6} {total:>4} images ({total / max(len(items), 1):>5.1%})  "
            f"{cluster_counts[split]:>4} clusters  "
            + "  ".join(f"{l}={counts[split].get(l, 0)}" for l in labels))
    return {"items": len(items), "clusters": len(clusters), "dropped": dropped,
            "counts": {s: dict(counts[s]) for s in SPLITS},
            "cluster_counts": cluster_counts, "labels": labels, "out_dir": out_dir}


def main():
    cfg = load_config()
    excluded = load_exclusions(cfg)
    total_excluded = sum(len(v) for v in excluded.values())
    print(f"manual exclusions loaded: {total_excluded} "
          f"({ {k: len(v) for k, v in excluded.items()} })\n")

    print("PIYARATHNE - primary, split by patient")
    piyarathne = build_piyarathne(cfg, excluded, print)
    print("\nSMART-OM - external validation only")
    smartom = build_smartom(cfg, print)
    print("\nMIO FLAT FOLDERS - Track A, split by pHash cluster")
    mio = build_mio_flat(cfg, excluded, print)

    report = {"piyarathne": piyarathne, "smartom": smartom, "mio_flat": mio,
              "excluded": {k: sorted(v) for k, v in excluded.items()},
              "built_on": date.today().isoformat()}
    out = PROJECT_ROOT / cfg["paths"]["processed"] / "splits" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
