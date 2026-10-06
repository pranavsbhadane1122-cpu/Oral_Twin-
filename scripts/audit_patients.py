"""Patient structure of the new datasets. READ-ONLY on data/raw.

Piyarathne ships patient-level metadata, so patient-disjoint splits are
possible: this reports how many patients there are, how many images each has,
and whether every image file maps to a patient.

MIO is reported two ways: whatever its metadata workbook contains, and - because
repeat photographs of one mouth must never span splits - a perceptual-hash
clustering of its images. A tight cluster means the same mouth photographed more
than once, or the same photograph re-rendered with annotations drawn on it.

Usage: python scripts/audit_patients.py
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.prep_common import UnionFind, hamming_pairs  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_files(root, pattern):
    return [p for p in root.rglob(pattern) if p.is_file()]


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def describe_counts(counts, log, label):
    values = sorted(counts)
    if not values:
        log(f"  {label}: none")
        return
    arr = np.asarray(values)
    log(f"  {label}: min {arr.min()}  median {int(np.median(arr))}  "
        f"mean {arr.mean():.1f}  max {arr.max()}")
    histogram = Counter(values)
    common = ", ".join(f"{k} img x{v}" for k, v in sorted(histogram.items())[:10])
    log(f"    distribution (smallest first): {common}")


def audit_piyarathne(raw, log):
    root = raw / "piyarathne_oral"
    log(f"\n{'=' * 78}\nPIYARATHNE - patient structure\n{'=' * 78}")
    if not root.exists():
        log("  MISSING")
        return

    patientwise = find_files(root, "Patientwise*.csv")
    imagewise = find_files(root, "Imagewise*.csv")
    log(f"  Patientwise CSV: {[str(p.relative_to(root)) for p in patientwise]}")
    log(f"  Imagewise  CSV: {[str(p.relative_to(root)) for p in imagewise]}")
    if not imagewise:
        log("  no image-level metadata found, cannot map images to patients")
        return

    rows = read_csv_rows(imagewise[0])
    log(f"  Imagewise rows: {len(rows)}")
    if not rows:
        return
    columns = list(rows[0])
    log(f"  columns: {columns}")

    def pick(*candidates):
        for want in candidates:
            for col in columns:
                if want in col.lower().replace(" ", "").replace("_", ""):
                    return col
        return None

    import re

    patient_col = pick("patientid", "patient", "subject")
    image_col = pick("imagename", "imageid", "filename", "image")
    log(f"  patient column {patient_col!r}, image column {image_col!r}")
    if not image_col:
        log("  could not identify the image column, stopping here")
        return

    prefix = re.compile(r"^([A-Za-z]+-\d+)-\d+")
    per_patient = defaultdict(set)
    if patient_col:
        for row in rows:
            per_patient[str(row[patient_col]).strip()].add(str(row[image_col]).strip())
    else:
        # no patient column, but the image names encode it: R-01-03 is the third
        # image of patient R-01
        matched = 0
        for row in rows:
            name = str(row[image_col]).strip()
            found = prefix.match(name)
            if found:
                per_patient[found.group(1)].add(name)
                matched += 1
        log(f"  no patient column; patient id taken from the image-name prefix "
            f"<PATIENT>-<INDEX>, {matched}/{len(rows)} names matched")

    log(f"\n  unique patient IDs: {len(per_patient)}")
    describe_counts([len(v) for v in per_patient.values()], log, "images per patient")

    category_col = pick("category")
    diagnosis_col = pick("clinicaldiagnosis", "diagnosis")
    if category_col:
        log(f"\n  images per category: {dict(Counter(r[category_col] for r in rows))}")
    if diagnosis_col:
        log(f"  top clinical diagnoses: "
            f"{Counter(r[diagnosis_col] for r in rows).most_common(10)}")
    if category_col:
        spread = defaultdict(set)
        for row in rows:
            found = prefix.match(str(row[image_col]).strip())
            if found:
                spread[found.group(1)].add(row[category_col])
        multi = sum(1 for v in spread.values() if len(v) > 1)
        log(f"  patients whose images span more than one category: {multi} of "
            f"{len(spread)} - so a patient-level split cannot also balance "
            f"categories exactly")

    if patientwise:
        prows = read_csv_rows(patientwise[0])
        log(f"\n  Patientwise rows: {len(prows)}  columns: {list(prows[0]) if prows else []}")
        if prows:
            id_col = list(prows[0])[0]
            declared = {str(r[id_col]).strip() for r in prows}
            log(f"  declared patient ids: {len(declared)}")
            log(f"  ids also present in the image names: {len(declared & set(per_patient))}"
                f"  (declared but unseen: {len(declared - set(per_patient))}, "
                f"seen but undeclared: {len(set(per_patient) - declared)})")

    files = {p.name for p in root.rglob("*") if p.is_file()
             and p.suffix.lower() in IMAGE_EXTS}
    listed = {img for images in per_patient.values() for img in images}
    stems = {Path(f).stem for f in files}
    listed_stems = {Path(i).stem for i in listed}
    log(f"\n  image files on disk: {len(files)}")
    log(f"  images named in metadata: {len(listed)}")
    log(f"  on disk but not in metadata: {len(stems - listed_stems)}")
    log(f"  in metadata but not on disk: {len(listed_stems - stems)}")
    missing = sorted(listed_stems - stems)[:5]
    if missing:
        log(f"    e.g. {missing}")
    extra = sorted(stems - listed_stems)[:5]
    if extra:
        log(f"    e.g. unmapped files {extra}")
    coverage = 100.0 * len(stems & listed_stems) / max(len(stems), 1)
    log(f"  => {coverage:.1f}% of image files map to a patient")


def audit_mio_patient_ids(raw, cfg, log):
    """MIO filenames carry SMITA patient identifiers for most of the dataset."""
    import re
    root = raw / "mio"
    log("")
    log("=" * 78)
    log("MIO - patient identifiers in filenames")
    log("=" * 78)
    cache_path = (PROJECT_ROOT / cfg["paths"]["processed"] / "audit_cache" / "hashes.json")
    if not cache_path.exists() or "mio" not in json.loads(
            cache_path.read_text(encoding="utf-8")):
        log("  hash cache missing - run scripts/audit_hashes.py first")
        return
    entries = json.loads(cache_path.read_text(encoding="utf-8"))["mio"]["entries"]
    paths = [e["path"] for e in entries]

    pattern = re.compile(r"(SMITA\d+)", re.I)
    ids = [(pattern.search(p).group(1).upper() if pattern.search(p) else None)
           for p in paths]
    have = [i for i in ids if i]
    log(f"  images with a patient id in the filename: {len(have)} / {len(paths)} "
        f"({100.0 * len(have) / max(len(paths), 1):.1f}%)")
    log(f"  unique patient ids: {len(set(have))}")
    describe_counts(list(Counter(have).values()), log, "images per patient id")
    missing = Counter(p.split("/")[0] for p, i in zip(paths, ids) if not i)
    log("  images WITHOUT a patient id, by top-level folder:")
    for folder, n in missing.most_common():
        log(f"    {n:>5}  {folder}")
    log("  => patient-disjoint splitting IS possible for the SMART-OM part of MIO.")


def audit_mio_metadata(raw, log):
    root = raw / "mio"
    log(f"\n{'=' * 78}\nMIO - metadata\n{'=' * 78}")
    if not root.exists():
        log("  MISSING")
        return
    books = find_files(root, "*.xlsx")
    log(f"  workbooks found: {len(books)}")
    import openpyxl
    for path in books:
        log(f"\n  {path.relative_to(root)}")
        try:
            book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        except Exception as exc:
            log(f"    PARSE FAILED: {exc!r}")
            continue
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            header = next(rows, ())
            header = [str(h).strip() for h in header if h is not None]
            log(f"    sheet {sheet.title!r}: {sheet.max_row} rows x {sheet.max_column} cols")
            log(f"      columns: {header[:14]}")
            sample = next(rows, None)
            if sample:
                log(f"      first row: {[str(v)[:24] for v in sample[:14]]}")
        book.close()


def audit_mio_clusters(raw, cfg, log):
    root = raw / "mio"
    log(f"\n{'=' * 78}\nMIO - perceptual-hash clustering\n{'=' * 78}")
    cache_path = (PROJECT_ROOT / cfg["paths"]["processed"] / "audit_cache" / "hashes.json")
    if not cache_path.exists():
        log("  hash cache missing - run scripts/audit_hashes.py first")
        return
    blob = json.loads(cache_path.read_text(encoding="utf-8"))
    if "mio" not in blob:
        log("  mio not in the hash cache")
        return

    entries = blob["mio"]["entries"]
    paths = [e["path"] for e in entries]
    hashes = np.asarray([int(e["phash"]) for e in entries], dtype=np.uint64)
    threshold = cfg["audit"]["phash_cluster_threshold"]
    log(f"  images hashed: {len(paths)}   threshold: Hamming <= {threshold}")

    log("")
    log("  threshold sensitivity (a loose threshold chains unrelated"
        " photographs into one cluster):")
    for candidate in (2, 4, 6, 8, 10):
        probe = UnionFind(len(paths))
        for i, j in hamming_pairs(hashes, hashes, candidate):
            if i != j:
                probe.union(i, j)
        buckets = defaultdict(int)
        for index in range(len(paths)):
            buckets[probe.find(index)] += 1
        sizes_probe = sorted(buckets.values(), reverse=True)
        log(f"    threshold {candidate:>2}: {len(buckets):>5} clusters   "
            f"largest {sizes_probe[:5]}")

    uf = UnionFind(len(paths))
    for i, j in hamming_pairs(hashes, hashes, threshold):
        if i != j:
            uf.union(i, j)
    groups = defaultdict(list)
    for index in range(len(paths)):
        groups[uf.find(index)].append(index)
    clusters = list(groups.values())
    sizes = sorted((len(c) for c in clusters), reverse=True)

    log(f"\n  tight clusters: {len(clusters)}")
    log(f"  singleton clusters: {sum(1 for s in sizes if s == 1)}")
    log(f"  clusters with >1 image: {sum(1 for s in sizes if s > 1)}")
    log(f"  largest cluster sizes: {sizes[:12]}")
    log(f"  => {len(paths)} image files collapse to {len(clusters)} distinct "
        f"photographed scenes")

    biggest = max(clusters, key=len)
    log("\n  example of the largest cluster (first 8 members):")
    for index in biggest[:8]:
        log(f"    {paths[index]}")

    exact = defaultdict(list)
    for e in entries:
        exact[e["md5"]].append(e["path"])
    dupes = {h: v for h, v in exact.items() if len(v) > 1}
    log(f"\n  exact MD5 duplicate groups inside MIO: {len(dupes)} "
        f"({sum(len(v) - 1 for v in dupes.values())} redundant files)")


def main():
    cfg = load_config()
    raw = PROJECT_ROOT / cfg["paths"]["raw"]
    out = PROJECT_ROOT / cfg["paths"]["processed"] / "patient_structure.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("PATIENT STRUCTURE AUDIT (data/raw is read-only)")
    audit_piyarathne(raw, log)
    audit_mio_patient_ids(raw, cfg, log)
    audit_mio_metadata(raw, log)
    audit_mio_clusters(raw, cfg, log)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
