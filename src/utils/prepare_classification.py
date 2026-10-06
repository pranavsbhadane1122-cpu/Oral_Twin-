"""Build the de-biased 4/3-class classification dataset in
data/processed/classification/{train,val,test}/<class>/.

Source mapping (data/raw is READ-ONLY):
  caries     <- oral_diseases/Data caries (ORIGINAL set only; see below)
  calculus   <- oral_diseases/Calculus
  gingivitis <- oral_diseases/Gingivitis
  healthy    <- (no healthy folder exists in the source dataset - stays empty
                 until a healthy-teeth source is added to SOURCE_MAP)

Skipped on purpose: Tooth Discoloration, hypodontia, Mouth Ulcer (not our
classes) and the YOLO-annotated detection subset (mixed classes per image).

Phase 2b de-biasing
-------------------
Grad-CAM inspection showed the v1 classifier attending to lips, borders and
background rather than pathology. Three exclusions (all approved by the user)
remove the provenance signal it was exploiting:

  1. dedup.excluded_source_dirs - the "caries augmented data set" folder is
     dropped wholesale. It held ~85% of the caries class as augmented copies of
     a few base photos, and every illustration and smeared upscaled crop found
     during review lived there. Caries falls back to its 219 original photos.
  2. clipped images - blown-out augmentation artifacts (clip_frac above the
     detector's threshold), which carry a class-correlated exposure signature.
  3. images scoring at or above dedup.synthetic_score_thresh in
     data/processed/synthetic_flags.csv - drawn or texture-free images.

Then augmentation families are collapsed: images are clustered at
dedup.family_phash_threshold (looser than the leakage threshold, so a family is
always a superset of a leakage cluster) and at most dedup.max_per_family images
survive per family. Finally each class is capped so none exceeds
classification.max_class_ratio times the smallest, which stops one class
dominating.

Splits are assigned at FAMILY level, so no base photo can appear in two splits.

Usage: python -m src.utils.prepare_classification
"""

import csv
import json
import random
from collections import defaultdict

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import (
    cluster_near_duplicates,
    export_split,
    gather_images,
    split_clusters,
)


def load_exclusions(cfg, log=print):
    """Relative raw paths to skip, from data/processed/synthetic_flags.csv."""
    dcfg = cfg["dedup"]
    flags = PROJECT_ROOT / cfg["paths"]["processed"] / "synthetic_flags.csv"
    excluded, reasons = set(), defaultdict(int)
    if not flags.exists():
        log(f"  WARNING: {flags.name} not found - run src.utils.detect_synthetic first; "
            f"no per-image exclusions applied")
        return excluded, reasons

    with open(flags, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["path"].replace("\\", "/")
            clip = float(row["clip_frac"])
            score = float(row["score"])
            if dcfg["exclude_clipped"] and clip >= 0.60:
                excluded.add(path)
                reasons["clipped/blown-out"] += 1
            elif dcfg["exclude_synthetic"] and score >= dcfg["synthetic_score_thresh"]:
                excluded.add(path)
                reasons["drawn/texture-free"] += 1

    # images ruled out by hand, each with its evidence recorded in the CSV
    manual = PROJECT_ROOT / cfg["paths"]["processed"] / "exclusions.csv"
    if manual.exists():
        with open(manual, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel = f"{row['dataset']}/{row['path']}".replace("\\", "/")
                if rel not in excluded:
                    excluded.add(rel)
                    reasons[row["reason"][:48]] += 1
    else:
        log(f"  WARNING: {manual.name} not found; no manual exclusions applied")
    return excluded, reasons


def build_source_map(raw_root, cfg):
    od = raw_root / "oral_diseases"
    excluded_dirs = [str(d).replace("\\", "/") for d in cfg["dedup"]["excluded_source_dirs"]]

    def keep(path):
        rel = str(path.relative_to(raw_root)).replace("\\", "/")
        return not any(rel.startswith(d) for d in excluded_dirs)

    candidates = {
        "healthy": [],
        "caries": [od / "Data caries" / "Data caries" / "caries orignal data set",
                   od / "Data caries" / "Data caries" / "caries augmented data set"],
        "calculus": [od / "Calculus"],
        "gingivitis": [od / "Gingivitis"],
    }
    return {cls: [d for d in dirs if keep(d)] for cls, dirs in candidates.items()}


def cap_families(items, families, max_per_family, seed):
    """Keep at most max_per_family images per family. Returns (kept_indices,
    family_of_index) with families renumbered over the survivors."""
    rng = random.Random(seed)
    kept, family_of = [], {}
    for family_id, members in enumerate(families):
        members = sorted(members, key=lambda i: str(items[i]["path"]))
        if len(members) > max_per_family:
            members = sorted(rng.sample(members, max_per_family))
        for idx in members:
            family_of[idx] = family_id
            kept.append(idx)
    return sorted(kept), family_of


def rebalance(items, kept, family_of, classes, cap, seed, log=print):
    """Trim each class to `cap` images, removing whole families at a time so a
    base photo is never partially represented."""
    rng = random.Random(seed)
    by_class = defaultdict(lambda: defaultdict(list))
    for idx in kept:
        by_class[items[idx]["cls"]][family_of[idx]].append(idx)

    selected = []
    for cls in classes:
        family_ids = sorted(by_class[cls])
        rng.shuffle(family_ids)
        taken = []
        for family_id in family_ids:
            if len(taken) >= cap:
                break
            taken.extend(by_class[cls][family_id])
        if len(taken) > cap:  # last family overshot; drop its tail
            taken = taken[:cap]
        selected.extend(taken)
        log(f"    {cls}: {len(taken)} images from {len(family_ids)} families "
            f"(cap {cap})")
    return sorted(selected)


def main():
    cfg = load_config()
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    out_root = PROJECT_ROOT / cfg["paths"]["processed"] / "classification"
    ccfg = cfg["classification"]
    dcfg = cfg["dedup"]
    classes = ccfg["classes"]
    img_size = ccfg["img_size"]
    fractions = {k: cfg["split"][k] for k in ("train", "val", "test")}
    seed = cfg["split"]["seed"]

    source_map = build_source_map(raw_root, cfg)
    print("class mapping (de-biased):")
    for cls, dirs in source_map.items():
        shown = [str(d.relative_to(raw_root)) for d in dirs] or ["(none - empty class)"]
        print(f"  {cls} <- {shown}")
    print(f"  excluded source dirs: {dcfg['excluded_source_dirs']}")

    print("loading exclusion flags ...")
    excluded, reasons = load_exclusions(cfg)
    for reason, n in sorted(reasons.items()):
        print(f"  {reason}: {n} images flagged")

    print("gathering images (MD5 dedup + exclusions) ...")
    items = gather_images(source_map, img_size, exclude_rel=excluded, rel_to=raw_root)
    before = {cls: sum(1 for it in items if it["cls"] == cls) for cls in classes}

    print("collapsing augmentation families ...")
    families = cluster_near_duplicates(items, dcfg["family_phash_threshold"])
    print(f"  {len(items)} images -> {len(families)} families")
    kept, family_of = cap_families(items, families, dcfg["max_per_family"], seed)
    print(f"  after keeping at most {dcfg['max_per_family']} per family: {len(kept)} images")

    per_class = defaultdict(int)
    for idx in kept:
        per_class[items[idx]["cls"]] += 1
    non_empty = [n for n in per_class.values() if n]
    cap = min(ccfg["max_per_class"],
              int(round(ccfg["max_class_ratio"] * min(non_empty)))) if non_empty else 0
    print(f"  rebalancing to at most {cap} per class "
          f"({ccfg['max_class_ratio']}x the smallest class):")
    selected = rebalance(items, kept, family_of, classes, cap, seed)

    sub_items = [items[i] for i in selected]
    index_of = {old: new for new, old in enumerate(selected)}
    sub_families = defaultdict(list)
    for old in selected:
        sub_families[family_of[old]].append(index_of[old])
    clusters = list(sub_families.values())

    assignment = split_clusters(sub_items, clusters, fractions, seed)
    print("exporting splits ...")
    counts = export_split(sub_items, assignment, out_root, img_size, classes)

    after = {cls: sum(1 for it in sub_items if it["cls"] == cls) for cls in classes}
    print("\nbefore -> after de-biasing (per class):")
    for cls in classes:
        print(f"  {cls}: {before.get(cls, 0)} -> {after.get(cls, 0)}")
    print(f"families kept: {len(clusters)}")

    manifest = {
        "task": "classification",
        "version": "v2_debiased",
        "classes": classes,
        "img_size": img_size,
        "seed": seed,
        "phash_threshold": dcfg["phash_threshold"],
        "family_phash_threshold": dcfg["family_phash_threshold"],
        "max_per_family": dcfg["max_per_family"],
        "max_per_class_effective": cap,
        "excluded_source_dirs": dcfg["excluded_source_dirs"],
        "exclusion_counts": dict(reasons),
        "counts_before_debias": before,
        "counts_after_debias": after,
        "n_families": len(clusters),
        "mapping": {c: [str(d) for d in ds] for c, ds in source_map.items()},
        "counts": counts,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nfinal per-class counts per split:")
    for split, per_cls in counts.items():
        print(f"  {split}: " + ", ".join(f"{c}={n}" for c, n in per_cls.items()))
    print(f"manifest saved to {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
