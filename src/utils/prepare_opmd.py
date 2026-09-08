"""Build the 2-class OPMD / oral-cancer-pattern dataset in
data/processed/opmd/{train,val,test}/{suspicious,normal}/.

Source (data/raw is READ-ONLY): shivam17299 "Oral Cancer (Lips and Tongue)"
  suspicious <- oral_cancer/OralCancer/cancer
  normal     <- oral_cancer/OralCancer/non-cancer

These are clinical mouth photos (verified by inspection); no microscope or
histopathology imagery is present, so nothing had to be excluded. If a future
dataset added here contains histopathology slides instead of mouth photos,
EXCLUDE it from the source map and flag it.

Same pipeline as prepare_classification: MD5 dedup -> dihedral-pHash
near-duplicate clustering -> cluster-level 70/15/15 split -> resize -> jpg.
Writes data/processed/opmd/manifest.json.

Usage: python -m src.utils.prepare_opmd
"""

import json

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import (
    cluster_near_duplicates,
    export_split,
    gather_images,
    split_clusters,
)


def build_source_map(raw_root):
    oc = raw_root / "oral_cancer" / "OralCancer"
    return {
        "suspicious": [oc / "cancer"],
        "normal": [oc / "non-cancer"],
    }


def main():
    cfg = load_config()
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    out_root = PROJECT_ROOT / cfg["paths"]["processed"] / "opmd"
    classes = cfg["opmd"]["classes"]
    img_size = cfg["opmd"]["img_size"]
    fractions = {k: cfg["split"][k] for k in ("train", "val", "test")}
    seed = cfg["split"]["seed"]
    threshold = cfg["dedup"]["phash_threshold"]

    source_map = build_source_map(raw_root)
    print("class mapping:")
    for cls, dirs in source_map.items():
        print(f"  {cls} <- {[str(d.relative_to(raw_root)) for d in dirs]}")

    print("gathering images (MD5 dedup) ...")
    items = gather_images(source_map)
    clusters = cluster_near_duplicates(items, threshold)
    print(f"  near-duplicate clusters with >1 image: "
          f"{sum(1 for c in clusters if len(c) > 1)}")

    assignment = split_clusters(items, clusters, fractions, seed)
    print("exporting splits ...")
    counts = export_split(items, assignment, out_root, img_size, classes)

    manifest = {
        "task": "opmd",
        "classes": classes,
        "img_size": img_size,
        "seed": seed,
        "phash_threshold": threshold,
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
