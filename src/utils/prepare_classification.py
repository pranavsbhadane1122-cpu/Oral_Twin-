"""Build the 4-class classification dataset in
data/processed/classification/{train,val,test}/{healthy,caries,calculus,gingivitis}/.

Source mapping (data/raw is READ-ONLY):
  caries     <- oral_diseases/Data caries (original + augmented)
  calculus   <- oral_diseases/Calculus
  gingivitis <- oral_diseases/Gingivitis
  healthy    <- (no healthy folder exists in the source dataset - stays empty
                 until a healthy-teeth source is added to SOURCE_MAP)

Skipped on purpose: Tooth Discoloration, hypodontia, Mouth Ulcer (not our
classes) and the YOLO-annotated detection subset (mixed classes per image).

Pipeline: drop MD5 exact duplicates -> cluster near-duplicates with dihedral
pHash (conservative threshold from config) -> split whole clusters 70/15/15
(seed from config) so no identical/near-identical image lands in two splits ->
resize to img_size -> save jpg. Prints per-class counts per split and writes
them to data/processed/classification/manifest.json.

Usage: python -m src.utils.prepare_classification
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
    od = raw_root / "oral_diseases"
    return {
        "healthy": [],  # no healthy source available in this dataset
        "caries": [
            od / "Data caries" / "Data caries" / "caries orignal data set",
            od / "Data caries" / "Data caries" / "caries augmented data set",
        ],
        "calculus": [od / "Calculus"],
        "gingivitis": [od / "Gingivitis"],
    }


def main():
    cfg = load_config()
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    out_root = PROJECT_ROOT / cfg["paths"]["processed"] / "classification"
    classes = cfg["classification"]["classes"]
    img_size = cfg["classification"]["img_size"]
    fractions = {k: cfg["split"][k] for k in ("train", "val", "test")}
    seed = cfg["split"]["seed"]
    threshold = cfg["dedup"]["phash_threshold"]

    source_map = build_source_map(raw_root)
    print("class mapping:")
    for cls, dirs in source_map.items():
        shown = [str(d.relative_to(raw_root)) for d in dirs] or ["(none - empty class)"]
        print(f"  {cls} <- {shown}")

    print("gathering images (MD5 dedup) ...")
    items = gather_images(source_map, img_size)
    clusters = cluster_near_duplicates(items, threshold + cfg["dedup"]["cluster_margin"])
    n_multi = sum(1 for c in clusters if len(c) > 1)
    print(f"  near-duplicate clusters with >1 image: {n_multi}")

    assignment = split_clusters(items, clusters, fractions, seed)
    print("exporting splits ...")
    counts = export_split(items, assignment, out_root, img_size, classes)

    manifest = {
        "task": "classification",
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
