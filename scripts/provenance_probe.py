"""Can a classifier tell which DATASET an image came from, ignoring disease?

If a small model separates the sources almost perfectly on appearance alone,
then any model trained on a mixture of them can reach high accuracy by learning
provenance instead of pathology - which is exactly how the Phase 2 classifier
failed. This probe measures that separability directly.

Disease labels are never read. The probe sees a heavily downsampled image, so
it is picking up gross appearance - colour cast, framing, exposure, resolution -
not fine clinical detail.

To stop near-duplicates inflating the result, one image per perceptual-hash
cluster is sampled within each source, and the train/test split is stratified.

READ-ONLY on data/raw.

Usage: python scripts/provenance_probe.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.utils.config import load_config  # noqa: E402

# the Kaggle sets are treated as one source: they are what the parked classifier
# was trained on
SOURCES = {
    "piyarathne": ["piyarathne_oral"],
    "mio": ["mio"],
    "kaggle": ["oral_diseases", "oral_cancer"],
}


def sample_paths(blob, datasets, n_wanted, seed):
    """One image per pHash cluster, up to n_wanted, from the given datasets."""
    rng = np.random.default_rng(seed)
    by_hash = defaultdict(list)
    for name in datasets:
        if name not in blob:
            continue
        root = Path(blob[name]["root"])
        for e in blob[name]["entries"]:
            by_hash[e["phash"]].append(root / e["path"])
    representatives = [paths[0] for paths in by_hash.values()]
    representatives.sort()
    if len(representatives) > n_wanted:
        index = rng.choice(len(representatives), size=n_wanted, replace=False)
        representatives = [representatives[i] for i in index]
    return representatives


def load_features(paths, size):
    rows, kept = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                try:
                    im.draft("RGB", (size * 2, size * 2))
                except Exception:
                    pass
                im = im.convert("RGB").resize((size, size), Image.BILINEAR)
                rows.append(np.asarray(im, dtype=np.float32).ravel() / 255.0)
            kept.append(p)
        except Exception:
            continue
    return np.asarray(rows), kept


def main():
    cfg = load_config()
    acfg = cfg["audit"]
    processed = PROJECT_ROOT / cfg["paths"]["processed"]
    cache_path = processed / "audit_cache" / "hashes.json"
    out = processed / "provenance_probe.txt"
    seed = cfg["split"]["seed"]
    size = acfg["probe_image_size"]
    lines = []

    def log(text=""):
        print(text, flush=True)
        lines.append(text)

    log("PROVENANCE PROBE - can a model tell the datasets apart by appearance alone?")
    if not cache_path.exists():
        log("hash cache missing - run scripts/audit_hashes.py first")
        out.write_text("\n".join(lines), encoding="utf-8")
        return
    blob = json.loads(cache_path.read_text(encoding="utf-8"))

    x_parts, y_parts, names = [], [], []
    for label, (source, datasets) in enumerate(SOURCES.items()):
        paths = sample_paths(blob, datasets, acfg["probe_images_per_source"], seed + label)
        features, kept = load_features(paths, size)
        if not len(features):
            log(f"  {source}: no images, skipped")
            continue
        x_parts.append(features)
        y_parts.append(np.full(len(features), len(names)))
        names.append(source)
        log(f"  {source:<12} {len(kept)} images sampled "
            f"(one per pHash cluster from {datasets})")

    if len(names) < 2:
        log("fewer than two sources available, nothing to probe")
        out.write_text("\n".join(lines), encoding="utf-8")
        return

    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    log(f"\n  feature vector: {size}x{size} RGB downsample = {x.shape[1]} values")
    log(f"  total images: {len(y)}   chance accuracy: {1.0 / len(names):.3f}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=acfg["probe_test_fraction"], random_state=seed, stratify=y)
    # two probes: the linear one is a lower bound on separability, the forest is
    # a fairer estimate of what a real network could exploit
    probes = {
        "logistic regression (linear)": LogisticRegression(max_iter=2000),
        "random forest (non-linear)": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1),
    }
    scores = {}
    accuracy, predicted = 0.0, None
    for probe_name, candidate in probes.items():
        candidate.fit(x_train, y_train)
        guess = candidate.predict(x_test)
        scores[probe_name] = float((guess == y_test).mean())
        if scores[probe_name] > accuracy:
            accuracy, predicted = scores[probe_name], guess

    log(f"\n  train: {len(y_train)}   test: {len(y_test)}")
    for probe_name, score in scores.items():
        log(f"    {probe_name:<30} accuracy {score:.4f}")
    log("")
    log(f"  SOURCE-PREDICTION ACCURACY (best probe): {accuracy:.4f}")
    cm = confusion_matrix(y_test, predicted)
    log(f"\n  confusion matrix (rows = true source, order {names}):")
    for row_name, row in zip(names, cm):
        log(f"    {row_name:<12} {row.tolist()}")
    for i, source in enumerate(names):
        support = int(cm[i].sum())
        recall = cm[i, i] / support if support else 0.0
        log(f"    {source:<12} recall {recall:.3f}  (n={support})")

    if accuracy >= 0.95:
        verdict = ("INTERPRETATION: the sources are trivially separable on appearance "
                   "alone. Any model trained on a mixture of them can score highly by "
                   "learning which dataset an image came from instead of what is wrong "
                   "with the mouth. Mixing these sources without controlling for "
                   "provenance would repeat the Phase 2 failure.")
    elif accuracy >= 0.80:
        verdict = ("INTERPRETATION: the sources are largely separable on appearance "
                   "alone, so provenance is a strong shortcut available to any model "
                   "trained on a mixture of them.")
    else:
        verdict = ("INTERPRETATION: the sources are not trivially separable at this "
                   "resolution, so provenance is a weaker shortcut than it was for the "
                   "Phase 2 corpus - but it is still present and must be controlled.")
    log(f"\n{verdict}")
    log("\nNote: disease labels were never used. One image per perceptual-hash cluster "
        "was sampled, so this is not inflated by repeat photographs.")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
