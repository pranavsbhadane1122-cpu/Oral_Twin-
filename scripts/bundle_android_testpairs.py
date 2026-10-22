"""Copy 5 longitudinal test pairs into the Android assets, with their
ground-truth homographies and the Python reference figures.

WHY THIS IS A SCRIPT AND NOT A COMMIT
-------------------------------------
The images are derivatives of the Kaggle "Oral Diseases" dataset. The project's
own data/raw/LICENCES.md says of the Kaggle sets: "check each dataset page
before redistributing anything derived from them." Committing them would
publish them through a public repository, which is redistribution. That check
has not been done, so the assets are git-ignored and produced locally instead.

This is NOT the Piyarathne question. Piyarathne is CC BY-NC-ND and could never
be bundled. These pairs were traced and are not Piyarathne-derived: every one
of the 200 longitudinal pairs has a source_image under
data/processed/classification/test, whose classes (calculus, caries,
gingivitis) come from data/raw/oral_diseases, not data/raw/piyarathne_oral.
The script re-verifies that on every run and refuses rather than trusting it.

Pair selection: five spanning the Python corner-error distribution, chosen so
their median equals the 200-pair median (1.18 px). A median taken over five
pairs that happened to be easy would make the gate meaningless.

Usage: python scripts/bundle_android_testpairs.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402

PAIRS = ["pair_0111", "pair_0107", "pair_0085", "pair_0132", "pair_0118"]
FORBIDDEN = "piyarathne"
ASSET_DIR = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "testpairs"


def python_reference(cfg):
    """Per-pair figures from the canonical Phase 4 evaluation run."""
    results = (PROJECT_ROOT / cfg["paths"]["processed"] / "alignment" / "results.json")
    if not results.exists():
        raise SystemExit(
            f"{results} not found. Run python -m src.alignment.evaluate_alignment first."
        )
    return {r["pair"]: r for r in json.loads(results.read_text(encoding="utf-8"))}


def main():
    cfg = load_config()
    longitudinal = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    reference = python_reference(cfg)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    entries = []

    for name in PAIRS:
        pair_dir = longitudinal / name
        truth = json.loads((pair_dir / "ground_truth.json").read_text(encoding="utf-8"))

        # refuse rather than trust: re-check provenance on every run
        source = str(truth.get("source_image", "")).replace("\\", "/")
        if FORBIDDEN in source.lower():
            raise SystemExit(
                f"REFUSING: {name} traces to {source}, which is Piyarathne-derived "
                f"and is CC BY-NC-ND. Pick a different pair."
            )
        if "processed/classification" not in source:
            raise SystemExit(
                f"REFUSING: {name} has an unexpected source_image {source!r}. "
                f"Provenance could not be confirmed, so it is not bundled."
            )

        row = reference.get(name)
        if row is None or not row.get("aligned"):
            raise SystemExit(f"no Python reference figure for {name}")

        for visit in ("visit1", "visit2"):
            shutil.copy2(pair_dir / f"{visit}.jpg", ASSET_DIR / f"{name}_{visit}.jpg")

        entries.append({
            "pair": name,
            "source_class": truth.get("source_class", "?"),
            # as stored: maps visit1 -> visit2. The app inverts it, exactly as
            # evaluate_alignment.py does.
            "homography": truth["homography"],
            "python": {
                "corner_error_px": round(float(row["corner_error"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "n_inliers": int(row["n_inliers"]),
            },
        })
        print(f"  {name}: python corner error {row['corner_error']:.3f} px, "
              f"confidence {row['confidence']:.3f}, {row['n_inliers']} inliers")

    errors = sorted(e["python"]["corner_error_px"] for e in entries)
    median = errors[len(errors) // 2]

    manifest = {
        "note": ("Test pairs for the Kotlin alignment port. Derived from the Kaggle "
                 "Oral Diseases dataset via data/processed/classification. NOT "
                 "Piyarathne-derived: no CC BY-NC-ND material is in this APK."),
        "generated_by": "scripts/bundle_android_testpairs.py",
        "python_median_corner_error_px": median,
        "python_median_all_200_pairs_px": 1.183,
        "gate": "Kotlin median corner error within 2x of the Python median",
        "pairs": entries,
    }
    (ASSET_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(entries)} pairs -> {ASSET_DIR}")
    print(f"python median over these 5: {median:.3f} px "
          f"(all 200 pairs: 1.183 px)")
    print(f"gate: Kotlin median must be <= {2 * median:.3f} px")


if __name__ == "__main__":
    main()
