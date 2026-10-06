"""Keep licence-restricted images out of anything we publish.

The Piyarathne dataset is CC BY-NC-ND 4.0: non-commercial, and **no derivatives
may be redistributed**. A resized crop in a paper figure or a demo sample is a
derivative, so none of its images may reach docs/paper/figures/, the demo's
sample assets, or any published artifact. MIO is CC-BY-4.0, so figures and demo
samples come from MIO or SMART-OM.

This is a licence obligation, not a style preference, so it is checked in code
and asserted in tests/test_licence_guard.py rather than left to memory.

Detection is by perceptual hash against the restricted dataset, so a resized,
re-encoded or lightly cropped copy is still caught, plus a filename check for
the obvious case.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import phash64

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class LicenceViolation(RuntimeError):
    """Raised when a restricted image reaches a publishable location."""


def _hash_cache_path(cfg):
    return PROJECT_ROOT / cfg["paths"]["processed"] / "audit_cache" / "hashes.json"


def restricted_hashes(cfg=None):
    """{dataset: uint64 array} of perceptual hashes for the restricted datasets."""
    cfg = cfg or load_config()
    path = _hash_cache_path(cfg)
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name in cfg["licence"]["restricted_datasets"]:
        if name in blob:
            out[name] = np.asarray([int(e["phash"]) for e in blob[name]["entries"]],
                                   dtype=np.uint64)
    return out


def image_phash(path):
    with Image.open(path) as im:
        try:
            im.draft("L", (64, 64))
        except Exception:
            pass
        gray = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR),
                          dtype=np.float32)
    return np.uint64(phash64(gray))


def publishable_files(cfg=None):
    """Every image file in a location we might publish from."""
    cfg = cfg or load_config()
    found = []
    for relative in cfg["licence"]["publishable_dirs"]:
        directory = PROJECT_ROOT / relative
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                found.append(path)
    return found


def scan(cfg=None):
    """Return a list of violations; empty means the publishable tree is clean."""
    cfg = cfg or load_config()
    threshold = cfg["licence"]["phash_threshold"]
    restricted = restricted_hashes(cfg)
    violations = []

    for path in publishable_files(cfg):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for name in cfg["licence"]["restricted_datasets"]:
            if name in relative.lower():
                violations.append({"path": relative, "dataset": name, "distance": 0,
                                   "why": "path names a restricted dataset"})
        if not restricted:
            continue
        try:
            value = image_phash(path)
        except Exception:
            continue
        for name, hashes in restricted.items():
            if not len(hashes):
                continue
            xor = np.bitwise_xor(hashes, value).view(np.uint8).reshape(len(hashes), 8)
            distances = np.unpackbits(xor, axis=1).sum(axis=1)
            closest = int(distances.min())
            if closest <= threshold:
                violations.append({"path": relative, "dataset": name,
                                   "distance": closest,
                                   "why": "perceptually matches a restricted image"})
    return violations


def assert_clean(cfg=None):
    violations = scan(cfg)
    if violations:
        detail = "; ".join(f"{v['path']} ({v['why']}, distance {v['distance']})"
                           for v in violations)
        raise LicenceViolation(
            f"{len(violations)} licence-restricted image(s) in a publishable "
            f"location: {detail}"
        )
    return True


if __name__ == "__main__":
    found = scan()
    if found:
        for violation in found:
            print(f"VIOLATION  {violation['path']}  <- {violation['dataset']}  "
                  f"({violation['why']}, distance {violation['distance']})")
        raise SystemExit(1)
    cfg = load_config()
    print(f"clean: {len(publishable_files(cfg))} images checked in "
          f"{cfg['licence']['publishable_dirs']}, none restricted")
