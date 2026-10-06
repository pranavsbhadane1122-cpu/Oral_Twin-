"""The only sanctioned way to read SMART-OM images.

Why this module exists
----------------------
SMART-OM stores every photograph three or four times: once clean under
"01. Unannotated", and again under "02. Region annotation", "03. Full
annotation" and "04. Lesion annotation" with the annotations BURNED INTO THE
PIXELS. A model trained on those levels would learn "a drawn outline means
disease" - a shortcut even more direct than the illustration contamination that
invalidated the Phase 2 classifier, because the outline is painted on top of
exactly the lesion the label refers to.

So the rule is enforced here in code rather than left to discipline: every
pipeline reads SMART-OM through list_images(), which returns files from
"01. Unannotated" only and re-checks each path before handing it back.

A subtlety worth knowing: the CATEGORIES are also numbered - "02. Variation from
normal", "03. OPMD", "04. Oral Cancer" - so a rule that blocked any path
containing "02." would silently throw away a whole disease category. The block
therefore targets annotation LEVELS by name, not by number.

READ-ONLY on data/raw.
"""

import re
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# "02. Region annotation", "03. Full annotation", "04. Lesion annotation" and the
# json subfolders beneath them; never "01. Unannotated"
ANNOTATED_LEVEL = re.compile(r"^\s*0[2-4]\.\s*(region|full|lesion)\s+annotation\s*$", re.I)


class AnnotatedLevelError(RuntimeError):
    """Raised when a burned-in annotation level would have been returned."""


def smartom_root(cfg=None):
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["smartom"]["root"]


def is_annotated_level_part(part):
    """True for a path component that is a burned-in annotation level."""
    name = str(part).strip().lower()
    if "unannotated" in name:
        return False
    if ANNOTATED_LEVEL.match(str(part)):
        return True
    # the json folders sit inside those levels and carry their own names
    return name.endswith("annotation") or name.endswith("json")


def is_blocked(path):
    """True if any component of the path is a burned-in annotation level."""
    return any(is_annotated_level_part(part) for part in Path(path).parts)


def list_images(cfg=None, category=None):
    """Every SMART-OM photograph, from the unannotated level only.

    category: optional exact category folder name, e.g. "03. OPMD".
    """
    cfg = cfg or load_config()
    scfg = cfg["smartom"]
    if not scfg.get("annotated_levels_blocked", True):
        raise AnnotatedLevelError(
            "smartom.annotated_levels_blocked is false. The annotated levels have "
            "the annotations burned into the pixels and must never be trained on; "
            "set it back to true."
        )

    root = smartom_root(cfg)
    categories = [category] if category else list(scfg["categories"])
    level = scfg["unannotated_level"]

    found = []
    for name in categories:
        directory = root / name / level
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                found.append(path)

    # belt and braces: re-check what is about to be returned
    for path in found:
        if is_blocked(path.relative_to(root)):
            raise AnnotatedLevelError(f"blocked annotation level leaked into the "
                                      f"loader output: {path}")
    return found


def category_of(path, cfg=None):
    """The SMART-OM category folder a path belongs to, or None."""
    cfg = cfg or load_config()
    parts = Path(path).parts
    for name in cfg["smartom"]["categories"]:
        if name in parts:
            return name
    return None


def patient_id_of(path):
    """The SMITA patient id embedded in the filename, or None."""
    match = re.search(r"(SMITA\d+)", Path(path).name, re.I)
    return match.group(1).upper() if match else None


def count_unannotated(cfg=None):
    """Per-category counts straight from disk, for cross-checking the loader."""
    cfg = cfg or load_config()
    root = smartom_root(cfg)
    level = cfg["smartom"]["unannotated_level"]
    counts = {}
    for name in cfg["smartom"]["categories"]:
        directory = root / name / level
        counts[name] = sum(
            1 for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ) if directory.is_dir() else 0
    return counts
