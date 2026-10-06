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


CONFOUNDED_CATEGORY = "04. Oral Cancer"

CONFOUNDED_REASON = (
    "SMART-OM's Oral Cancer class is excluded from every performance figure. All 20 "
    "of its images use a different naming scheme from the rest of the dataset, 8 of "
    "them are PNG where no other SMART-OM image is, only 2 carry EXIF where 97% of "
    "the rest do, and their median resolution is five times lower. Naming, format, "
    "EXIF presence and resolution are therefore all confounded with the label, so a "
    "metric computed here partly measures acquisition rather than disease. "
    "Piyarathne's OCA images are the cancer performance estimate. These 20 images "
    "are retained only as a documented probe for the appendix, always labelled "
    "\"20 images from a distinct acquisition stream, one subject each - not a "
    "sensitivity estimate\"."
)


class AnnotatedLevelError(RuntimeError):
    """Raised when a burned-in annotation level would have been returned."""


class ConfoundedSubsetError(RuntimeError):
    """Raised when a metric would have been computed over a confounded subset."""


def is_confounded_category(category):
    return str(category).strip() == CONFOUNDED_CATEGORY


def assert_scoreable(categories, allow_confounded_subset=False):
    """Guard for any evaluation that touches SMART-OM.

    Pass the categories a metric is about to be computed over. If the confounded
    Oral Cancer subset is among them the call raises, unless the caller has
    explicitly asked for it with allow_confounded_subset=True - which is only
    appropriate for the appendix probe, never for a reported figure.
    """
    if isinstance(categories, (str, Path)):
        categories = [categories]
    offending = [c for c in categories if is_confounded_category(c)]
    if offending and not allow_confounded_subset:
        raise ConfoundedSubsetError(
            f"refusing to score {offending}. {CONFOUNDED_REASON} "
            f"If this really is the appendix probe, pass allow_confounded_subset=True."
        )
    return True


def scoreable_categories(cfg=None):
    """The SMART-OM categories that may appear in a reported figure."""
    cfg = cfg or load_config()
    return [c for c in cfg["smartom"]["categories"] if not is_confounded_category(c)]


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
