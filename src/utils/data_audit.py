"""Audit the raw datasets in data/raw (READ-ONLY — never modifies anything there).

For every dataset (top-level folder under data/raw) reports:
  - number of images per class/folder
  - image size distribution
  - corrupt/unreadable files
  - exact duplicates (MD5 hash)

Writes the report to data/processed/audit_report.txt and prints it.

Usage: python -m src.utils.data_audit
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.utils.config import PROJECT_ROOT, load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(ds_root: Path):
    """Return an audit dict for one dataset folder."""
    per_folder = Counter()
    sizes = Counter()
    corrupt = []
    md5_map = defaultdict(list)

    for img_path in iter_images(ds_root):
        rel_folder = str(img_path.parent.relative_to(ds_root)) or "."
        per_folder[rel_folder] += 1
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                sizes[im.size] += 1
        except Exception as exc:
            corrupt.append((str(img_path.relative_to(ds_root)), repr(exc)))
            continue
        md5_map[md5_of(img_path)].append(str(img_path.relative_to(ds_root)))

    duplicates = {h: paths for h, paths in md5_map.items() if len(paths) > 1}
    return {
        "per_folder": per_folder,
        "sizes": sizes,
        "corrupt": corrupt,
        "duplicates": duplicates,
        "n_images": sum(per_folder.values()),
    }


def format_report(results: dict) -> str:
    lines = ["=" * 70, "RAW DATA AUDIT (data/raw is read-only; nothing was modified)", "=" * 70]
    for ds_name, r in results.items():
        lines += ["", f"DATASET: {ds_name}", "-" * 70, f"total images: {r['n_images']}", "", "images per folder:"]
        for folder, n in sorted(r["per_folder"].items()):
            lines.append(f"  {folder}: {n}")

        lines += ["", "image size distribution (top 15):"]
        for (w, h), n in r["sizes"].most_common(15):
            lines.append(f"  {w}x{h}: {n}")
        if len(r["sizes"]) > 15:
            lines.append(f"  ... and {len(r['sizes']) - 15} more unique sizes")

        lines += ["", f"corrupt/unreadable files: {len(r['corrupt'])}"]
        for rel, err in r["corrupt"][:20]:
            lines.append(f"  {rel}  ({err})")

        n_dup_files = sum(len(v) - 1 for v in r["duplicates"].values())
        lines += ["", f"exact-duplicate groups (MD5): {len(r['duplicates'])} "
                      f"({n_dup_files} redundant files)"]
        for h, paths in list(r["duplicates"].items())[:10]:
            lines.append(f"  {h[:10]}...: {len(paths)} copies, e.g. {paths[0]}")
        if len(r["duplicates"]) > 10:
            lines.append(f"  ... and {len(r['duplicates']) - 10} more groups")
    lines.append("")
    return "\n".join(lines)


def main():
    cfg = load_config()
    raw_root = PROJECT_ROOT / cfg["paths"]["raw"]
    processed_root = PROJECT_ROOT / cfg["paths"]["processed"]
    processed_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for ds_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        print(f"auditing {ds_dir.name} ...")
        results[ds_dir.name] = audit_dataset(ds_dir)

    report = format_report(results)
    out = processed_root / "audit_report.txt"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"report saved to {out}")


if __name__ == "__main__":
    main()
