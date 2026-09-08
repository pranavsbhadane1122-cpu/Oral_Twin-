"""Create oraltwin_colab.zip for training on Google Colab.

Contents: src/, configs/, requirements.txt and data/processed/classification/
(nothing else - no raw data, no venv, no git history).

Usage: python scripts/make_colab_bundle.py
"""

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402


def main():
    cfg = load_config()
    out = PROJECT_ROOT / "oraltwin_colab.zip"
    include = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / cfg["paths"]["processed"] / "classification",
    ]
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in include:
            if item.is_file():
                zf.write(item, item.relative_to(PROJECT_ROOT))
                n += 1
                continue
            for p in sorted(item.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, p.relative_to(PROJECT_ROOT))
                    n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
