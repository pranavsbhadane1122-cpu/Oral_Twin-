"""Assemble the photograph description and the visit comparison into one report.

Every report ends with LIMITATIONS, verbatim. That sentence is the whole point
of this layer: a system that describes photographs is easily mistaken for one
that reads them, and the only defence is to say plainly, every time, that it
does not. tests/test_caption.py asserts its presence on every path, including
the failure paths, because a report that ends early on an error is exactly the
one a user would most likely misread.

CLI:
  python -m src.caption.render --pair data/longitudinal/pair_0001
"""

import argparse
import json
from pathlib import Path

from src.caption import compare as compare_caption
from src.caption import describe as describe_photo
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode

LIMITATIONS = (
    "This tool does not identify dental conditions. It describes whether your "
    "photograph is usable and what has changed since your last one. Any concern "
    "should be checked by a dentist."
)


def render(description, comparison=None):
    """The full user-facing report as one string. Always ends with LIMITATIONS."""
    blocks = ["YOUR PHOTOGRAPH", ""]
    blocks += describe_photo.sentences(description)
    if not description["usable"]:
        # said once, explicitly: an unusable photo must never read as a
        # processed one that simply happened to find nothing
        blocks.append("This photograph may not be good enough to rely on.")

    if comparison is not None:
        blocks += ["", "COMPARED WITH YOUR LAST VISIT", ""]
        blocks += compare_caption.sentences(comparison)

    blocks += ["", "WHAT THIS TOOL CANNOT TELL YOU", "", LIMITATIONS]
    return "\n".join(blocks)


def report_for_pair(pair_dir, cfg=None):
    """(text, payload) for a longitudinal pair, using its existing delta report."""
    cfg = cfg or load_config()
    pair_dir = Path(pair_dir)
    img2 = imread_unicode(pair_dir / "visit2.jpg")
    if img2 is None:
        raise FileNotFoundError(f"could not read {pair_dir / 'visit2.jpg'}")

    description = describe_photo.describe(img2, cfg)

    delta_path = (PROJECT_ROOT / cfg["paths"]["processed"] / "delta" / pair_dir.name
                  / "change_report.json")
    comparison = None
    if delta_path.exists():
        change_report = json.loads(delta_path.read_text(encoding="utf-8"))
        comparison = compare_caption.summarise(change_report, cfg)

    payload = {"pair": pair_dir.name, "description": description,
               "comparison": comparison}
    return render(description, comparison), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True,
                        help="path to a data/longitudinal/pair_XXXX folder")
    args = parser.parse_args()

    cfg = load_config()
    pair_dir = Path(args.pair)
    if not pair_dir.is_absolute():
        pair_dir = PROJECT_ROOT / pair_dir

    text, _ = report_for_pair(pair_dir, cfg)
    print(text)


if __name__ == "__main__":
    main()
