"""Screenshot the four demo cases from a running Streamlit app.

The app must already be serving, e.g.:
  streamlit run app/oraltwin_app.py --server.headless true --server.port 8502

Then:
  python scripts/capture_demo_screenshots.py --url http://localhost:8502

Each case is deep-linked with ?case=<key>, so no clicking is needed and the
captures are reproducible. Images land in docs/paper/figures/demo_<key>.png.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

from src.demo.pipeline import demo_cases  # noqa: E402

READY_TEXT = "What this demo does not do"
CAPTION_HEADING = "0 - In plain language"


def wait_for_images(page, timeout):
    """Wait until every <img> has actually decoded.

    The limitations panel appearing means the Python run finished, but Streamlit
    streams the photographs in afterwards over its media endpoint. Screenshotting
    on the panel alone captured a page of grey placeholders - correct text, no
    images - so the pixels themselves have to be waited for.
    """
    page.wait_for_function(
        """() => {
            const imgs = Array.from(document.images);
            return imgs.length > 0 && imgs.every(i => i.complete && i.naturalWidth > 0);
        }""",
        timeout=timeout,
    )
    page.wait_for_timeout(1500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8502")
    parser.add_argument("--timeout", type=int, default=90000)
    # Streamlit scrolls inside its own container, so full_page alone captures only
    # the first screen: the viewport has to be tall enough to hold the whole run
    parser.add_argument("--height", type=int, default=3600)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "docs" / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(demo_cases())

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": args.height},
                                device_scale_factor=1.5)
        for key in keys:
            url = f"{args.url}/?case={key}"
            print(f"capturing {key} from {url}")
            page.goto(url, wait_until="load", timeout=args.timeout)
            # the panel is rendered last, so its presence means the run finished
            page.get_by_text(READY_TEXT).first.wait_for(timeout=args.timeout)
            wait_for_images(page, args.timeout)
            target = out_dir / f"demo_{key}.png"
            page.screenshot(path=str(target), full_page=True)
            print(f"  wrote {target.relative_to(PROJECT_ROOT)} "
                  f"({target.stat().st_size / 1000:.0f} kB)")

        # Two close-ups of the caption layer. The whole-page shots above are too
        # tall for the caption text to be readable in print, and the caption is
        # the part a reader most needs to be able to read word for word.
        for key, name in (("good", "caption"),
                          ("unusable", "caption_disagreement")):
            url = f"{args.url}/?case={key}"
            print(f"capturing {name} (caption close-up) from {url}")
            page.goto(url, wait_until="load", timeout=args.timeout)
            page.get_by_text(READY_TEXT).first.wait_for(timeout=args.timeout)
            wait_for_images(page, args.timeout)
            heading = page.get_by_text(CAPTION_HEADING).first
            heading.wait_for(timeout=args.timeout)
            # clip from the heading down to the start of section 1: an ancestor
            # locator grabbed only the heading itself, so the box is measured
            box = heading.bounding_box()
            nxt = page.get_by_text("1 - The two photographs").first
            end = nxt.bounding_box()
            height = (end["y"] - box["y"]) if end else 900
            target = out_dir / f"demo_{name}.png"
            page.screenshot(path=str(target), clip={
                "x": max(box["x"] - 20, 0), "y": max(box["y"] - 20, 0),
                "width": 1000, "height": max(height, 300)})
            print(f"  wrote {target.relative_to(PROJECT_ROOT)} "
                  f"({target.stat().st_size / 1000:.0f} kB)")
        browser.close()


if __name__ == "__main__":
    main()
