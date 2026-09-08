"""Simulate longitudinal visit pairs with known ground truth.

Input: TEST-split images of the processed classification dataset (never train
images), per configs/config.yaml longitudinal.source_split.

For each pair:
  visit1.jpg  = the source image
  visit2.jpg  = the same image after a KNOWN random transform:
                perspective warp (small random homography built from rotation
                +/-15 deg, scale 0.85-1.15, crop shift, corner jitter),
                brightness/contrast/color-temperature jitter, slight blur.
                For a random ~50% of pairs a lesion change is simulated first:
                a circular region is slightly enlarged and darkened.
  ground_truth.json = exact 3x3 homography (visit1 -> visit2 pixel coords),
                lesion-change parameters (or null) and photometric parameters.

Output: data/longitudinal/pair_XXXX/{visit1.jpg, visit2.jpg, ground_truth.json}
plus data/longitudinal/README.md. All parameters come from configs/config.yaml.

Usage: python -m src.utils.longitudinal_sim
"""

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config
from src.utils.prep_common import imread_unicode, imwrite_jpg

README = """# Simulated longitudinal visit pairs

Each `pair_XXXX/` folder simulates the same mouth photographed at two visits:

- `visit1.jpg` — the original photo (taken from the classification TEST split
  only, so no training image ever leaks in here).
- `visit2.jpg` — the same photo after a known random capture change
  (perspective, rotation, scale, shift, lighting, blur), optionally with a
  simulated lesion change applied before warping.
- `ground_truth.json`:
  - `source_image`, `source_class` — where visit1 came from.
  - `homography` — exact 3x3 matrix H mapping visit1 pixel coords (x, y, 1)
    to visit2 coords. Invertible; `np.linalg.inv(H)` maps visit2 -> visit1.
  - `lesion_change` — `null` for unchanged pairs; otherwise
    `{center_xy, radius_px, darken_factor, enlarge_factor}` given in VISIT1
    pixel coordinates (the change is applied before the homography warp).
  - `photometric` — brightness/contrast/color-temperature/blur parameters
    (applied after the warp; they do not move pixels).

This data is synthetic and exists to evaluate alignment and change-detection
code against exact ground truth. It is part of a screening aid, not a
diagnostic tool — any real finding must be checked by a dentist.
"""


def random_homography(rng, w, h, cfg):
    """Compose rotation/scale/shift and mild corner jitter into one exact H."""
    angle = rng.uniform(-cfg["max_rotation_deg"], cfg["max_rotation_deg"])
    scale = rng.uniform(*cfg["scale_range"])
    tx = rng.uniform(-1, 1) * cfg["max_shift_frac"] * w
    ty = rng.uniform(-1, 1) * cfg["max_shift_frac"] * h

    a = math.radians(angle)
    cx, cy = w / 2.0, h / 2.0
    # rotate+scale about center, then translate
    affine = np.array(
        [
            [scale * math.cos(a), -scale * math.sin(a),
             cx - scale * (cx * math.cos(a) - cy * math.sin(a)) + tx],
            [scale * math.sin(a), scale * math.cos(a),
             cy - scale * (cx * math.sin(a) + cy * math.cos(a)) + ty],
            [0.0, 0.0, 1.0],
        ]
    )

    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    ones = np.ones((4, 1))
    warped = (affine @ np.hstack([corners, ones]).T).T
    warped = warped[:, :2] / warped[:, 2:3]
    jitter = cfg["perspective_jitter_frac"]
    warped += np.array(
        [[rng.uniform(-1, 1) * jitter * w, rng.uniform(-1, 1) * jitter * h] for _ in range(4)]
    )
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), warped.astype(np.float32))
    return H.astype(np.float64)


def apply_lesion_change(img, rng, cfg):
    """Enlarge and darken a circular region slightly. Returns (img, params)."""
    h, w = img.shape[:2]
    r_frac = rng.uniform(*cfg["lesion_radius_frac"])
    radius = max(6, int(r_frac * min(h, w)))
    cx = rng.randint(radius, w - 1 - radius)
    cy = rng.randint(radius, h - 1 - radius)
    darken = rng.uniform(*cfg["lesion_darken_range"])
    enlarge = rng.uniform(*cfg["lesion_enlarge_range"])

    # local radial expansion: pixels inside the region sample closer to the
    # center, which magnifies (enlarges) the central pattern
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    falloff = np.clip(1.0 - dist / radius, 0.0, 1.0)
    shrink = 1.0 / enlarge
    factor = 1.0 - (1.0 - shrink) * falloff  # 1 outside, `shrink` at center
    map_x = cx + dx * factor
    map_y = cy + dy * factor
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # smooth darkening inside the region
    mask = falloff[..., None]
    out = out.astype(np.float32) * (1.0 - mask * (1.0 - darken))
    out = np.clip(out, 0, 255).astype(np.uint8)

    params = {
        "center_xy": [int(cx), int(cy)],
        "radius_px": int(radius),
        "darken_factor": round(darken, 4),
        "enlarge_factor": round(enlarge, 4),
    }
    return out, params


def apply_photometric(img, rng, cfg):
    brightness = rng.uniform(-1, 1) * cfg["brightness_jitter"]
    contrast = 1.0 + rng.uniform(-1, 1) * cfg["contrast_jitter"]
    temp = rng.uniform(-1, 1) * cfg["color_temp_jitter"]
    blur_sigma = rng.uniform(0.0, cfg["max_blur_sigma"])

    out = img.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * contrast + mean + brightness * 255.0
    out[..., 2] *= 1.0 + temp  # warmer/cooler: push R and B apart (BGR order)
    out[..., 0] *= 1.0 - temp
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur_sigma > 0.15:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)

    return out, {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "color_temp": round(temp, 4),
        "blur_sigma": round(blur_sigma, 4),
    }


def main():
    cfg = load_config()
    lcfg = cfg["longitudinal"]
    src_root = (
        PROJECT_ROOT / cfg["paths"]["processed"] / "classification" / lcfg["source_split"]
    )
    out_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    out_root.mkdir(parents=True, exist_ok=True)

    sources = sorted(src_root.rglob("*.jpg"))
    if not sources:
        raise SystemExit(f"no source images found in {src_root} - run prepare_classification first")
    print(f"{len(sources)} candidate images in '{lcfg['source_split']}' split")

    n_pairs = lcfg["n_pairs"]
    rng = random.Random(cfg["split"]["seed"])
    picks = (
        rng.sample(sources, n_pairs)
        if len(sources) >= n_pairs
        else [sources[i % len(sources)] for i in range(n_pairs)]
    )

    n_lesion = 0
    for i, src in enumerate(picks):
        pair_dir = out_root / f"pair_{i:04d}"
        pair_dir.mkdir(exist_ok=True)
        img = imread_unicode(src)
        h, w = img.shape[:2]

        lesion_params = None
        visit2 = img.copy()
        if rng.random() < lcfg["lesion_change_prob"]:
            visit2, lesion_params = apply_lesion_change(visit2, rng, lcfg)
            n_lesion += 1

        H = random_homography(rng, w, h, lcfg)
        visit2 = cv2.warpPerspective(
            visit2, H.astype(np.float32), (w, h), borderMode=cv2.BORDER_REFLECT
        )
        visit2, photo_params = apply_photometric(visit2, rng, lcfg)

        imwrite_jpg(pair_dir / "visit1.jpg", img)
        imwrite_jpg(pair_dir / "visit2.jpg", visit2)
        gt = {
            "source_image": str(src.relative_to(PROJECT_ROOT)),
            "source_class": src.parent.name,
            "homography": [[float(v) for v in row] for row in H],
            "lesion_change": lesion_params,
            "photometric": photo_params,
        }
        (pair_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")

    (out_root / "README.md").write_text(README, encoding="utf-8")
    print(f"generated {n_pairs} pairs in {out_root} ({n_lesion} with lesion change)")


if __name__ == "__main__":
    main()
