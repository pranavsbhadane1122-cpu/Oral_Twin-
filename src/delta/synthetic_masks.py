"""Build synthetic lesion masks for the longitudinal pairs.

These stand in for what the Phase 3 U-Net will eventually output, so the delta
module can be built and validated now. data/longitudinal is read-only input;
masks are written to data/processed/delta_masks/pair_XXXX/.

How the masks follow the ground truth
-------------------------------------
`longitudinal_sim.py` recorded, per pair, either a lesion change
(center_xy, radius_px, enlarge_factor, applied in VISIT1 coordinates before the
homography) or null.

  - lesion change present: visit1 gets a blob of radius r at the recorded
    centre; visit2 gets the SAME blob scaled by enlarge_factor. That mirrors
    the local magnification the simulator actually painted into the pixels.
  - no lesion change: a deterministic base blob (seeded by the pair name) is
    placed identically in both visits, so "no change" pairs still exercise the
    false-change path instead of trivially having nothing to compare.

Blobs are circles with a little radial harmonic noise, so compactness/edge
metrics see something less artificial than a perfect disc. The visit2 mask is
then warped into visit2's own coordinate frame with the ground-truth
homography, exactly as a real segmentation of the visit2 photo would be.

The injected area change stored in meta.json is measured from the rasterized
masks (not assumed from enlarge_factor**2), so it is exact ground truth.

Usage: python -m src.delta.synthetic_masks
"""

import json
import random

import cv2
import numpy as np

from src.utils.config import PROJECT_ROOT, load_config


def blob_mask(shape, center, radius, seed, cfg):
    """Filled near-circular blob with deterministic radial noise."""
    scfg = cfg["delta"]["synthetic"]
    rng = random.Random(seed)
    n_harm = scfg["edge_noise_harmonics"]
    amp = scfg["edge_noise_amplitude"]
    phases = [rng.uniform(0, 2 * np.pi) for _ in range(n_harm)]
    weights = [rng.uniform(-1.0, 1.0) / (k + 2) for k in range(n_harm)]

    thetas = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    wobble = np.ones_like(thetas)
    for k, (w, ph) in enumerate(zip(weights, phases), start=2):
        wobble += amp * w * np.sin(k * thetas + ph)
    wobble = np.clip(wobble, 0.6, 1.4)

    xs = center[0] + radius * wobble * np.cos(thetas)
    ys = center[1] + radius * wobble * np.sin(thetas)
    pts = np.stack([xs, ys], axis=1).round().astype(np.int32)

    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def masks_for_pair(pair_name, gt, shape, cfg):
    """Return (mask1, mask2_in_visit2_coords, meta)."""
    h, w = shape[:2]
    H_gt = np.asarray(gt["homography"], float)  # visit1 -> visit2
    lesion = gt["lesion_change"]
    seed = f"{pair_name}-lesion"

    if lesion is not None:
        center = tuple(lesion["center_xy"])
        radius = float(lesion["radius_px"])
        enlarge = float(lesion["enlarge_factor"])
        injected = True
    else:
        rng = random.Random(f"{pair_name}-base")
        frac = rng.uniform(*cfg["delta"]["synthetic"]["base_radius_frac"])
        radius = frac * min(h, w)
        margin = int(radius * 1.5) + 2
        center = (rng.randint(margin, w - margin), rng.randint(margin, h - margin))
        enlarge = 1.0
        injected = False

    mask1 = blob_mask(shape, center, radius, seed, cfg)
    mask2_v1 = blob_mask(shape, center, radius * enlarge, seed, cfg)
    mask2 = cv2.warpPerspective(mask2_v1, H_gt, (w, h), flags=cv2.INTER_NEAREST)

    a1 = int((mask1 > 127).sum())
    a2 = int((mask2_v1 > 127).sum())  # measured in visit1 coords: comparable to a1
    meta = {
        "pair": pair_name,
        "injected_lesion_change": injected,
        "center_xy": [int(center[0]), int(center[1])],
        "base_radius_px": round(radius, 2),
        "enlarge_factor": round(enlarge, 4),
        "area_visit1_px": a1,
        "area_visit2_px": a2,
        "injected_area_change_pct": round(100.0 * (a2 - a1) / a1, 3) if a1 else None,
        "darken_factor": lesion["darken_factor"] if lesion else None,
    }
    return mask1, mask2, meta


def main():
    cfg = load_config()
    long_root = PROJECT_ROOT / cfg["paths"]["longitudinal"]
    out_root = PROJECT_ROOT / cfg["paths"]["processed"] / "delta_masks"
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = sorted(p for p in long_root.glob("pair_*") if p.is_dir())
    n_injected = 0
    for pair in pairs:
        gt = json.loads((pair / "ground_truth.json").read_text(encoding="utf-8"))
        img1 = cv2.imread(str(pair / "visit1.jpg"))
        mask1, mask2, meta = masks_for_pair(pair.name, gt, img1.shape, cfg)

        out_dir = out_root / pair.name
        out_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(out_dir / "mask1.png"), mask1)
        cv2.imwrite(str(out_dir / "mask2.png"), mask2)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        n_injected += bool(meta["injected_lesion_change"])

    print(f"wrote masks for {len(pairs)} pairs to {out_root}")
    print(f"  with injected lesion growth: {n_injected}")
    print(f"  unchanged (base lesion in both visits): {len(pairs) - n_injected}")


if __name__ == "__main__":
    main()
