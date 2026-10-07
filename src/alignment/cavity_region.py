"""The trained cavity segmenter, used as a region-of-interest mask.

Phase 3's lesion channel failed its gate and is parked. The cavity channel
passed at Dice 0.933 [0.926, 0.940] on the Piyarathne test split, and this is
where it earns its keep: restricting ORB keypoints to the mouth, a job
previously done by an HSV colour heuristic.

Out of domain, and knowingly so
-------------------------------
The segmenter was trained on Piyarathne. The longitudinal pairs are built from
the MIO classification test split - different cameras, different clinic,
different framing. 0.933 is an IN-domain figure and nothing here entitles us to
assume it transfers; the external validation in Phase 8 showed how badly a
Piyarathne-trained model can degrade elsewhere. That is precisely why the swap
is measured against the heuristic rather than assumed to improve on it, and why
a degenerate prediction falls back rather than being trusted.

The model is loaded once and cached. If TensorFlow or the checkpoint is
unavailable the caller falls back to the heuristic, so importing this module
must never be fatal.
"""

import numpy as np

from src.utils.config import PROJECT_ROOT, load_config

_MODEL = None
_LOAD_FAILED = None          # the reason, once, so we do not retry 200 times

CAVITY_CHANNEL = 0


def _checkpoint(cfg):
    return PROJECT_ROOT / cfg["paths"]["models"] / "segmenter.h5"


def available(cfg=None):
    cfg = cfg or load_config()
    return _checkpoint(cfg).exists() and _LOAD_FAILED is None


def load_model(cfg=None):
    """Return the segmenter, or None if it cannot be loaded. Cached."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED is not None:
        return None
    cfg = cfg or load_config()
    path = _checkpoint(cfg)
    if not path.exists():
        _LOAD_FAILED = f"checkpoint not found: {path}"
        return None
    try:
        import tensorflow as tf

        _MODEL = tf.keras.models.load_model(path, compile=False)
    except Exception as exc:                    # noqa: BLE001 - must never be fatal
        _LOAD_FAILED = f"{type(exc).__name__}: {exc}"
        return None
    return _MODEL


def load_failure_reason():
    return _LOAD_FAILED


def predict_cavity(img_bgr, cfg=None):
    """Binary cavity mask (uint8 0/255) at the input image's own resolution.

    Returns None if the model is unavailable - the caller decides what to do.
    """
    import cv2

    cfg = cfg or load_config()
    model = load_model(cfg)
    if model is None:
        return None

    size = cfg["segmentation"]["img_size"]
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    batch = (small.astype(np.float32) / 127.5 - 1.0)[None]

    predicted = model.predict(batch, verbose=0)[0][..., CAVITY_CHANNEL]
    mask = (predicted > cfg["alignment"]["cavity_threshold"]).astype(np.uint8) * 255
    # nearest-neighbour back up: this is a label map, not an image
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def is_degenerate(mask, shape, cfg=None):
    """True if the prediction is too small to be a usable mouth region.

    A near-empty mask starves ORB of keypoints, which is worse than a crude
    mask that at least covers the mouth. Also catches the model confidently
    predicting nothing on an out-of-domain photo.
    """
    cfg = cfg or load_config()
    if mask is None:
        return True
    h, w = shape[:2]
    return mask.sum() / 255.0 < cfg["alignment"]["cavity_min_area_frac"] * h * w


def cavity_mask(img_bgr, cfg=None):
    """(mask, reason). mask is None when the cavity model cannot supply one.

    reason is None on success, otherwise a short string naming why the caller
    should fall back - recorded in the alignment output so the path taken is
    visible rather than silent.
    """
    import cv2

    cfg = cfg or load_config()
    mask = predict_cavity(img_bgr, cfg)
    if mask is None:
        return None, f"cavity model unavailable ({load_failure_reason()})"
    if is_degenerate(mask, img_bgr.shape, cfg):
        frac = mask.sum() / 255.0 / (img_bgr.shape[0] * img_bgr.shape[1])
        return None, f"degenerate cavity prediction ({frac:.3%} of frame)"

    dilate_px = cfg["alignment"]["cavity_dilate_px"]
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask, None
