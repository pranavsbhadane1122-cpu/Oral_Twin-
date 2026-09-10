"""Grad-CAM for the OralTwin classifier.

Grad-CAM (Selvaraju et al., ICCV 2017) weights the final convolutional feature
maps by the gradient of a class score with respect to those maps, so the result
shows which image regions pushed the model toward that class.

The classifier is a functional model whose MobileNetV2 base is a nested
sub-model, so the gradient model is rebuilt explicitly here: the base is split
into (target feature map, base output) and the saved head layers are re-applied
on top. That also means this works with the locally converted TF 2.15 model in
models/classifier_3class.h5 (see scripts/convert_colab_model.py).

The target layer comes from configs/config.yaml explain.gradcam_layer; "auto"
resolves to the last spatial (4-D) layer of the base, which for MobileNetV2 is
out_relu (7x7x1280).

Flat-CAM guard
--------------
A heatmap whose standard deviation is below explain.flat_cam_std_eps carries no
localisation information - every region looks equally responsible. compute()
returns a `flat` flag in that case so callers never present a meaningless
picture as if it explained something.
"""

import cv2
import numpy as np
import tensorflow as tf

from src.utils.config import PROJECT_ROOT, load_config

BASE_LAYER = "mobilenetv2_1.00_224"
HEAD_LAYERS = ("gap", "dropout", "probs")


def load_classifier(cfg=None):
    cfg = cfg or load_config()
    n = len(cfg["classification"]["classes"])
    path = PROJECT_ROOT / cfg["paths"]["models"] / f"classifier_{n}class.h5"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - train or convert the model first")
    return tf.keras.models.load_model(path, compile=False)


def resolve_layer_name(model, cfg=None):
    """Return the Grad-CAM target layer name inside the MobileNetV2 base."""
    cfg = cfg or load_config()
    requested = cfg["explain"]["gradcam_layer"]
    base = model.get_layer(BASE_LAYER)
    if requested and str(requested).lower() != "auto":
        base.get_layer(requested)  # raises if the name is wrong
        return requested
    spatial = [l.name for l in base.layers if len(l.output_shape) == 4]
    if not spatial:
        raise ValueError("no spatial layer found in the base model")
    return spatial[-1]


def build_grad_model(model, layer_name=None, cfg=None):
    """Model mapping input -> (target feature map, class probabilities)."""
    cfg = cfg or load_config()
    layer_name = layer_name or resolve_layer_name(model, cfg)
    base = model.get_layer(BASE_LAYER)

    inner = tf.keras.Model(
        base.input, [base.get_layer(layer_name).output, base.output], name="inner_split"
    )
    inputs = tf.keras.Input(shape=model.input_shape[1:], name="image")
    features, base_out = inner(inputs)
    x = base_out
    for name in HEAD_LAYERS:
        x = model.get_layer(name)(x)
    return tf.keras.Model(inputs, [features, x], name="gradcam_model")


def preprocess(img_bgr, img_size):
    """BGR uint8 image -> batched MobileNetV2 input tensor."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(resized.astype(np.float32))
    return x[None, ...]


def compute(model, img_bgr, class_index=None, cfg=None, grad_model=None):
    """Grad-CAM heatmap for one image.

    Returns dict: heatmap (float32 [0,1] at input resolution), class_index,
    probability, flat (bool), heatmap_std.
    """
    cfg = cfg or load_config()
    img_size = cfg["classification"]["img_size"]
    grad_model = grad_model or build_grad_model(model, cfg=cfg)
    x = preprocess(img_bgr, img_size)

    with tf.GradientTape() as tape:
        features, preds = grad_model(x, training=False)
        tape.watch(features)
        idx = int(tf.argmax(preds[0])) if class_index is None else int(class_index)
        score = preds[:, idx]

    grads = tape.gradient(score, features)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None - the graph is disconnected")

    weights = tf.reduce_mean(grads, axis=(1, 2))          # global-average-pool the gradients
    cam = tf.reduce_sum(features * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy().astype(np.float32)      # only positive evidence

    heatmap, std, flat = finalize_heatmap(cam, img_bgr.shape[:2], cfg)
    return {
        "heatmap": heatmap,
        "class_index": idx,
        "class_name": cfg["classification"]["classes"][idx],
        "probability": float(preds[0][idx]),
        "heatmap_std": round(std, 6),
        "flat": flat,
    }


def finalize_heatmap(cam, out_shape, cfg=None):
    """Normalize a raw CAM to [0,1] at out_shape; return (heatmap, std, flat).

    A CAM with no spread normalizes to zeros rather than amplifying noise, and
    is reported flat so callers do not present it as an explanation.
    """
    cfg = cfg or load_config()
    cam = np.asarray(cam, np.float32)
    span = float(cam.max() - cam.min())
    cam = (cam - cam.min()) / span if span > 1e-12 else np.zeros_like(cam)
    h, w = out_shape[:2]
    heatmap = np.clip(cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    std = float(heatmap.std())
    return heatmap, std, std < cfg["explain"]["flat_cam_std_eps"]


def colormap_id(name):
    return getattr(cv2, f"COLORMAP_{str(name).upper()}", cv2.COLORMAP_JET)


def overlay(img_bgr, heatmap, cfg=None):
    """Blend a [0,1] heatmap over the image using the configured colormap."""
    cfg = cfg or load_config()
    ecfg = cfg["explain"]
    hm = np.clip(np.asarray(heatmap, np.float32), 0.0, 1.0)
    if hm.shape[:2] != img_bgr.shape[:2]:
        hm = cv2.resize(hm, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((hm * 255).astype(np.uint8), colormap_id(ecfg["colormap"]))
    alpha = float(ecfg["overlay_alpha"])
    return cv2.addWeighted(colored, alpha, img_bgr, 1.0 - alpha, 0.0)
