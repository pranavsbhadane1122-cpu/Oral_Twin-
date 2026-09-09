"""Convert a classifier .h5 saved by Colab's newer Keras 3 (TF 2.20) into a
file loadable by the local TF 2.15 / Keras 2 install.

Keras 3 h5 files store an InputLayer config (batch_shape/optional kwargs) that
Keras 2 cannot deserialize, and order nested-model weights differently, so
neither load_model nor positional load_weights works. This script instead
rebuilds the architecture with src.classification.model.build_model and copies
every weight BY NAME from the h5, then re-saves.

Usage: python scripts/convert_colab_model.py <in.h5> <out.h5>
"""

import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def collect_datasets(h5group, prefix=""):
    out = {}
    for key in h5group:
        item = h5group[key]
        if isinstance(item, h5py.Dataset):
            out[prefix + key] = np.asarray(item)
        else:
            out.update(collect_datasets(item, prefix + key + "/"))
    return out


def candidate_keys(weight_name):
    """Possible h5 paths for one Keras-2 weight name (without ':0')."""
    n = weight_name
    cands = [n]
    top = n.split("/")[0]
    cands.append(f"{top}/{n}")  # head layers: probs/kernel -> probs/probs/kernel
    base = "mobilenetv2_1.00_224"
    cands.append(f"{base}/{n}")  # base layers stored under the base group
    for c in list(cands):
        if "depthwise_kernel" in c:
            cands.append(c.replace("depthwise_kernel", "kernel"))
    return cands


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    ccfg = load_config()["classification"]
    model = build_model(len(ccfg["classes"]), ccfg["img_size"], ccfg["dropout"])

    with h5py.File(src, "r") as f:
        stored = collect_datasets(f["model_weights"])

    assigned, missing = 0, []
    for w in model.weights:
        name = w.name.split(":")[0]
        for key in candidate_keys(name):
            if key in stored:
                arr = stored[key]
                if tuple(arr.shape) != tuple(w.shape):
                    raise SystemExit(f"shape mismatch for {name}: h5 {arr.shape} vs model {w.shape}")
                w.assign(arr)
                assigned += 1
                break
        else:
            missing.append(name)

    if missing:
        raise SystemExit(f"unmatched weights ({len(missing)}): {missing[:10]}")
    print(f"assigned {assigned}/{len(model.weights)} weights by name")
    model.save(dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
