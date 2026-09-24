"""TFLite conversion and benchmark harness.

Converts any .h5 classifier to float32, float16 and int8 TFLite, measures the
size and mean inference latency of each, and - when a labelled test set is
available - reports accuracy before and after conversion.

READ THE LABEL ON THE OUTPUT. The only classifier this project has is parked
because it learned image provenance rather than pathology, so running this on it
measures engineering feasibility (does the model fit and run fast enough on a
phone?) and nothing else. The accuracy column is there to show what quantisation
costs relative to the same model unquantised; it is not a performance claim, and
the underlying model must not ship.

Usage:
  python -m src.deploy.quantize                        # parked classifier
  python -m src.deploy.quantize --model path/to.h5     # any other model
"""

import argparse
import time

import numpy as np
import tensorflow as tf

from src.utils.config import PROJECT_ROOT, load_config

FEASIBILITY_LABEL = (
    "FEASIBILITY MEASUREMENT ON A PARKED MODEL. The classifier below is not fit for "
    "use: Phase 2b showed it separates classes by image provenance rather than by "
    "pathology. These numbers describe size and speed on this hardware. The accuracy "
    "column shows only what quantisation costs relative to the same parked model; it "
    "is NOT a performance claim and must never be quoted as one."
)


def load_test_arrays(cfg, limit=None):
    """(X preprocessed, y) from the processed test split, or (None, None)."""
    classes = cfg["classification"]["classes"]
    size = cfg["classification"]["img_size"]
    root = PROJECT_ROOT / cfg["paths"]["processed"] / "classification" / "test"
    paths, labels = [], []
    for index, name in enumerate(classes):
        for p in sorted((root / name).glob("*.jpg")):
            paths.append(p)
            labels.append(index)
    if not paths:
        return None, None
    if limit:
        paths, labels = paths[:limit], labels[:limit]

    images = []
    for p in paths:
        raw = tf.io.decode_jpeg(tf.io.read_file(str(p)), channels=3)
        raw = tf.image.resize(raw, (size, size))
        images.append(tf.keras.applications.mobilenet_v2.preprocess_input(raw).numpy())
    return np.stack(images).astype(np.float32), np.asarray(labels)


def representative_dataset(x, samples):
    def generator():
        for row in x[:samples]:
            yield [row[None, ...].astype(np.float32)]
    return generator


def convert(model, variant, x_rep=None, samples=100):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif variant == "int8":
        if x_rep is None:
            raise ValueError("int8 needs representative data")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(x_rep, samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # float input/output keeps the harness comparable across variants
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    return converter.convert()


def interpreter_for(blob):
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    return interp


def predict_tflite(interp, x):
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    out = []
    for row in x:
        interp.set_tensor(in_detail["index"], row[None, ...].astype(in_detail["dtype"]))
        interp.invoke()
        out.append(interp.get_tensor(out_detail["index"])[0])
    return np.asarray(out)


def measure_latency(interp, shape, runs, warmup):
    in_detail = interp.get_input_details()[0]
    sample = np.zeros(shape, dtype=in_detail["dtype"])
    for _ in range(warmup):
        interp.set_tensor(in_detail["index"], sample)
        interp.invoke()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        interp.set_tensor(in_detail["index"], sample)
        interp.invoke()
        times.append((time.perf_counter() - start) * 1000.0)
    times = np.asarray(times)
    return float(times.mean()), float(np.percentile(times, 95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="path to a .h5 (default: parked classifier)")
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    dcfg = cfg["deploy"]
    runs = args.runs or dcfg["latency_runs"]
    n_classes = len(cfg["classification"]["classes"])
    model_path = (PROJECT_ROOT / args.model if args.model
                  else PROJECT_ROOT / cfg["paths"]["models"] / f"classifier_{n_classes}class.h5")
    out_dir = PROJECT_ROOT / cfg["paths"]["processed"] / "deploy"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    x, y = load_test_arrays(cfg)
    size = cfg["classification"]["img_size"]

    lines = ["=" * 78, "TFLITE CONVERSION AND BENCHMARK", "=" * 78, "",
             FEASIBILITY_LABEL, "",
             f"model:        {model_path.relative_to(PROJECT_ROOT)}",
             f"input:        {size}x{size}x3, {n_classes} classes",
             f"latency:      mean of {runs} runs after {dcfg['warmup_runs']} warm-up, "
             f"batch 1, CPU", ""]

    keras_size = model_path.stat().st_size / 1e6
    if x is not None:
        keras_acc = float((model.predict(x, verbose=0).argmax(axis=1) == y).mean())
        lines.append(f"test set:     {len(y)} images from the de-biased split")
        lines.append(f"Keras .h5:    {keras_size:.2f} MB, accuracy {keras_acc:.4f}")
    else:
        keras_acc = None
        lines.append(f"Keras .h5:    {keras_size:.2f} MB (no labelled test set found)")
    lines += ["", f"{'variant':<10} {'size MB':>9} {'vs .h5':>8} {'mean ms':>9} "
                  f"{'p95 ms':>8} {'accuracy':>9} {'delta':>8}", "-" * 78]

    results = {}
    for variant in dcfg["tflite_variants"]:
        try:
            blob = convert(model, variant, x, dcfg["representative_samples"])
        except Exception as exc:  # conversion genuinely can fail per variant
            lines.append(f"{variant:<10} conversion failed: {exc}")
            continue
        path = out_dir / f"classifier_{n_classes}class_{variant}.tflite"
        path.write_bytes(blob)

        interp = interpreter_for(blob)
        mean_ms, p95_ms = measure_latency(interp, (1, size, size, 3), runs,
                                          dcfg["warmup_runs"])
        mb = len(blob) / 1e6
        if x is not None:
            acc = float((predict_tflite(interp, x).argmax(axis=1) == y).mean())
            acc_txt, delta_txt = f"{acc:.4f}", f"{acc - keras_acc:+.4f}"
        else:
            acc, acc_txt, delta_txt = None, "n/a", "n/a"
        results[variant] = {"mb": mb, "mean_ms": mean_ms, "p95_ms": p95_ms, "accuracy": acc}
        lines.append(f"{variant:<10} {mb:>9.2f} {mb / keras_size:>7.0%} {mean_ms:>9.2f} "
                     f"{p95_ms:>8.2f} {acc_txt:>9} {delta_txt:>8}")

    budget = dcfg["size_budget_mb"]
    lines += ["", f"Phase 8 size budget: {budget} MB per model."]
    for variant, r in results.items():
        verdict = "within budget" if r["mb"] <= budget else "OVER BUDGET"
        lines.append(f"  {variant:<9} {r['mb']:.2f} MB - {verdict}")
    lines += ["", FEASIBILITY_LABEL, ""]

    report = "\n".join(lines)
    (out_dir / "quantization_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved {out_dir / 'quantization_report.txt'}")


if __name__ == "__main__":
    main()
