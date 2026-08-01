"""Export the trained Keras model to a quantized TFLite model for on-device
inference (Step 3, Android path).

Uses post-training DYNAMIC-RANGE quantization (int8 weights, float32
activations computed on the fly) rather than full-integer quantization.
This was a deliberate choice after measuring both on the full held-out test
set: full-integer quantization (int8 weights AND activations, calibrated
against a 200-image representative dataset) collapsed recall from 99.1% to
69.0% and AUC from 0.899 to 0.845 -- MobileNetV3's hard-swish activations
and squeeze-excite blocks are known to be sensitive to naive activation
quantization. Dynamic-range quantization lands at essentially the same
model size (~1.1MB either way) while preserving recall (98.8%) and AUC
(0.894) almost exactly -- given this app's hard requirement to not miss
real bills, that's the only one worth shipping. Re-run this comparison
(see git history for the eval script) if the architecture changes.

Usage:
    python scripts/export_tflite.py
"""

import csv
import random
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
TFLITE_PATH = REPO_ROOT / "models" / "invoice_classifier.tflite"
MANIFEST = REPO_ROOT / "data" / "labeled" / "manifest.csv"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
IMG_SIZE = 224
TARGET_MAX_BYTES = 3 * 1024 * 1024  # README target: under 3MB
DECISION_THRESHOLD = 0.15  # see models/README.md -- recall-optimized, not the 0.5 default
MIN_ACCEPTABLE_RECALL = 0.95  # quantization must not meaningfully erode the recall guarantee


def sample_paths(n: int, seed: int = 1) -> list[Path]:
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(rows)
    return [REPO_ROOT / r["filepath"] for r in rows[:n]]


def convert() -> bytes:
    model = tf.keras.models.load_model(MODEL_PATH)

    # TFLiteConverter.from_keras_model() hits an MLIR bug on this TF version
    # ("missing attribute 'value'" while freezing a conv ReadVariableOp,
    # reproducible even on a freshly-built, never-serialized model -- so it's
    # a converter/tracing issue, not a corrupt checkpoint). Routing through
    # an explicit SavedModel export sidesteps it.
    export_dir = REPO_ROOT / "models" / "_tmp_savedmodel"
    model.export(str(export_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    # Dynamic-range quantization: int8 weights, float activations. No
    # representative_dataset / supported_ops restriction needed -- see the
    # module docstring for why full-integer quantization was rejected.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    shutil.rmtree(export_dir, ignore_errors=True)
    return tflite_model


def run_tflite(interpreter: tf.lite.Interpreter, img_arr: np.ndarray) -> float:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], np.expand_dims(img_arr, 0))
    interpreter.invoke()
    return float(interpreter.get_tensor(out["index"]).ravel()[0])


def benchmark(interpreter: tf.lite.Interpreter, paths: list[Path], n_runs: int = 20) -> None:
    paths = paths[:n_runs]
    times = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        t0 = time.perf_counter()
        run_tflite(interpreter, img)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"\nCPU inference benchmark over {len(paths)} runs (no GPU delegate):")
    print(f"  mean={times.mean():.2f}ms  p50={np.median(times):.2f}ms  p95={np.percentile(times, 95):.2f}ms")


def evaluate_on_test_split(interpreter: tf.lite.Interpreter) -> None:
    """Verify quantization didn't erode recall -- the whole reason dynamic-range
    was chosen over full-integer quantization in the first place."""
    with open(SPLITS_PATH) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    y_true = np.array([int(r["invoice_label"]) for r in rows])
    probs = []
    for r in rows:
        img = np.array(Image.open(REPO_ROOT / r["filepath"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        probs.append(run_tflite(interpreter, img))
    probs = np.array(probs)
    y_pred = (probs >= DECISION_THRESHOLD).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    recall = recall_score(y_true, y_pred)

    print(f"\nTFLite accuracy on held-out test split (n={len(y_true)}, threshold={DECISION_THRESHOLD}):")
    print(f"  accuracy={acc:.4f}  auc={auc:.4f}  recall={recall:.4f}")
    if recall < MIN_ACCEPTABLE_RECALL:
        print(
            f"  WARNING: recall {recall:.4f} is below the {MIN_ACCEPTABLE_RECALL} floor -- "
            "quantization may have eroded the recall guarantee this app depends on."
        )


def main() -> None:
    tflite_model = convert()
    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_model)

    size_bytes = TFLITE_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    status = "OK" if size_bytes <= TARGET_MAX_BYTES else "OVER TARGET"
    print(f"\nSaved {TFLITE_PATH} ({size_mb:.2f} MB) -- target <3MB: {status}")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    benchmark(interpreter, sample_paths(50))
    evaluate_on_test_split(interpreter)


if __name__ == "__main__":
    main()
