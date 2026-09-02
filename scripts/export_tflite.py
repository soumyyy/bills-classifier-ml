"""Export the trained Keras model to a quantized TFLite model for on-device
inference (Step 3, Android path).

Uses post-training DYNAMIC-RANGE quantization (int8 weights, float32
activations computed on the fly) rather than full-integer quantization.
The original model's full-integer export substantially reduced recall;
dynamic-range quantization is therefore retained and every new export is
checked on the full test split. Re-run a full-int8 comparison if the
architecture or converter changes.

Usage:
    python scripts/export_tflite.py
"""

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

from _common import check_split_matches, require_file

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
DECISION_THRESHOLD = 0.15  # app operating point; see logs/deployment_metrics.json
DEPLOYMENT_METRICS_PATH = REPO_ROOT / "logs" / "deployment_metrics.json"
FINAL_METRICS_PATH = REPO_ROOT / "logs" / "final_metrics.json"
MIN_ACCEPTABLE_RECALL = 0.95  # quantization must not meaningfully erode the recall guarantee


def sample_paths(n: int, seed: int = 1) -> list[Path]:
    with open(require_file(MANIFEST, "python scripts/build_dataset.py")) as f:
        rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(rows)
    return [REPO_ROOT / r["filepath"] for r in rows[:n]]


def recorded_run() -> dict | None:
    """The run fingerprint saved when the current model was trained."""
    if not FINAL_METRICS_PATH.exists():
        return None
    with open(FINAL_METRICS_PATH) as f:
        return json.load(f).get("run")


def deployment_threshold() -> float:
    if DEPLOYMENT_METRICS_PATH.exists():
        with open(DEPLOYMENT_METRICS_PATH) as f:
            return float(json.load(f)["threshold"])
    return DECISION_THRESHOLD


def convert() -> bytes:
    model = tf.keras.models.load_model(require_file(MODEL_PATH, "python scripts/train.py"))

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


def evaluate_on_test_split(interpreter: tf.lite.Interpreter, threshold: float) -> None:
    """Verify quantization didn't erode recall -- the whole reason dynamic-range
    was chosen over full-integer quantization in the first place."""
    with open(require_file(SPLITS_PATH, "python scripts/train.py")) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

    y_true = np.array([int(r["invoice_label"]) for r in rows])
    probs = []
    for r in rows:
        img = np.array(Image.open(REPO_ROOT / r["filepath"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        probs.append(run_tflite(interpreter, img))
    probs = np.array(probs)
    y_pred = (probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    recall = recall_score(y_true, y_pred)

    print(f"\nTFLite accuracy on held-out test split (n={len(y_true)}, threshold={threshold:.4f}):")
    print(f"  accuracy={acc:.4f}  auc={auc:.4f}  recall={recall:.4f}")
    return recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-split-mismatch",
        action="store_true",
        help="Evaluate even when splits.csv no longer matches the split the model was trained on. The resulting metrics describe a different split from the one the model saw.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1,
        help="Seed for the benchmark image sample (was hardcoded).",
    )
    parser.add_argument(
        "--allow-recall-regression",
        action="store_true",
        help=(
            f"Export even when test recall is below {MIN_ACCEPTABLE_RECALL}. "
            "Without this the model is not written at all."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold used for the export check (defaults to deployment metrics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold = deployment_threshold() if args.threshold is None else args.threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    check_split_matches(recorded_run(), SPLITS_PATH, allow_mismatch=args.allow_split_mismatch)

    tflite_model = convert()

    size_bytes = len(tflite_model)
    size_mb = size_bytes / (1024 * 1024)
    status = "OK" if size_bytes <= TARGET_MAX_BYTES else "OVER TARGET"
    print(f"\nConverted model is {size_mb:.2f} MB -- target <3MB: {status}")

    # Evaluated from the in-memory model, before anything is written. The
    # recall check below used to run after write_bytes and only print a
    # warning, so a quantized model that had eroded the recall guarantee was
    # already on disk - and the script still exited 0, so CI would not have
    # noticed either.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    benchmark(interpreter, sample_paths(50, seed=args.sample_seed))
    recall = evaluate_on_test_split(interpreter, threshold)

    if recall < MIN_ACCEPTABLE_RECALL and not args.allow_recall_regression:
        raise SystemExit(
            f"\nAborted: recall {recall:.4f} is below the "
            f"{MIN_ACCEPTABLE_RECALL} floor, so {TFLITE_PATH.name} was NOT "
            "written and the existing model is untouched. Quantization has "
            "eroded the recall guarantee the app depends on. Re-run with "
            "--allow-recall-regression to export anyway."
        )

    TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_model)
    print(f"\nSaved {TFLITE_PATH}")


if __name__ == "__main__":
    main()
