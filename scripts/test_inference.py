"""Sanity-check the exported TFLite model on a handful of images before
calling the on-device pipeline done (Step 4 of the README).

By default, samples a few real invoice/bill images and a few clearly
non-bill images from the labeled dataset. Pass --images to check specific
files instead (e.g. real phone photos of your own bills).

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --images photo1.jpg photo2.jpg
"""

import argparse
import csv
import json
import random
from pathlib import Path

from _common import require_file

import numpy as np
import tensorflow as tf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
TFLITE_PATH = REPO_ROOT / "models" / "invoice_classifier.tflite"
MANIFEST = REPO_ROOT / "data" / "labeled" / "manifest.csv"
IMG_SIZE = 224
DECISION_THRESHOLD = 0.15  # app operating point; can be overridden on the CLI
DEPLOYMENT_METRICS_PATH = REPO_ROOT / "logs" / "deployment_metrics.json"


def deployment_threshold() -> float:
    if DEPLOYMENT_METRICS_PATH.exists():
        with open(DEPLOYMENT_METRICS_PATH) as f:
            return float(json.load(f)["threshold"])
    return DECISION_THRESHOLD


def default_sample_images(n_per_class: int = 3, seed: int = 3) -> list[tuple[Path, int]]:
    with open(require_file(MANIFEST, "python scripts/build_dataset.py")) as f:
        rows = list(csv.DictReader(f))
    positives = [r for r in rows if r["invoice_label"] == "1" and not r["source"].startswith("synthetic_")]
    negatives = [r for r in rows if r["invoice_label"] == "0" and not r["source"].startswith("synthetic_")]
    rng = random.Random(seed)
    sample = rng.sample(positives, n_per_class) + rng.sample(negatives, n_per_class)
    return [(REPO_ROOT / r["filepath"], int(r["invoice_label"])) for r in sample]


def predict(interpreter: tf.lite.Interpreter, path: Path) -> float:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    interpreter.set_tensor(inp["index"], arr)
    interpreter.invoke()
    return float(interpreter.get_tensor(out["index"]).ravel()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="*", help="Specific image paths to test (skips the default sample)")
    parser.add_argument("--threshold", type=float, help="override the calibrated deployment threshold")
    parser.add_argument(
        "--sample-seed", type=int, default=3,
        help="Seed for the default image sample (was hardcoded).",
    )
    args = parser.parse_args()

    if not TFLITE_PATH.exists():
        raise SystemExit(f"{TFLITE_PATH} not found -- run scripts/export_tflite.py first.")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    threshold = args.threshold if args.threshold is not None else deployment_threshold()

    if args.images:
        items = [(Path(p), None) for p in args.images]
    else:
        items = default_sample_images(seed=args.sample_seed)
        print("No --images given; sampling from the labeled dataset:\n")

    n_correct, n_total = 0, 0
    for path, true_label in items:
        prob = predict(interpreter, path)
        pred_label = int(prob >= threshold)
        pred_str = "invoice/bill" if pred_label else "not invoice/bill"

        line = f"  {path.name:40s} prob={prob:.4f}  ->  {pred_str}"
        if true_label is not None:
            correct = pred_label == true_label
            n_total += 1
            n_correct += correct
            line += "  [OK]" if correct else "  [WRONG]"
        print(line)

    if n_total:
        print(f"\n{n_correct}/{n_total} correct at threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
