"""Export and evaluate the MobileNetV3-Large multiclass verifier as TFLite.

The verifier runs only after the existing binary gate accepts an image. This
script reports standalone verifier metrics and the real two-stage cascade on
the held-out split and the three app regression folders.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
KERAS_PATH = REPO_ROOT / "models" / "invoice_verifier_large.keras"
TFLITE_PATH = REPO_ROOT / "models" / "invoice_verifier_large.tflite"
BINARY_PATH = REPO_ROOT / "models" / "invoice_classifier.tflite"
METRICS_PATH = REPO_ROOT / "logs" / "verifier_large_metrics.json"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "verifier_large_splits.csv"
DEPLOYMENT_PATH = REPO_ROOT / "logs" / "verifier_large_deployment_metrics.json"
IMAGE_SIZE = 224
BINARY_THRESHOLD = 0.15
BILL_CLASS_INDICES = (0, 1)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def convert() -> bytes:
    model = tf.keras.models.load_model(KERAS_PATH)
    saved_model = REPO_ROOT / "models" / "_tmp_verifier_savedmodel"
    model.export(str(saved_model))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = converter.convert()
    shutil.rmtree(saved_model, ignore_errors=True)
    TFLITE_PATH.write_bytes(blob)
    return blob


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)


def run(interpreter: tf.lite.Interpreter, image: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], np.expand_dims(image, 0))
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"]).ravel().astype(float)


def binary_metrics(true: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "n": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted)),
        "recall": float(recall_score(true, predicted)),
        "auc": float(roc_auc_score(true, scores)),
    }


def evaluate_split(
    binary: tf.lite.Interpreter,
    verifier: tf.lite.Interpreter,
    verifier_threshold: float,
) -> dict:
    with open(SPLITS_PATH) as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "test"]
    true = np.array([int(row["class_index"]) in BILL_CLASS_INDICES for row in rows], dtype=int)
    binary_scores = []
    verifier_scores = []
    verifier_classes = []
    for row in rows:
        image = image_array(REPO_ROOT / row["filepath"])
        binary_scores.append(float(run(binary, image)[0]))
        verifier_probability = run(verifier, image)
        verifier_scores.append(float(verifier_probability[list(BILL_CLASS_INDICES)].sum()))
        verifier_classes.append(int(verifier_probability.argmax()))
    binary_scores = np.array(binary_scores)
    verifier_scores = np.array(verifier_scores)
    gate = binary_scores >= BINARY_THRESHOLD
    verified = verifier_scores >= verifier_threshold
    cascade = gate & verified
    return {
        "binary_gate": binary_metrics(true, gate.astype(int), binary_scores),
        "verifier": binary_metrics(true, verified.astype(int), verifier_scores),
        "cascade": binary_metrics(true, cascade.astype(int), binary_scores * verifier_scores),
        "verifier_multiclass_accuracy": float(
            accuracy_score(
                np.array([int(row["class_index"]) for row in rows]),
                np.array(verifier_classes),
            )
        ),
    }


def evaluate_folder(
    folder: Path,
    positive: bool,
    binary: tf.lite.Interpreter,
    verifier: tf.lite.Interpreter,
    verifier_threshold: float,
) -> dict:
    paths = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    correct = 0
    details = []
    for path in paths:
        image = image_array(path)
        gate_score = float(run(binary, image)[0])
        probabilities = run(verifier, image)
        verifier_score = float(probabilities[list(BILL_CLASS_INDICES)].sum())
        cascade_bill = gate_score >= BINARY_THRESHOLD and verifier_score >= verifier_threshold
        is_correct = cascade_bill == positive
        correct += int(is_correct)
        details.append(
            {
                "file": path.name,
                "binary_score": gate_score,
                "verifier_bill_score": verifier_score,
                "verifier_class": int(probabilities.argmax()),
                "cascade_bill": bool(cascade_bill),
                "correct": bool(is_correct),
            }
        )
    return {"correct": correct, "total": len(paths), "details": details}


def benchmark(interpreter: tf.lite.Interpreter, paths: list[Path]) -> dict:
    durations = []
    for path in paths[:50]:
        image = image_array(path)
        started = time.perf_counter()
        run(interpreter, image)
        durations.append((time.perf_counter() - started) * 1000)
    values = np.array(durations)
    return {
        "runs": len(durations),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Evaluate the existing verifier TFLite artifact without converting again.",
    )
    parser.add_argument(
        "--regression-root",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing TestPhotos, Negative, and N2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(METRICS_PATH) as source:
        training_metrics = json.load(source)
    verifier_threshold = float(training_metrics["threshold"])
    blob = TFLITE_PATH.read_bytes() if args.skip_convert else convert()
    binary = tf.lite.Interpreter(model_path=str(BINARY_PATH), num_threads=1)
    verifier = tf.lite.Interpreter(model_content=blob, num_threads=1)
    binary.allocate_tensors()
    verifier.allocate_tensors()

    with open(SPLITS_PATH) as source:
        benchmark_paths = [
            REPO_ROOT / row["filepath"]
            for row in csv.DictReader(source)
            if row["split"] == "test"
        ]
    regression = {}
    for folder, positive in (("TestPhotos", True), ("Negative", False), ("N2", False)):
        try:
            regression[folder] = evaluate_folder(
                args.regression_root / folder,
                positive,
                binary,
                verifier,
                verifier_threshold,
            )
        except (FileNotFoundError, PermissionError) as error:
            regression[folder] = {"error": f"{type(error).__name__}: {error}"}

    results = {
        "binary_threshold": BINARY_THRESHOLD,
        "verifier_threshold": verifier_threshold,
        "model_size_bytes": len(blob),
        "benchmark": benchmark(verifier, benchmark_paths),
        "test": evaluate_split(binary, verifier, verifier_threshold),
        "regression": regression,
    }
    with open(DEPLOYMENT_PATH, "w") as output:
        json.dump(results, output, indent=2)
    summary = {
        **{key: value for key, value in results.items() if key != "regression"},
        "regression": {
            folder: (
                f"{metrics['correct']}/{metrics['total']}"
                if "error" not in metrics
                else metrics["error"]
            )
            for folder, metrics in results["regression"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    for folder, metrics in results["regression"].items():
        if "error" in metrics:
            continue
        misses = [item for item in metrics["details"] if not item["correct"]]
        if misses:
            print(f"\n{folder} misses:")
            for item in misses:
                print(
                    f"  {item['file']}: gate={item['binary_score']:.4f} "
                    f"verifier={item['verifier_bill_score']:.4f} class={item['verifier_class']}"
                )


if __name__ == "__main__":
    main()
