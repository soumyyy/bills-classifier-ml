"""Export and evaluate the experimental multi-task verifier as TFLite."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "verifier_v2"
MODEL_DIR = REPO_ROOT / "models" / "experiments" / EXPERIMENT_NAME
LOG_DIR = REPO_ROOT / "logs" / "experiments" / EXPERIMENT_NAME
KERAS_PATH = MODEL_DIR / "model.keras"
TFLITE_PATH = MODEL_DIR / "model.tflite"
TRAINING_METRICS_PATH = LOG_DIR / "metrics.json"
DEPLOYMENT_METRICS_PATH = LOG_DIR / "deployment_metrics.json"
BASELINE_PATH = REPO_ROOT / "logs" / "verifier_large_deployment_metrics.json"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "verifier_large_splits.csv"
BINARY_PATH = REPO_ROOT / "models" / "invoice_classifier.tflite"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
BILL_CLASS_INDICES = (0, 1)
BINARY_THRESHOLD = 0.15


def convert(keras_path: Path, tflite_path: Path) -> bytes:
    model = tf.keras.models.load_model(keras_path)
    temp_root = Path(tempfile.mkdtemp(prefix="bill-verifier-v2-"))
    saved_model = temp_root / "saved_model"
    try:
        model.export(str(saved_model))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = converter.convert()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    tflite_path.write_bytes(blob)
    return blob


def direct_square(path: Path, image_size: int = 224) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )


def white_letterbox(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.contain(
            image.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        canvas.paste(
            image,
            ((image_size - image.width) // 2, (image_size - image.height) // 2),
        )
        return np.asarray(canvas, dtype=np.float32)


def verifier_image(path: Path, image_size: int, resize_mode: str) -> np.ndarray:
    if resize_mode == "direct":
        return direct_square(path, image_size)
    if resize_mode == "letterbox":
        return white_letterbox(path, image_size)
    raise ValueError(f"Unsupported resize mode: {resize_mode}")


def run_binary(interpreter: tf.lite.Interpreter, image: np.ndarray) -> float:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], np.expand_dims(image, 0))
    interpreter.invoke()
    return float(interpreter.get_tensor(output_detail["index"]).ravel()[0])


def run_verifier(
    interpreter: tf.lite.Interpreter, image: np.ndarray
) -> tuple[float, np.ndarray]:
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], np.expand_dims(image, 0))
    interpreter.invoke()
    outputs = [
        interpreter.get_tensor(detail["index"]).ravel().astype(float)
        for detail in interpreter.get_output_details()
    ]
    bill_outputs = [value for value in outputs if value.size == 1]
    class_outputs = [value for value in outputs if value.size == 7]
    if len(bill_outputs) != 1 or len(class_outputs) != 1:
        shapes = [value.shape for value in outputs]
        raise ValueError(f"Unexpected verifier outputs: {shapes}")
    return float(bill_outputs[0][0]), class_outputs[0]


def classification_metrics(true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(np.int32)
    negatives = true == 0
    positives = true == 1
    return {
        "n": int(len(true)),
        "positive": int(positives.sum()),
        "negative": int(negatives.sum()),
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted)),
        "fpr": float(predicted[negatives].mean()),
        "tp": int((predicted & positives).sum()),
        "fn": int(((1 - predicted) & positives).sum()),
        "fp": int((predicted & negatives).sum()),
        "tn": int(((1 - predicted) & negatives).sum()),
        "auc": float(roc_auc_score(true, scores)),
    }


def read_test_rows(splits_path: Path = SPLITS_PATH) -> list[dict]:
    with open(splits_path) as source:
        return [row for row in csv.DictReader(source) if row["split"] == "test"]


def evaluate_test(
    binary: tf.lite.Interpreter,
    verifier: tf.lite.Interpreter,
    verifier_threshold: float,
    image_size: int,
    resize_mode: str,
    splits_path: Path = SPLITS_PATH,
) -> dict:
    rows = read_test_rows(splits_path)
    true = np.array(
        [int(row["class_index"]) in BILL_CLASS_INDICES for row in rows], dtype=np.int32
    )
    binary_scores = []
    verifier_scores = []
    verifier_classes = []
    for index, row in enumerate(rows, start=1):
        path = REPO_ROOT / row["filepath"]
        binary_scores.append(run_binary(binary, direct_square(path)))
        score, classes = run_verifier(
            verifier, verifier_image(path, image_size, resize_mode)
        )
        verifier_scores.append(score)
        verifier_classes.append(int(classes.argmax()))
        if index % 100 == 0:
            print(f"Evaluated {index}/{len(rows)} held-out images")
    binary_scores = np.asarray(binary_scores)
    verifier_scores = np.asarray(verifier_scores)
    gate = binary_scores >= BINARY_THRESHOLD
    cascade_scores = verifier_scores.copy()
    cascade_scores[~gate] = -1.0
    return {
        "binary_gate": classification_metrics(true, binary_scores, BINARY_THRESHOLD),
        "verifier": classification_metrics(true, verifier_scores, verifier_threshold),
        "cascade": classification_metrics(true, cascade_scores, verifier_threshold),
        "verifier_multiclass_accuracy": float(
            accuracy_score(
                np.array([int(row["class_index"]) for row in rows]),
                np.asarray(verifier_classes),
            )
        ),
    }


def folder_paths(folder: Path) -> list[Path]:
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def evaluate_folder(
    folder: Path,
    positive: bool,
    binary: tf.lite.Interpreter,
    verifier: tf.lite.Interpreter,
    verifier_threshold: float,
    image_size: int,
    resize_mode: str,
) -> dict:
    details = []
    correct = 0
    for path in folder_paths(folder):
        binary_score = run_binary(binary, direct_square(path))
        verifier_score, classes = run_verifier(
            verifier, verifier_image(path, image_size, resize_mode)
        )
        standalone_bill = verifier_score >= verifier_threshold
        cascade_bill = binary_score >= BINARY_THRESHOLD and standalone_bill
        is_correct = cascade_bill == positive
        correct += int(is_correct)
        details.append(
            {
                "file": path.name,
                "binary_score": binary_score,
                "verifier_score": verifier_score,
                "verifier_class": int(classes.argmax()),
                "standalone_bill": bool(standalone_bill),
                "cascade_bill": bool(cascade_bill),
                "correct": bool(is_correct),
            }
        )
    return {"correct": correct, "total": len(details), "details": details}


def benchmark(
    binary: tf.lite.Interpreter,
    verifier: tf.lite.Interpreter,
    paths: list[Path],
    image_size: int,
    resize_mode: str,
) -> dict:
    verifier_times = []
    cascade_times = []
    verifier_runs = 0
    # Warm both interpreters before measuring.
    for path in paths[:5]:
        run_binary(binary, direct_square(path))
        run_verifier(verifier, verifier_image(path, image_size, resize_mode))
    for path in paths[:100]:
        square = direct_square(path)
        verifier_input = verifier_image(path, image_size, resize_mode)
        started = time.perf_counter()
        binary_score = run_binary(binary, square)
        gate_ms = (time.perf_counter() - started) * 1000.0
        verifier_ms = 0.0
        if binary_score >= BINARY_THRESHOLD:
            started = time.perf_counter()
            run_verifier(verifier, verifier_input)
            verifier_ms = (time.perf_counter() - started) * 1000.0
            verifier_times.append(verifier_ms)
            verifier_runs += 1
        cascade_times.append(gate_ms + verifier_ms)
    standalone_times = []
    for path in paths[:100]:
        image = verifier_image(path, image_size, resize_mode)
        started = time.perf_counter()
        run_verifier(verifier, image)
        standalone_times.append((time.perf_counter() - started) * 1000.0)

    def summary(values: list[float]) -> dict:
        values_array = np.asarray(values)
        return {
            "runs": int(len(values)),
            "mean_ms": float(values_array.mean()),
            "p50_ms": float(np.percentile(values_array, 50)),
            "p95_ms": float(np.percentile(values_array, 95)),
        }

    return {
        "standalone_verifier": summary(standalone_times),
        "cascade_observed": summary(cascade_times),
        "cascade_verifier_invocations": verifier_runs,
        "verifier_invocation_times": summary(verifier_times),
        "note": "Preprocessing and image decode are excluded from inference timing.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--splits-path", type=Path, default=SPLITS_PATH)
    parser.add_argument("--regression-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = REPO_ROOT / "models" / "experiments" / args.experiment_name
    log_dir = REPO_ROOT / "logs" / "experiments" / args.experiment_name
    keras_path = model_dir / "model.keras"
    tflite_path = model_dir / "model.tflite"
    training_metrics_path = log_dir / "metrics.json"
    deployment_metrics_path = log_dir / "deployment_metrics.json"
    with open(training_metrics_path) as source:
        training_metrics = json.load(source)
    verifier_threshold = float(training_metrics["threshold"])
    image_size = int(training_metrics["image_size"])
    resize_mode = training_metrics.get("resize_mode", "letterbox")
    blob = (
        tflite_path.read_bytes()
        if args.skip_convert
        else convert(keras_path, tflite_path)
    )

    binary = tf.lite.Interpreter(model_path=str(BINARY_PATH), num_threads=1)
    verifier = tf.lite.Interpreter(model_content=blob, num_threads=1)
    binary.allocate_tensors()
    verifier.allocate_tensors()

    rows = read_test_rows(args.splits_path)
    test_paths = [REPO_ROOT / row["filepath"] for row in rows]
    regression = {}
    folders = () if args.skip_regression else (("TestPhotos", True), ("Negative", False), ("N2", False))
    for folder, positive in folders:
        try:
            regression[folder] = evaluate_folder(
                args.regression_root / folder,
                positive,
                binary,
                verifier,
                verifier_threshold,
                image_size,
                resize_mode,
            )
        except (FileNotFoundError, PermissionError, OSError) as error:
            regression[folder] = {"error": f"{type(error).__name__}: {error}"}

    baseline = None
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as source:
            baseline = json.load(source)
    results = {
        "binary_threshold": BINARY_THRESHOLD,
        "verifier_threshold": verifier_threshold,
        "image_size": image_size,
        "resize_mode": resize_mode,
        "model_size_bytes": len(blob),
        "benchmark": benchmark(binary, verifier, test_paths, image_size, resize_mode),
        "test": evaluate_test(
            binary,
            verifier,
            verifier_threshold,
            image_size,
            resize_mode,
            args.splits_path,
        ),
        "regression": regression,
        "production_baseline": baseline,
        "evaluation_note": (
            "The 921-image test split is the primary comparison. Negative contains "
            "12 train, 2 validation, and 3 test examples, so its aggregate regression "
            "score is not an independent generalization estimate."
        ),
    }
    with open(deployment_metrics_path, "w") as output:
        json.dump(results, output, indent=2)
    summary = {
        "model_size_bytes": results["model_size_bytes"],
        "benchmark": results["benchmark"],
        "test": results["test"],
        "regression": {
            folder: (
                f"{metrics['correct']}/{metrics['total']}"
                if "error" not in metrics
                else metrics["error"]
            )
            for folder, metrics in regression.items()
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved deployment evaluation to {deployment_metrics_path}")


if __name__ == "__main__":
    main()
