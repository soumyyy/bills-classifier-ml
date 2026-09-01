"""Calibrate the two heads of V2 and write a reproducible production report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import evaluate_verifier_v2 as evaluation


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = REPO_ROOT / "data" / "labeled" / "verifier_v2_production_splits.csv"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "experiments" / "verifier_v2" / "calibration.json"
DATALESS_FLAG = 0x40000000
BILL_CLASSES = (0, 1)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def ensemble_score(direct: np.ndarray, class_bill: np.ndarray, alpha: float) -> np.ndarray:
    return sigmoid(alpha * logit(direct) + (1.0 - alpha) * logit(class_bill))


def threshold_for_recall(true: np.ndarray, scores: np.ndarray, target: float) -> float:
    positives = np.sort(scores[true == 1])
    required = int(np.ceil(target * len(positives)))
    index = len(positives) - required
    return float(np.nextafter(positives[index], -np.inf))


def metrics(true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive = true == 1
    negative = ~positive
    return {
        "n": int(len(true)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "recall": float(predicted[positive].mean()),
        "fpr": float(predicted[negative].mean()),
        "precision": float(true[predicted].mean()),
        "auc": float(roc_auc_score(true, scores)),
        "tp": int((predicted & positive).sum()),
        "fn": int((~predicted & positive).sum()),
        "fp": int((predicted & negative).sum()),
        "tn": int((~predicted & negative).sum()),
    }


def load_rows(path: Path) -> list[dict]:
    with open(path) as source:
        return list(csv.DictReader(source))


def score_rows(interpreter: tf.lite.Interpreter, rows: list[dict]) -> tuple[np.ndarray, ...]:
    true = []
    direct = []
    class_bill = []
    for index, row in enumerate(rows, start=1):
        path = REPO_ROOT / row["filepath"]
        if getattr(path.stat(), "st_flags", 0) & DATALESS_FLAG:
            raise RuntimeError(
                f"Calibration split contains an iCloud placeholder: {path}. "
                "Materialize the full split or pass a deliberate local-subset split."
            )
        image = evaluation.direct_square(path)
        direct_score, classes = evaluation.run_verifier(interpreter, image)
        true.append(int(int(row["class_index"]) in BILL_CLASSES))
        direct.append(direct_score)
        class_bill.append(float(classes[list(BILL_CLASSES)].sum()))
        if index % 200 == 0:
            print(f"Scored {index}/{len(rows)} {row['split']} rows")
    return np.asarray(true), np.asarray(direct), np.asarray(class_bill)


def choose_calibration(
    true: np.ndarray,
    direct: np.ndarray,
    class_bill: np.ndarray,
    target_recall: float,
    alpha_steps: int,
) -> dict:
    best = None
    for alpha in np.linspace(0.0, 1.0, alpha_steps):
        scores = ensemble_score(direct, class_bill, float(alpha))
        threshold = threshold_for_recall(true, scores, target_recall)
        result = metrics(true, scores, threshold)
        candidate = (result["fpr"], -result["precision"], float(alpha), threshold, result)
        if best is None or candidate < best:
            best = candidate
    _, _, alpha, threshold, result = best
    return {
        "alpha_direct_head": alpha,
        "alpha_class_bill_head": 1.0 - alpha,
        "threshold": threshold,
        "metrics": result,
    }


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate_folder(
    interpreter: tf.lite.Interpreter,
    folder: Path,
    positive: bool,
    alpha: float,
    threshold: float,
    exclude_digests: set[str] | None = None,
) -> dict:
    details = []
    skipped_offloaded = []
    skipped_duplicates = []
    for path in evaluation.folder_paths(folder):
        if getattr(path.stat(), "st_flags", 0) & DATALESS_FLAG:
            skipped_offloaded.append(path.name)
            continue
        file_digest = digest(path)
        if exclude_digests is not None and file_digest in exclude_digests:
            skipped_duplicates.append(path.name)
            continue
        direct_score, classes = evaluation.run_verifier(
            interpreter, evaluation.direct_square(path)
        )
        class_score = float(classes[list(BILL_CLASSES)].sum())
        score = float(
            ensemble_score(
                np.asarray([direct_score]), np.asarray([class_score]), alpha
            )[0]
        )
        predicted = score >= threshold
        details.append(
            {
                "file": path.name,
                "direct_score": direct_score,
                "class_bill_score": class_score,
                "ensemble_score": score,
                "predicted_bill": bool(predicted),
                "correct": bool(predicted == positive),
            }
        )
    return {
        "correct": sum(item["correct"] for item in details),
        "evaluated": len(details),
        "skipped_offloaded": skipped_offloaded,
        "skipped_duplicates": skipped_duplicates,
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--target-validation-recall", type=float, default=1.0)
    parser.add_argument("--alpha-steps", type=int, default=21)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tflite-path", type=Path, default=evaluation.TFLITE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interpreter = tf.lite.Interpreter(
        model_path=str(args.tflite_path), num_threads=1
    )
    interpreter.allocate_tensors()
    rows = load_rows(args.splits_path)
    validation_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    validation = score_rows(interpreter, validation_rows)
    test = score_rows(interpreter, test_rows)

    calibration = choose_calibration(
        *validation, args.target_validation_recall, args.alpha_steps
    )
    alpha = calibration["alpha_direct_head"]
    threshold = calibration["threshold"]
    test_scores = ensemble_score(test[1], test[2], alpha)

    negative_digests = {
        digest(path)
        for path in evaluation.folder_paths(REPO_ROOT / "Negative")
        if not (getattr(path.stat(), "st_flags", 0) & DATALESS_FLAG)
    }
    folders = {
        "TestPhotos": evaluate_folder(
            interpreter, REPO_ROOT / "TestPhotos", True, alpha, threshold
        ),
        "Negative_fitted": evaluate_folder(
            interpreter, REPO_ROOT / "Negative", False, alpha, threshold
        ),
        "N2_all_available": evaluate_folder(
            interpreter, REPO_ROOT / "N2", False, alpha, threshold
        ),
        "N2_independent_available": evaluate_folder(
            interpreter,
            REPO_ROOT / "N2",
            False,
            alpha,
            threshold,
            exclude_digests=negative_digests,
        ),
    }
    result = {
        "architecture": "single MobileNetV3Large V2; direct and class heads ensembled",
        "model_size_bytes": args.tflite_path.stat().st_size,
        "model_path": str(args.tflite_path),
        "target_validation_recall": args.target_validation_recall,
        "calibration": calibration,
        "test": metrics(test[0], test_scores, threshold),
        "folders": folders,
        "data_coverage": {
            "split_rows_evaluated": len(validation_rows) + len(test_rows),
            "split_path": str(args.splits_path),
            "warning": (
                "This report used the locally materialized subset when a /tmp split "
                "was supplied. Rerun on the full split after iCloud placeholders download."
            ),
        },
        "interpretation": (
            "Negative_fitted is a regression result, not generalization evidence. "
            "N2_independent_available excludes byte-identical overlap with Negative."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
    summary = {
        "calibration": calibration,
        "test": result["test"],
        "folders": {
            name: {
                "correct": value["correct"],
                "evaluated": value["evaluated"],
                "skipped_offloaded": value["skipped_offloaded"],
                "skipped_duplicates": value["skipped_duplicates"],
            }
            for name, value in folders.items()
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
