"""Compare baseline and screenshot-tuned V2 models on identical held-out groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import calibrate_verifier_v2 as calibration
import evaluate_verifier_v2 as evaluation


REPO_ROOT = Path(__file__).resolve().parent.parent
BILL_CLASSES = (0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-splits", type=Path, default=Path("/tmp/verifier_v2_production_local_splits.csv"))
    parser.add_argument(
        "--screenshot-manifest",
        type=Path,
        default=REPO_ROOT / "data/raw/screenshot_negatives_aitw/manifest.csv",
    )
    parser.add_argument(
        "--screenshot-positive-manifest",
        type=Path,
        default=REPO_ROOT / "data/raw/screenshot_positive_bills/manifest.csv",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=REPO_ROOT / "models/experiments/verifier_v2/model.tflite",
    )
    parser.add_argument(
        "--baseline-calibration",
        type=Path,
        default=REPO_ROOT / "logs/experiments/verifier_v2/calibration_balanced.json",
    )
    parser.add_argument(
        "--candidate-model",
        type=Path,
        default=REPO_ROOT / "models/experiments/verifier_v2_screenshot/model.tflite",
    )
    parser.add_argument(
        "--candidate-calibration",
        type=Path,
        default=REPO_ROOT / "logs/experiments/verifier_v2_screenshot/calibration_balanced.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs/experiments/verifier_v2_screenshot/comparison.json",
    )
    return parser.parse_args()


def read_rows(path: Path, split: str) -> list[dict]:
    with open(path, newline="") as source:
        return [row for row in csv.DictReader(source) if row["split"] == split]


def load_operating_point(path: Path) -> tuple[float, float]:
    with open(path) as source:
        report = json.load(source)
    point = report["calibration"]
    return float(point["alpha_direct_head"]), float(point["threshold"])


def load_interpreter(path: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter


def scores(interpreter: tf.lite.Interpreter, rows: list[dict]) -> np.ndarray:
    values = []
    for index, row in enumerate(rows, start=1):
        direct, classes = evaluation.run_verifier(
            interpreter, evaluation.direct_square(REPO_ROOT / row["filepath"])
        )
        values.append((direct, float(classes[list(BILL_CLASSES)].sum())))
        if index % 500 == 0:
            print(f"Scored {index}/{len(rows)}")
    return np.asarray(values)


def combine(raw_scores: np.ndarray, alpha: float) -> np.ndarray:
    return calibration.ensemble_score(raw_scores[:, 0], raw_scores[:, 1], alpha)


def screenshot_metrics(values: np.ndarray, threshold: float) -> dict:
    predicted = values >= threshold
    return {
        "n": int(len(values)),
        "false_positives": int(predicted.sum()),
        "fpr": float(predicted.mean()),
        "mean_score": float(values.mean()),
        "p95_score": float(np.percentile(values, 95)),
        "max_score": float(values.max()),
    }


def screenshot_bill_metrics(values: np.ndarray, threshold: float) -> dict:
    predicted = values >= threshold
    return {
        "n": int(len(values)),
        "true_positives": int(predicted.sum()),
        "false_negatives": int((~predicted).sum()),
        "recall": float(predicted.mean()),
        "mean_score": float(values.mean()),
        "p05_score": float(np.percentile(values, 5)),
        "min_score": float(values.min()),
    }


def screenshot_bill_context_metrics(
    rows: list[dict], values: np.ndarray, threshold: float
) -> dict:
    result = {}
    contexts = sorted({row.get("context", "unknown") for row in rows})
    for context in contexts:
        indices = [index for index, row in enumerate(rows) if row.get("context", "unknown") == context]
        context_values = values[indices]
        result[context] = screenshot_bill_metrics(context_values, threshold)
    return result


def main() -> None:
    args = parse_args()
    base_rows = read_rows(args.base_splits, "test")
    screenshot_rows = read_rows(args.screenshot_manifest, "test")
    screenshot_bill_rows = read_rows(args.screenshot_positive_manifest, "test")
    true = np.asarray(
        [int(int(row["class_index"]) in BILL_CLASSES) for row in base_rows]
    )
    baseline_alpha, baseline_threshold = load_operating_point(args.baseline_calibration)
    candidate_alpha, candidate_threshold = load_operating_point(args.candidate_calibration)

    results = {}
    decisions = {}
    for name, model_path, alpha, threshold in (
        ("baseline", args.baseline_model, baseline_alpha, baseline_threshold),
        ("candidate", args.candidate_model, candidate_alpha, candidate_threshold),
    ):
        print(f"=== {name} ===")
        interpreter = load_interpreter(model_path)
        base_values = combine(scores(interpreter, base_rows), alpha)
        screenshot_values = combine(scores(interpreter, screenshot_rows), alpha)
        screenshot_bill_values = combine(scores(interpreter, screenshot_bill_rows), alpha)
        results[name] = {
            "model": str(model_path),
            "alpha_direct": alpha,
            "threshold": threshold,
            "original_test": calibration.metrics(true, base_values, threshold),
            "screenshot_test": screenshot_metrics(screenshot_values, threshold),
            "screenshot_bill_test": screenshot_bill_metrics(
                screenshot_bill_values, threshold
            ),
            "screenshot_bill_by_context": screenshot_bill_context_metrics(
                screenshot_bill_rows, screenshot_bill_values, threshold
            ),
            "screenshot_bill_misses": [
                {
                    "filepath": row["filepath"],
                    "parent_filepath": row.get("parent_filepath"),
                    "context": row.get("context"),
                    "score": float(score),
                }
                for row, score in zip(screenshot_bill_rows, screenshot_bill_values)
                if score < threshold
            ],
        }
        decisions[name] = {
            "base": base_values >= threshold,
            "screenshots": screenshot_values >= threshold,
            "screenshot_bills": screenshot_bill_values >= threshold,
        }

    base_positive = true == 1
    base_negative = ~base_positive
    comparison = {
        "screenshots_fixed": int(
            ((decisions["baseline"]["screenshots"]) & ~decisions["candidate"]["screenshots"]).sum()
        ),
        "screenshots_regressed": int(
            ((~decisions["baseline"]["screenshots"]) & decisions["candidate"]["screenshots"]).sum()
        ),
        "screenshot_bills_lost": int(
            (
                decisions["baseline"]["screenshot_bills"]
                & ~decisions["candidate"]["screenshot_bills"]
            ).sum()
        ),
        "screenshot_bills_recovered": int(
            (
                ~decisions["baseline"]["screenshot_bills"]
                & decisions["candidate"]["screenshot_bills"]
            ).sum()
        ),
        "original_negatives_fixed": int(
            (
                decisions["baseline"]["base"]
                & ~decisions["candidate"]["base"]
                & base_negative
            ).sum()
        ),
        "original_negatives_regressed": int(
            (
                ~decisions["baseline"]["base"]
                & decisions["candidate"]["base"]
                & base_negative
            ).sum()
        ),
        "original_bills_lost": int(
            (
                decisions["baseline"]["base"]
                & ~decisions["candidate"]["base"]
                & base_positive
            ).sum()
        ),
        "original_bills_recovered": int(
            (
                ~decisions["baseline"]["base"]
                & decisions["candidate"]["base"]
                & base_positive
            ).sum()
        ),
    }
    output = {"results": results, "paired_changes": comparison}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as destination:
        json.dump(output, destination, indent=2)
    print(json.dumps(output, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
