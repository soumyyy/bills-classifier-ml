"""Score DocLayNet pages and build source-balanced hard-negative splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

import calibrate_verifier_v2 as calibration
import evaluate_verifier_v2 as evaluation


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data/labeled/doclaynet_hard_negatives.csv"
DEFAULT_MODEL = (
    REPO_ROOT / "models/experiments/verifier_v2_screenshot_balanced/model.tflite"
)
DEFAULT_CALIBRATION = (
    REPO_ROOT / "logs/experiments/verifier_v2_screenshot_balanced/calibration_balanced.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/labeled/doclaynet_mined_splits.csv"
DEFAULT_REPORT = REPO_ROOT / "logs/experiments/doclaynet_hard_negative_mining.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--train-per-category", type=int, default=200)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def score(interpreter: tf.lite.Interpreter, path: Path, alpha: float) -> float:
    direct, classes = evaluation.run_verifier(
        interpreter, evaluation.direct_square(path)
    )
    class_bill = float(classes[[0, 1]].sum())
    return float(
        calibration.ensemble_score(
            np.asarray([direct]), np.asarray([class_bill]), alpha
        )[0]
    )


def summarize(rows: list[dict], threshold: float) -> dict:
    values = np.asarray([float(row["baseline_score"]) for row in rows])
    return {
        "n": len(rows),
        "false_positives": int((values >= threshold).sum()),
        "fpr": float((values >= threshold).mean()),
        "mean_score": float(values.mean()),
        "p50_score": float(np.median(values)),
        "p90_score": float(np.percentile(values, 90)),
        "p95_score": float(np.percentile(values, 95)),
        "max_score": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="") as source:
        rows = list(csv.DictReader(source))
    calibration_report = json.loads(args.calibration.read_text())
    point = calibration_report["calibration"]
    alpha = float(point["alpha_direct_head"])
    threshold = float(point["threshold"])
    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=1)
    interpreter.allocate_tensors()

    scored = []
    for index, original in enumerate(rows, start=1):
        row = dict(original)
        row["baseline_score"] = score(
            interpreter, REPO_ROOT / row["filepath"], alpha
        )
        scored.append(row)
        if index % 250 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)}")

    by_split_category: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored:
        by_split_category[(row["split"], row["doc_category"])].append(row)

    selected = []
    selected_counts = Counter()
    for (split, category), category_rows in sorted(by_split_category.items()):
        if split == "train":
            category_rows.sort(key=lambda row: float(row["baseline_score"]), reverse=True)
            category_rows = category_rows[: args.train_per_category]
            for row in category_rows:
                row["selection"] = (
                    "baseline_false_positive"
                    if float(row["baseline_score"]) >= threshold
                    else "category_top_score"
                )
        else:
            for row in category_rows:
                row["selection"] = "external_holdout"
        selected.extend(category_rows)
        selected_counts[(split, category)] += len(category_rows)

    fieldnames = list(rows[0]) + ["baseline_score", "selection"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    split_report = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in scored if row["split"] == split]
        split_report[split] = {
            "overall": summarize(split_rows, threshold),
            "by_category": {
                category: summarize(
                    [row for row in split_rows if row["doc_category"] == category],
                    threshold,
                )
                for category in sorted({row["doc_category"] for row in split_rows})
            },
        }
    result = {
        "source_manifest": str(args.manifest),
        "baseline_model": str(args.model),
        "alpha_direct": alpha,
        "threshold": threshold,
        "downloaded_rows": len(scored),
        "selected_rows": len(selected),
        "selected_counts": {
            f"{split}/{category}": count
            for (split, category), count in sorted(selected_counts.items())
        },
        "scores": split_report,
        "interpretation": (
            "Train contains the highest-scoring pages per category. Validation and test "
            "retain every downloaded page and must remain external negative holdouts."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output} and {args.report}")


if __name__ == "__main__":
    main()
