"""Mine AITW screenshots and append hard negatives to the V2 experiment split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

import evaluate_verifier_v2 as evaluation


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_SPLITS = Path("/tmp/verifier_v2_production_local_splits.csv")
DEFAULT_MANIFEST = REPO_ROOT / "data" / "raw" / "screenshot_negatives_aitw" / "manifest.csv"
DEFAULT_MODEL = REPO_ROOT / "models" / "experiments" / "verifier_v2" / "model.tflite"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "labeled" / "verifier_v2_screenshot_splits.csv"
DEFAULT_REPORT = REPO_ROOT / "logs" / "experiments" / "verifier_v2_screenshot" / "mining.json"
FIELDS = ("filepath", "class_index", "class_name", "split", "source_doc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-splits", type=Path, default=DEFAULT_BASE_SPLITS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train-count", type=int, default=2500)
    parser.add_argument("--easy-diversity-count", type=int, default=250)
    parser.add_argument("--baseline-threshold", type=float, default=0.04790887981653213)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as source:
        return list(csv.DictReader(source))


def score_manifest(model_path: Path, rows: list[dict]) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    scores = []
    for index, row in enumerate(rows, start=1):
        image = evaluation.direct_square(REPO_ROOT / row["filepath"])
        score, _ = evaluation.run_verifier(interpreter, image)
        scores.append(score)
        if index % 500 == 0:
            print(f"Scored {index}/{len(rows)} screenshot candidates")
    return np.asarray(scores, dtype=np.float64)


def score_summary(scores: np.ndarray, threshold: float) -> dict:
    return {
        "n": int(len(scores)),
        "false_positive_rate": float((scores >= threshold).mean()),
        "mean": float(scores.mean()),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "max": float(scores.max()),
    }


def as_split_row(row: dict) -> dict:
    return {
        "filepath": row["filepath"],
        "class_index": "4",
        "class_name": "advertisement_presentation",
        "split": row["split"],
        "source_doc": (
            f"{row['source']}:{row['source_object']}:{row['episode_id']}:{row['step_id']}"
        ),
    }


def main() -> None:
    args = parse_args()
    base_rows = read_rows(args.base_splits)
    manifest_rows = read_rows(args.manifest)
    scores = score_manifest(args.model, manifest_rows)
    for row, score in zip(manifest_rows, scores):
        row["baseline_score"] = float(score)

    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    hard_count = args.train_count - args.easy_diversity_count
    if hard_count < 1 or args.train_count > len(train_rows):
        raise ValueError("Requested train-count/easy-diversity-count is invalid")
    ranked = sorted(train_rows, key=lambda row: row["baseline_score"], reverse=True)
    hard = ranked[:hard_count]
    remaining = ranked[hard_count:]
    easy = random.Random(args.seed).sample(remaining, args.easy_diversity_count)
    selected_ids = {row["filepath"] for row in hard + easy}

    screenshot_rows = []
    for row in manifest_rows:
        if row["split"] == "train" and row["filepath"] not in selected_ids:
            continue
        screenshot_rows.append(as_split_row(row))
    output_rows = base_rows + screenshot_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    by_split = {}
    for split in ("train", "val", "test"):
        split_scores = np.asarray(
            [row["baseline_score"] for row in manifest_rows if row["split"] == split]
        )
        by_split[split] = score_summary(split_scores, args.baseline_threshold)
    selected_scores = np.asarray([row["baseline_score"] for row in hard + easy])
    report = {
        "source": "Google Android-in-the-Wild WebShopping",
        "model": str(args.model),
        "baseline_threshold": args.baseline_threshold,
        "manifest_rows": len(manifest_rows),
        "base_rows": len(base_rows),
        "output_rows": len(output_rows),
        "selected_training_screenshots": len(hard) + len(easy),
        "hardest_training_screenshots": len(hard),
        "random_diversity_screenshots": len(easy),
        "selected_training_scores": score_summary(selected_scores, args.baseline_threshold),
        "baseline_by_screenshot_split": by_split,
        "top_training_examples": [
            {
                "filepath": row["filepath"],
                "score": row["baseline_score"],
                "goal": row["goal"],
                "current_activity": row["current_activity"],
            }
            for row in hard[:50]
        ],
        "split_policy": (
            "AITW episodes are hash-assigned 70/15/15. Only the top-scoring training "
            "candidates plus a random diversity sample are added to training; all "
            "validation and test screenshots remain source-disjoint from training."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "top_training_examples"}, indent=2))
    print(f"Saved splits: {args.output}")
    print(f"Saved mining report: {args.report}")


if __name__ == "__main__":
    main()
