"""Calibrate and evaluate the smart Small-router -> Large-verifier pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_auc_score

import calibrate_verifier_v2 as large_calibration


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = REPO_ROOT / "data/labeled/verifier_v2_screenshot_balanced_splits.csv"
DEFAULT_ROUTER = REPO_ROOT / "models/experiments/smart_router_v1/model.tflite"
DEFAULT_LARGE = REPO_ROOT / "models/experiments/verifier_v2_screenshot_balanced/model.tflite"
DEFAULT_LARGE_CALIBRATION = (
    REPO_ROOT / "logs/experiments/verifier_v2_screenshot_balanced/calibration_balanced.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs/experiments/smart_router_v1/pipeline_evaluation.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
DATALESS_FLAG = 0x40000000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--large", type=Path, default=DEFAULT_LARGE)
    parser.add_argument("--large-calibration", type=Path, default=DEFAULT_LARGE_CALIBRATION)
    parser.add_argument("--target-validation-recall", type=float, default=0.99)
    parser.add_argument("--router-threshold", type=float, default=0.2398375570774078)
    parser.add_argument("--grid-steps", type=int, default=101)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def interpreter(path: Path) -> tf.lite.Interpreter:
    result = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    result.allocate_tensors()
    return result


def image_array(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(
            opened.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )


def run_outputs(model: tf.lite.Interpreter, image: np.ndarray) -> list[np.ndarray]:
    detail = model.get_input_details()[0]
    model.set_tensor(detail["index"], image[None, ...])
    model.invoke()
    return [
        model.get_tensor(output["index"]).ravel().astype(float)
        for output in model.get_output_details()
    ]


def router_score(model: tf.lite.Interpreter, image: np.ndarray) -> float:
    values = [output for output in run_outputs(model, image) if output.size == 1]
    if len(values) != 1:
        raise ValueError(f"Expected one scalar router output, got {[x.shape for x in values]}")
    return float(values[0][0])


def large_score(model: tf.lite.Interpreter, image: np.ndarray, alpha: float) -> float:
    outputs = run_outputs(model, image)
    direct = [output for output in outputs if output.size == 1]
    classes = [output for output in outputs if output.size == 7]
    if len(direct) != 1 or len(classes) != 1:
        raise ValueError(f"Unexpected Large outputs: {[x.shape for x in outputs]}")
    class_bill = float(classes[0][[0, 1]].sum())
    return float(
        large_calibration.ensemble_score(
            np.asarray([direct[0][0]]), np.asarray([class_bill]), alpha
        )[0]
    )


def load_rows(path: Path, split: str) -> list[dict]:
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == split]
    return rows


def score_rows(
    rows: list[dict], router: tf.lite.Interpreter, large: tf.lite.Interpreter, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true, router_scores, large_scores = [], [], []
    for index, row in enumerate(rows, start=1):
        path = REPO_ROOT / row["filepath"]
        if getattr(path.stat(), "st_flags", 0) & DATALESS_FLAG:
            raise RuntimeError(f"Split contains an iCloud placeholder: {path}")
        image = image_array(path)
        true.append(int(int(row["class_index"]) in (0, 1)))
        router_scores.append(router_score(router, image))
        large_scores.append(large_score(large, image, alpha))
        if index % 250 == 0:
            print(f"Scored {index}/{len(rows)} {row['split']} images")
    return np.asarray(true), np.asarray(router_scores), np.asarray(large_scores)


def metrics(true: np.ndarray, predicted: np.ndarray, scores: np.ndarray | None = None) -> dict:
    positive = true == 1
    negative = ~positive
    selected = predicted == 1
    result = {
        "n": int(len(true)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "recall": float(predicted[positive].mean()),
        "fpr": float(predicted[negative].mean()),
        "precision": float(true[selected].mean()) if selected.any() else 0.0,
        "accuracy": float((predicted == true).mean()),
        "tp": int((selected & positive).sum()),
        "fn": int(((~selected) & positive).sum()),
        "fp": int((selected & negative).sum()),
        "tn": int(((~selected) & negative).sum()),
    }
    if scores is not None:
        result["auc"] = float(roc_auc_score(true, scores))
    return result


def route(
    router_scores: np.ndarray,
    large_scores: np.ndarray,
    low: float,
    high: float,
    large_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    uncertain = (router_scores >= low) & (router_scores < high)
    predicted = np.zeros(len(router_scores), dtype=np.int32)
    predicted[router_scores >= high] = 1
    predicted[uncertain] = (large_scores[uncertain] >= large_threshold).astype(np.int32)
    return predicted, uncertain


def candidate_thresholds(values: np.ndarray, steps: int) -> np.ndarray:
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, steps))
    return np.unique(np.concatenate(([-np.inf], quantiles, [np.inf])))


def calibrate_routes(
    true: np.ndarray,
    router_scores: np.ndarray,
    large_scores: np.ndarray,
    large_threshold: float,
    target_recall: float,
    steps: int,
) -> dict:
    thresholds = candidate_thresholds(router_scores, steps)
    candidates = []
    for low_index, low in enumerate(thresholds):
        for high in thresholds[low_index:]:
            predicted, uncertain = route(
                router_scores, large_scores, float(low), float(high), large_threshold
            )
            result = metrics(true, predicted)
            result.update(
                {
                    "low_threshold": float(low),
                    "high_threshold": float(high),
                    "route_rate": float(uncertain.mean()),
                }
            )
            candidates.append(result)

    recall_eligible = [item for item in candidates if item["recall"] >= target_recall]
    if not recall_eligible:
        raise RuntimeError(f"No route reaches validation recall {target_recall}")
    accuracy_first = min(
        recall_eligible,
        key=lambda item: (item["fpr"], item["route_rate"], -item["precision"]),
    )
    speed_profiles = {}
    for cap in (0.10, 0.20, 0.30, 0.50):
        within_cap = [item for item in candidates if item["route_rate"] <= cap + 1e-12]
        eligible = [item for item in within_cap if item["recall"] >= target_recall]
        if eligible:
            chosen = min(eligible, key=lambda item: (item["fpr"], -item["recall"]))
        else:
            chosen = max(within_cap, key=lambda item: (item["recall"], -item["fpr"]))
        speed_profiles[f"route_cap_{int(cap * 100)}pct"] = chosen
    return {"recommended_accuracy_first": accuracy_first, "speed_profiles": speed_profiles}


def evaluate_route(
    true: np.ndarray,
    router_scores: np.ndarray,
    large_scores: np.ndarray,
    point: dict,
    large_threshold: float,
) -> dict:
    predicted, uncertain = route(
        router_scores,
        large_scores,
        point["low_threshold"],
        point["high_threshold"],
        large_threshold,
    )
    return metrics(true, predicted) | {"route_rate": float(uncertain.mean())}


def group_metrics(
    rows: list[dict], true: np.ndarray, predicted: np.ndarray
) -> dict:
    screenshot = np.array(
        [row["filepath"].startswith("data/raw/screenshot_") for row in rows]
    )
    groups = {
        "direct_bill": (true == 1) & ~screenshot,
        "screenshot_bill": (true == 1) & screenshot,
        "direct_nonbill": (true == 0) & ~screenshot,
        "screenshot_nonbill": (true == 0) & screenshot,
    }
    result = {}
    for name, mask in groups.items():
        expected = 1 if name.endswith("bill") and not name.endswith("nonbill") else 0
        result[name] = {
            "n": int(mask.sum()),
            "correct": int((predicted[mask] == expected).sum()),
            "rate": float((predicted[mask] == expected).mean()),
        }
    return result


def folder_paths(path: Path) -> list[Path]:
    return [
        item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]


def score_folder(
    path: Path, router: tf.lite.Interpreter, large: tf.lite.Interpreter, alpha: float
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    names, small, verifier, skipped = [], [], [], []
    for item in folder_paths(path):
        if getattr(item.stat(), "st_flags", 0) & DATALESS_FLAG:
            skipped.append(item.name)
            continue
        image = image_array(item)
        names.append(item.name)
        small.append(router_score(router, image))
        verifier.append(large_score(large, image, alpha))
    return names, np.asarray(small), np.asarray(verifier), skipped


def benchmark(
    router: tf.lite.Interpreter, large: tf.lite.Interpreter, rows: list[dict], alpha: float
) -> dict:
    images = [image_array(REPO_ROOT / row["filepath"]) for row in rows[:100]]
    for image in images[:5]:
        router_score(router, image)
        large_score(large, image, alpha)
    result = {}
    for name, function in (
        ("router", lambda image: router_score(router, image)),
        ("large", lambda image: large_score(large, image, alpha)),
    ):
        durations = []
        for image in images:
            started = time.perf_counter()
            function(image)
            durations.append((time.perf_counter() - started) * 1000)
        values = np.asarray(durations)
        result[name] = {
            "mean_ms": float(values.mean()),
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    return result


def main() -> None:
    args = parse_args()
    router_model = interpreter(args.router)
    large_model = interpreter(args.large)
    large_report = json.loads(args.large_calibration.read_text())
    alpha = float(large_report["calibration"]["alpha_direct_head"])
    large_threshold = float(large_report["calibration"]["threshold"])
    validation_rows = load_rows(args.splits_path, "val")
    test_rows = load_rows(args.splits_path, "test")
    validation = score_rows(validation_rows, router_model, large_model, alpha)
    test = score_rows(test_rows, router_model, large_model, alpha)
    calibration = calibrate_routes(
        *validation,
        large_threshold,
        args.target_validation_recall,
        args.grid_steps,
    )
    recommended = calibration["recommended_accuracy_first"]
    test_predicted, test_uncertain = route(
        test[1],
        test[2],
        recommended["low_threshold"],
        recommended["high_threshold"],
        large_threshold,
    )
    folder_results = {}
    for folder, expected in (("TestPhotos", 1), ("Negative", 0), ("N2", 0)):
        names, small, verifier, skipped = score_folder(
            REPO_ROOT / folder, router_model, large_model, alpha
        )
        predicted, uncertain = route(
            small,
            verifier,
            recommended["low_threshold"],
            recommended["high_threshold"],
            large_threshold,
        )
        folder_results[folder] = {
            "evaluated": len(names),
            "correct": int((predicted == expected).sum()),
            "route_rate": float(uncertain.mean()) if len(uncertain) else 0.0,
            "skipped_offloaded": skipped,
            "details": [
                {
                    "file": name,
                    "router_score": float(small[index]),
                    "large_score": float(verifier[index]),
                    "routed": bool(uncertain[index]),
                    "predicted_bill": bool(predicted[index]),
                }
                for index, name in enumerate(names)
            ],
        }

    benchmark_result = benchmark(router_model, large_model, test_rows, alpha)
    average_ms = benchmark_result["router"]["mean_ms"] + float(
        test_uncertain.mean()
    ) * benchmark_result["large"]["mean_ms"]
    result = {
        "architecture": "context-aware MobileNetV3Small triage router with balanced MobileNetV3Large fallback",
        "router_model": str(args.router),
        "large_model": str(args.large),
        "large_alpha": alpha,
        "large_threshold": large_threshold,
        "target_validation_recall": args.target_validation_recall,
        "calibration": calibration,
        "test": {
            "smart_pipeline": metrics(test[0], test_predicted)
            | {"route_rate": float(test_uncertain.mean())},
            "smart_router_standalone": metrics(
                test[0], (test[1] >= args.router_threshold).astype(np.int32), test[1]
            ),
            "large_standalone": metrics(
                test[0], (test[2] >= large_threshold).astype(np.int32), test[2]
            ),
            "groups": group_metrics(test_rows, test[0], test_predicted),
            "speed_profiles": {
                name: evaluate_route(test[0], test[1], test[2], point, large_threshold)
                for name, point in calibration["speed_profiles"].items()
            },
        },
        "folders": folder_results,
        "benchmark": benchmark_result | {"smart_pipeline_mean_ms": average_ms},
        "model_bytes": {
            "router": args.router.stat().st_size,
            "large": args.large.stat().st_size,
            "combined": args.router.stat().st_size + args.large.stat().st_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
