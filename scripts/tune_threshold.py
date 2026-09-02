"""Calibrate a recall-optimized threshold without leaking the test set.

This use case treats a missed bill (false negative) as much more costly than
a false positive (flagging a non-bill for review) -- so instead of the
default 0.5 cutoff, we select a threshold on the validation split, then
report its performance once on the held-out test split.

Usage:
    python scripts/tune_threshold.py --target-recall 0.99
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
IMG_SIZE = 224
METRICS_PATH = REPO_ROOT / "logs" / "deployment_metrics.json"


def load_split(split: str) -> list[dict]:
    with open(SPLITS_PATH) as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def predict_all(model: tf.keras.Model, rows: list[dict], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = [], []
    batch_imgs, batch_labels = [], []

    def flush():
        nonlocal batch_imgs, batch_labels
        if not batch_imgs:
            return
        arr = np.stack(batch_imgs).astype(np.float32)
        probs = model.predict(arr, verbose=0).ravel()
        y_prob.extend(probs.tolist())
        y_true.extend(batch_labels)
        batch_imgs, batch_labels = [], []

    for r in rows:
        img = Image.open(REPO_ROOT / r["filepath"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        batch_imgs.append(np.array(img))
        batch_labels.append(int(r["invoice_label"]))
        if len(batch_imgs) == batch_size:
            flush()
    flush()
    return np.array(y_true), np.array(y_prob)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    eligible = np.where(recall[:-1] >= target_recall)[0]
    if len(eligible) == 0:
        raise ValueError(f"Could not reach target recall {target_recall}")

    # Maximize precision subject to the recall floor; if several operating
    # points tie, use the highest threshold to reject more false positives.
    best_precision = precision[eligible].max()
    best = eligible[np.where(precision[eligible] == best_precision)[0]]
    best_idx = best[np.argmax(thresholds[best])]
    return float(thresholds[best_idx]), float(recall[best_idx]), float(precision[best_idx])


def metrics_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-recall", type=float, default=0.99)
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    val_rows = load_split("val")
    y_val, p_val = predict_all(model, val_rows)
    print(f"n_val={len(y_val)}, n_positive={int(y_val.sum())}")

    print("\nValidation target_recall -> threshold, actual_recall, precision")
    for target in [0.90, 0.95, 0.97, 0.99, 1.0]:
        try:
            threshold, recall, precision = select_threshold(y_val, p_val, target)
        except ValueError:
            print(f"  recall>={target}: unreachable on this validation set")
            continue
        print(
            f"  recall>={target:.2f}: threshold={threshold:.4f}  "
            f"actual_recall={recall:.3f}  precision={precision:.3f}"
        )

    threshold, val_recall, val_precision = select_threshold(y_val, p_val, args.target_recall)
    test_rows = load_split("test")
    y_test, p_test = predict_all(model, test_rows)
    report = {
        "threshold": threshold,
        "target_validation_recall": args.target_recall,
        "validation": metrics_at(y_val, p_val, threshold),
        "test": metrics_at(y_test, p_test, threshold),
        "test_at_0_5": metrics_at(y_test, p_test, 0.5),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"\nSelected on validation: threshold={threshold:.4f} "
        f"(recall={val_recall:.3f}, precision={val_precision:.3f})"
    )
    print("Held-out test metrics at that fixed threshold:")
    for name, value in report["test"].items():
        print(f"  {name}: {value}")
    print(f"Saved deployment metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
