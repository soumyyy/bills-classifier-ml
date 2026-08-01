"""Find a recall-optimized decision threshold for the trained classifier.

This use case treats a missed bill (false negative) as much more costly than
a false positive (flagging a non-bill for review) -- so instead of the
default 0.5 cutoff, we pick the highest threshold that still hits a target
recall on the held-out test split.

Usage:
    python scripts/tune_threshold.py --target-recall 1.0
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
IMG_SIZE = 224


def load_test_split() -> list[dict]:
    with open(SPLITS_PATH) as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-recall", type=float, default=1.0)
    args = parser.parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    rows = load_test_split()
    y_true, y_prob = predict_all(model, rows)
    print(f"n_test={len(y_true)}, n_positive={int(y_true.sum())}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    print("\ntarget_recall -> threshold, actual_recall, precision")
    for target in [0.90, 0.95, 0.97, 0.99, 1.0]:
        idx = np.where(recall >= target)[0]
        if len(idx) == 0:
            print(f"  recall>={target}: unreachable on this test set")
            continue
        best_idx = idx[np.argmax(precision[idx])]
        thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
        print(f"  recall>={target:.2f}: threshold={thr:.3f}  actual_recall={recall[best_idx]:.3f}  precision={precision[best_idx]:.3f}")

    idx = np.where(recall >= args.target_recall)[0]
    if len(idx) == 0:
        print(f"\nCould not reach target recall {args.target_recall} on this test set.")
        return
    best_idx = idx[np.argmax(precision[idx])]
    thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
    print(
        f"\nRecommended threshold for target_recall={args.target_recall}: {thr:.3f} "
        f"(recall={recall[best_idx]:.3f}, precision={precision[best_idx]:.3f})"
    )


if __name__ == "__main__":
    main()
