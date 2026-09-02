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

from _common import check_split_matches, require_file

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
IMG_SIZE = 224
# Calibration writes here. It deliberately does NOT write
# logs/deployment_metrics.json: that file is the deployment record, holding
# the operating point the app actually ships plus the app_regression counts
# that were measured by hand. Calibration output has a different schema
# entirely, so writing it there replaced the deployed threshold and silently
# destroyed the provenance. Promoting a calibrated threshold is now an
# explicit, separate step - see --promote-threshold.
CALIBRATION_PATH = REPO_ROOT / "logs" / "calibration.json"
DEPLOYMENT_METRICS_PATH = REPO_ROOT / "logs" / "deployment_metrics.json"
FINAL_METRICS_PATH = REPO_ROOT / "logs" / "final_metrics.json"


def recorded_run() -> dict | None:
    """The run fingerprint saved when the current model was trained."""
    if not FINAL_METRICS_PATH.exists():
        return None
    with open(FINAL_METRICS_PATH) as f:
        return json.load(f).get("run")


def deployed_threshold() -> float | None:
    """The operating point the app currently ships, if it has been recorded."""
    if not DEPLOYMENT_METRICS_PATH.exists():
        return None
    with open(DEPLOYMENT_METRICS_PATH) as f:
        return json.load(f).get("threshold")


def promote_threshold(threshold: float) -> None:
    """Adopt a calibrated threshold, preserving the rest of the record.

    Reads, updates one key, writes back - rather than replacing the file -
    so `selection`, `tflite_test` and the hand-measured `app_regression`
    counts survive. They are stamped as stale, because they were measured at
    the previous threshold and no longer describe this one.
    """
    record = {}
    if DEPLOYMENT_METRICS_PATH.exists():
        with open(DEPLOYMENT_METRICS_PATH) as f:
            record = json.load(f)
    previous = record.get("threshold")
    record["threshold"] = threshold
    record["selection"] = (
        f"promoted from calibration (was {previous!r}); "
        "tflite_test and app_regression below predate this and must be re-run"
    )
    DEPLOYMENT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEPLOYMENT_METRICS_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Promoted threshold {previous!r} -> {threshold:.4f} in {DEPLOYMENT_METRICS_PATH}")


def load_split(split: str) -> list[dict]:
    with open(require_file(SPLITS_PATH, "python scripts/train.py")) as f:
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
    parser.add_argument(
        "--allow-split-mismatch",
        action="store_true",
        help="Evaluate even when splits.csv no longer matches the split the model was trained on. The resulting metrics describe a different split from the one the model saw.",
    )
    parser.add_argument(
        "--promote-threshold",
        action="store_true",
        help=(
            "Also update the threshold in logs/deployment_metrics.json. Every "
            "other field in that file is preserved, but the recorded "
            "tflite_test and app_regression results were measured at the old "
            "threshold and will no longer describe the new one - re-run the "
            "export and the app regression set after promoting."
        ),
    )
    args = parser.parse_args()

    check_split_matches(recorded_run(), SPLITS_PATH, allow_mismatch=args.allow_split_mismatch)

    model = tf.keras.models.load_model(require_file(MODEL_PATH, "python scripts/train.py"))
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
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"\nSelected on validation: threshold={threshold:.4f} "
        f"(recall={val_recall:.3f}, precision={val_precision:.3f})"
    )
    print("Held-out test metrics at that fixed threshold:")
    for name, value in report["test"].items():
        print(f"  {name}: {value}")
    print(f"Saved calibration to {CALIBRATION_PATH}")

    if args.promote_threshold:
        promote_threshold(threshold)
    else:
        print(
            f"\nThe app still ships {deployed_threshold()!r}. To adopt "
            f"{threshold:.4f}, re-run with --promote-threshold."
        )


if __name__ == "__main__":
    main()
