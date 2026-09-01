"""Train a context-aware MobileNetV3-Small router for bill detection.

The router is initialized from the deployed binary classifier and learns three
related tasks:

* bill versus non-bill (the deployment decision score);
* document taxonomy (receipt/invoice and five non-bill families);
* capture context (direct image versus screenshot).

Capture context is deliberately an auxiliary target.  A screenshot may be a
bill or a non-bill, so it never changes the primary label.  Outputs are written
to an experiment directory and never overwrite the deployed app model.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = REPO_ROOT / "data/labeled/verifier_v2_screenshot_balanced_splits.csv"
DEFAULT_INITIAL_MODEL = REPO_ROOT / "models/invoice_classifier.keras"
CLASS_NAMES = (
    "receipt",
    "invoice",
    "structured_document",
    "narrative_document",
    "advertisement_presentation",
    "text_scene",
    "natural_photo",
)
BILL_CLASS_INDICES = (0, 1)
CONTEXT_NAMES = ("direct", "screenshot")
MACOS_DATALESS_FLAG = 0x40000000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--splits-path", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--initial-model", type=Path, default=DEFAULT_INITIAL_MODEL)
    parser.add_argument(
        "--resume-weights",
        type=Path,
        help="Optional compatible smart-router checkpoint used to continue fine-tuning.",
    )
    parser.add_argument("--experiment-name", default="smart_router_v1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs-head", type=int, default=2)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--lr-finetune", type=float, default=8e-6)
    parser.add_argument("--unfreeze-frac", type=float, default=0.28)
    parser.add_argument("--target-validation-recall", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Load the saved best checkpoint, evaluate it, and export without retraining.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def is_screenshot(row: dict) -> bool:
    path = row["filepath"]
    return path.startswith("data/raw/screenshot_") or path.startswith("screenshot:")


def load_splits(data_root: Path, path: Path) -> dict[str, list[dict]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    result = {"train": [], "val": [], "test": []}
    missing: list[str] = []
    offloaded: list[str] = []
    for original in rows:
        row = dict(original)
        row["class_index"] = int(row["class_index"])
        row["bill_label"] = int(row["class_index"] in BILL_CLASS_INDICES)
        row["context_index"] = int(is_screenshot(row))
        absolute = data_root / row["filepath"]
        if not absolute.is_file():
            missing.append(str(absolute))
        elif getattr(absolute.stat(), "st_flags", 0) & MACOS_DATALESS_FLAG:
            offloaded.append(str(absolute))
        result[row["split"]].append(row)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files are missing; first paths:\n" + "\n".join(missing[:10])
        )
    if offloaded:
        raise RuntimeError(
            f"{len(offloaded)} files are iCloud placeholders; first paths:\n"
            + "\n".join(offloaded[:10])
        )
    return result


def describe_splits(splits: dict[str, list[dict]]) -> None:
    for split, rows in splits.items():
        bills = Counter(row["bill_label"] for row in rows)
        classes = Counter(row["class_name"] for row in rows)
        contexts = Counter(CONTEXT_NAMES[row["context_index"]] for row in rows)
        screenshot_bills = sum(
            row["bill_label"] == 1 and row["context_index"] == 1 for row in rows
        )
        print(
            f"{split}: n={len(rows)} bills={dict(bills)} contexts={dict(contexts)} "
            f"screenshot_bills={screenshot_bills} classes={dict(classes)}"
        )


def make_augment() -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.82, 1.04),
                translate_percent=(-0.06, 0.06),
                rotate=(-8, 8),
                shear=(-2, 2),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                p=0.65,
            ),
            A.RandomBrightnessContrast(0.20, 0.20, p=0.50),
            A.RandomGamma(gamma_limit=(75, 130), p=0.20),
            A.GaussNoise(std_range=(0.01, 0.045), p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.10),
            A.ImageCompression(quality_range=(55, 96), p=0.25),
        ]
    )


AUGMENT = make_augment()


def augment_numpy(image: np.ndarray) -> np.ndarray:
    return AUGMENT(image=np.clip(image, 0, 255).astype(np.uint8))["image"]


def balanced_weights(values: np.ndarray, classes: int, maximum: float) -> np.ndarray:
    counts = np.bincount(values, minlength=classes)
    weights = np.zeros(classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(values) / (present.sum() * counts[present])
    return np.minimum(weights[values], maximum)


def sample_weights(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bills = np.array([row["bill_label"] for row in rows], dtype=np.int32)
    classes = np.array([row["class_index"] for row in rows], dtype=np.int32)
    contexts = np.array([row["context_index"] for row in rows], dtype=np.int32)
    bill_weights = balanced_weights(bills, 2, 5.0)
    class_weights = balanced_weights(classes, len(CLASS_NAMES), 4.0)
    context_weights = balanced_weights(contexts, len(CONTEXT_NAMES), 3.0)
    for index, row in enumerate(rows):
        if row["filepath"].startswith("Negative/"):
            bill_weights[index] = min(bill_weights[index] * 2.0, 6.0)
        if row["context_index"] == 1:
            bill_weights[index] = min(bill_weights[index] * 1.15, 6.0)
    return bill_weights, class_weights, context_weights


def build_dataset(
    rows: list[dict],
    data_root: Path,
    image_size: int,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    include_targets: bool = True,
) -> tf.data.Dataset:
    paths = [str(data_root / row["filepath"]) for row in rows]
    bills = np.array([row["bill_label"] for row in rows], dtype=np.float32)
    classes = np.array([row["class_index"] for row in rows], dtype=np.int32)
    contexts = np.array([row["context_index"] for row in rows], dtype=np.int32)
    bill_weights, class_weights, context_weights = sample_weights(rows)
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, bills, classes, contexts, bill_weights, class_weights, context_weights)
    )

    def load(path, bill, document, context, bill_w, document_w, context_w):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(
            tf.cast(image, tf.float32), (image_size, image_size), antialias=True
        )
        image.set_shape((image_size, image_size, 3))
        return image, bill, document, context, bill_w, document_w, context_w

    dataset = dataset.map(load, num_parallel_calls=4)
    if shuffle:
        dataset = dataset.shuffle(min(3000, len(rows)), seed=0, reshuffle_each_iteration=True)
    if augment:
        def apply_augment(image, bill, document, context, bill_w, document_w, context_w):
            image = tf.numpy_function(augment_numpy, [image], tf.uint8)
            image.set_shape((image_size, image_size, 3))
            return image, bill, document, context, bill_w, document_w, context_w

        dataset = dataset.map(apply_augment, num_parallel_calls=4)

    if include_targets:
        def prepare(image, bill, document, context, bill_w, document_w, context_w):
            targets = {
                "bill_prob": tf.reshape(bill, (1,)),
                "document_class": tf.one_hot(document, len(CLASS_NAMES)),
                "capture_context": tf.one_hot(context, len(CONTEXT_NAMES)),
            }
            weights = {
                "bill_prob": bill_w,
                "document_class": document_w,
                "capture_context": context_w,
            }
            return tf.cast(image, tf.float32), targets, weights

        dataset = dataset.map(prepare, num_parallel_calls=4)
    else:
        dataset = dataset.map(
            lambda image, bill, document, context, bill_w, document_w, context_w: tf.cast(
                image, tf.float32
            ),
            num_parallel_calls=4,
        )
    return dataset.batch(batch_size).prefetch(1)


def build_model(initial_model_path: Path) -> tuple[tf.keras.Model, int]:
    legacy = tf.keras.models.load_model(initial_model_path, compile=False)
    feature_layer = legacy.get_layer("avg_pool")
    old_bill_head = legacy.get_layer("invoice_prob")
    dropout_rate = float(legacy.get_layer("dropout").rate)
    features = tf.keras.layers.Dropout(dropout_rate, name="smart_dropout")(
        feature_layer.output
    )
    bill_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="bill_prob")
    bill_prob = bill_layer(features)
    document_class = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="document_class"
    )(features)
    capture_context = tf.keras.layers.Dense(
        len(CONTEXT_NAMES), activation="softmax", name="capture_context"
    )(features)
    model = tf.keras.Model(
        legacy.input,
        {
            "bill_prob": bill_prob,
            "document_class": document_class,
            "capture_context": capture_context,
        },
        name="smart_bill_router",
    )
    bill_layer.set_weights(old_bill_head.get_weights())
    feature_index = model.layers.index(feature_layer)
    for layer in model.layers[: feature_index + 1]:
        layer.trainable = False
    return model, feature_index


def set_finetune_layers(model: tf.keras.Model, feature_index: int, fraction: float) -> int:
    first = int(feature_index * (1.0 - fraction))
    count = 0
    for index, layer in enumerate(model.layers[: feature_index + 1]):
        layer.trainable = index >= first and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
        count += int(layer.trainable)
    return count


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-5, clipnorm=1.0
        ),
        loss={
            "bill_prob": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "document_class": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
            "capture_context": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
        },
        loss_weights={"bill_prob": 0.72, "document_class": 0.20, "capture_context": 0.08},
        metrics={
            "bill_prob": [tf.keras.metrics.AUC(name="auc")],
            "document_class": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
            "capture_context": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        },
    )


def unpack_predictions(predictions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(predictions, dict):
        return (
            predictions["bill_prob"].ravel(),
            predictions["document_class"],
            predictions["capture_context"],
        )
    if isinstance(predictions, (list, tuple)):
        by_name = {}
        for values in predictions:
            width = int(values.shape[-1])
            if width == 1:
                by_name["bill_prob"] = values
            elif width == len(CLASS_NAMES):
                by_name["document_class"] = values
            elif width == len(CONTEXT_NAMES):
                by_name["capture_context"] = values
        if set(by_name) != {"bill_prob", "document_class", "capture_context"}:
            raise ValueError(
                f"Could not identify output tensors from shapes: "
                f"{[tuple(value.shape) for value in predictions]}"
            )
        return (
            by_name["bill_prob"].ravel(),
            by_name["document_class"],
            by_name["capture_context"],
        )
    raise TypeError(f"Unexpected prediction type: {type(predictions)}")


def threshold_at_recall(true: np.ndarray, scores: np.ndarray, target: float) -> float:
    positives = np.sort(scores[true == 1])
    misses = int(np.floor((1.0 - target) * len(positives) + 1e-12))
    misses = min(max(misses, 0), len(positives) - 1)
    return float(np.nextafter(positives[misses], -np.inf))


def binary_metrics(true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    negative = true == 0
    return {
        "n": int(len(true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted)),
        "fpr": float(predicted[negative].mean()),
        "auc": float(roc_auc_score(true, scores)),
    }


class RecallCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, images, true, target, path: Path) -> None:
        super().__init__()
        self.images = images
        self.true = true
        self.target = target
        self.path = path
        self.best = (np.inf, -np.inf)
        self.best_epoch = -1
        self.records: list[dict] = []

    def on_epoch_end(self, epoch, logs=None):
        scores, _, _ = unpack_predictions(self.model.predict(self.images, verbose=0))
        threshold = threshold_at_recall(self.true, scores, self.target)
        metrics = binary_metrics(self.true, scores, threshold)
        candidate = (metrics["fpr"], -metrics["auc"])
        saved = candidate < self.best
        if saved:
            self.best = candidate
            self.best_epoch = int(epoch)
            self.model.save_weights(self.path)
        self.records.append({"epoch": int(epoch), **metrics})
        print(
            f"\nrecall-constrained val: recall={metrics['recall']:.4f} "
            f"fpr={metrics['fpr']:.4f} threshold={threshold:.6f} "
            f"auc={metrics['auc']:.4f}" + (" [saved]" if saved else "")
        )


def evaluate(model, images, rows, target_recall: float) -> dict:
    scores, classes, contexts = unpack_predictions(model.predict(images, verbose=0))
    true = np.array([row["bill_label"] for row in rows], dtype=np.int32)
    threshold = threshold_at_recall(true, scores, target_recall)
    result = binary_metrics(true, scores, threshold)
    result["document_accuracy"] = float(
        accuracy_score([row["class_index"] for row in rows], classes.argmax(axis=1))
    )
    result["context_accuracy"] = float(
        accuracy_score([row["context_index"] for row in rows], contexts.argmax(axis=1))
    )
    result["scores"] = scores
    return result


def export_tflite(
    model: tf.keras.Model, path: Path, dynamic_range_quantization: bool = True
) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="smart-router-"))
    saved_model = temp_root / "saved_model"
    try:
        # Direct Keras conversion crashes in TF 2.16's MLIR variable freezer
        # for this multi-output graph.  Explicit SavedModel export uses the
        # stable conversion path already used by the production exporters.
        model.export(str(saved_model))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        if dynamic_range_quantization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        path.write_bytes(converter.convert())
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    data_root = args.data_root.resolve()
    splits = load_splits(data_root, args.splits_path.resolve())
    describe_splits(splits)
    if args.dry_run:
        return

    model_dir = REPO_ROOT / "models/experiments" / args.experiment_name
    log_dir = REPO_ROOT / "logs/experiments" / args.experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best.weights.h5"

    train = build_dataset(
        splits["train"], data_root, args.image_size, args.batch_size, True, True
    )
    validation = build_dataset(
        splits["val"], data_root, args.image_size, args.batch_size, False, False
    )
    validation_images = build_dataset(
        splits["val"], data_root, args.image_size, args.batch_size, False, False, False
    )
    test_images = build_dataset(
        splits["test"], data_root, args.image_size, args.batch_size, False, False, False
    )
    validation_true = np.array(
        [row["bill_label"] for row in splits["val"]], dtype=np.int32
    )

    model, feature_index = build_model(args.initial_model.resolve())
    if args.resume_weights is not None:
        print(f"Loading smart-router weights from {args.resume_weights}")
        model.load_weights(args.resume_weights)
    checkpoint = RecallCheckpoint(
        validation_images, validation_true, args.target_validation_recall, checkpoint_path
    )
    histories = {}
    if args.finalize_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint to finalize: {checkpoint_path}")
    else:
        if args.epochs_head > 0:
            compile_model(model, args.lr_head)
            histories["head"] = model.fit(
                train,
                validation_data=validation,
                epochs=args.epochs_head,
                callbacks=[checkpoint],
                verbose=2,
            ).history
        else:
            histories["head"] = {}
        trainable = set_finetune_layers(model, feature_index, args.unfreeze_frac)
        print(f"Fine-tuning {trainable} high-level feature layers; BatchNorm remains frozen")
        if args.epochs_finetune > 0:
            compile_model(model, args.lr_finetune)
            histories["finetune"] = model.fit(
                train,
                validation_data=validation,
                epochs=args.epochs_finetune,
                callbacks=[checkpoint],
                verbose=2,
            ).history
        else:
            histories["finetune"] = {}

    model.load_weights(checkpoint_path)
    validation_result = evaluate(
        model, validation_images, splits["val"], args.target_validation_recall
    )
    threshold = validation_result["threshold"]
    test_scores, test_classes, test_contexts = unpack_predictions(
        model.predict(test_images, verbose=0)
    )
    test_true = np.array([row["bill_label"] for row in splits["test"]], dtype=np.int32)
    test_result = binary_metrics(test_true, test_scores, threshold)
    test_result["document_accuracy"] = float(
        accuracy_score(
            [row["class_index"] for row in splits["test"]], test_classes.argmax(axis=1)
        )
    )
    test_result["context_accuracy"] = float(
        accuracy_score(
            [row["context_index"] for row in splits["test"]], test_contexts.argmax(axis=1)
        )
    )
    validation_result.pop("scores")

    keras_path = model_dir / "model.keras"
    tflite_path = model_dir / "model.tflite"
    model.save(keras_path)
    export_tflite(model, tflite_path)
    metrics = {
        "architecture": "MobileNetV3Small smart multi-task router",
        "outputs": ["bill_prob", "document_class", "capture_context"],
        "class_names": list(CLASS_NAMES),
        "context_names": list(CONTEXT_NAMES),
        "threshold": threshold,
        "best_epoch": checkpoint.best_epoch if not args.finalize_only else None,
        "validation": validation_result,
        "test": test_result,
        "tflite_size_bytes": tflite_path.stat().st_size,
        "training": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "recall_constrained_history": checkpoint.records,
    }
    (log_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    serializable_history = {
        phase: {key: [float(item) for item in values] for key, values in history.items()}
        for phase, history in histories.items()
    }
    (log_dir / "history.json").write_text(json.dumps(serializable_history, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved {keras_path} and {tflite_path}")


if __name__ == "__main__":
    main()
