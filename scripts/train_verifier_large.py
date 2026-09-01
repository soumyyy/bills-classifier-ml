"""Train a seven-class MobileNetV3-Large bill verifier.

The existing MobileNetV3-Small binary classifier remains the high-recall
gate. This model learns *why* a candidate is or is not a bill so that menus,
tables, reports, advertisements, and text-heavy scenes are not collapsed into
one overly broad negative class.

Usage:
    python scripts/train_verifier_large.py --dry-run
    python scripts/train_verifier_large.py --epochs-head 8 --epochs-finetune 12
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

import train as binary_train

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "verifier_large_best.weights.h5"
MODEL_PATH = REPO_ROOT / "models" / "invoice_verifier_large.keras"
METRICS_PATH = REPO_ROOT / "logs" / "verifier_large_metrics.json"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "verifier_large_splits.csv"
IMAGENET_WEIGHTS = REPO_ROOT / "models" / "checkpoints" / "weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5"

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

STRUCTURED_LABELS = {
    "budget",
    "file folder",
    "form",
    "questionnaire",
    "specification",
}
NARRATIVE_LABELS = {
    "email",
    "handwritten",
    "letter",
    "memo",
    "news article",
    "resume",
    "scientific publication",
    "scientific report",
}
ADVERTISEMENT_LABELS = {"advertisement", "presentation"}


def negative_class(predicted_class: str) -> str:
    label = predicted_class.strip().lower().replace("_", " ")
    if label in STRUCTURED_LABELS:
        return "structured_document"
    if label in NARRATIVE_LABELS:
        return "narrative_document"
    if label in ADVERTISEMENT_LABELS:
        return "advertisement_presentation"
    return "structured_document"


def class_name_for(row: dict, by_path: dict[str, dict]) -> str:
    source = row.get("source", "")
    source_doc = row.get("source_doc", "")

    if row["invoice_label"] == "1":
        if source in {"cord_receipts", "sroie_receipts"}:
            return "receipt"
        if source == "synthetic_hard_pos":
            return "receipt" if "cord_receipts" in source_doc or "sroie_receipts" in source_doc else "invoice"
        return "invoice"

    if source == "coco_photos":
        return "natural_photo"
    if source == "hiertext_text_scene":
        return "text_scene"
    if source == "target_hard_negative":
        return "structured_document"
    if source.startswith("rvl_targeted_"):
        return negative_class(source.removeprefix("rvl_targeted_"))
    if source in {"synthetic_hard_neg", "synthetic_hard_pos"}:
        parent = by_path.get(source_doc)
        if parent is not None:
            return class_name_for(parent, by_path)
    if source == "rvlcdip_other":
        return negative_class(row.get("predicted_class", ""))
    return negative_class(row.get("predicted_class", ""))


def load_rows() -> list[dict]:
    rows = binary_train.load_manifest()
    by_path = {row["filepath"]: row for row in rows}
    labeled = []
    for row in rows:
        class_name = class_name_for(row, by_path)
        item = dict(row)
        item["class_name"] = class_name
        item["class_index"] = CLASS_NAMES.index(class_name)
        labeled.append(item)
    return labeled


def make_splits(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    forced = [row for row in rows if row.get("forced_split")]
    regular = [row for row in rows if not row.get("forced_split")]
    labels = [row["class_index"] for row in regular]
    groups = [row.get("source_doc") or row["filepath"] for row in regular]

    first = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(first.split(regular, labels, groups))
    train_rows = [regular[i] for i in train_idx]
    temp_rows = [regular[i] for i in temp_idx]
    temp_labels = [row["class_index"] for row in temp_rows]
    temp_groups = [row.get("source_doc") or row["filepath"] for row in temp_rows]

    second = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_idx, test_idx = next(second.split(temp_rows, temp_labels, temp_groups))
    result = {
        "train": train_rows,
        "val": [temp_rows[i] for i in val_idx],
        "test": [temp_rows[i] for i in test_idx],
    }
    for row in forced:
        split = row["forced_split"]
        if split not in result:
            raise ValueError(f"Invalid forced split {split!r}: {row['filepath']}")
        result[split].append(row)

    seen: dict[str, str] = {}
    for split, split_rows in result.items():
        for row in split_rows:
            group = row.get("source_doc") or row["filepath"]
            previous = seen.get(group)
            if previous is not None and previous != split:
                raise ValueError(f"Leakage: {group} appears in {previous} and {split}")
            seen[group] = split

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_PATH, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["filepath", "class_index", "class_name", "split", "source_doc"])
        for split, split_rows in result.items():
            for row in split_rows:
                writer.writerow(
                    [
                        row["filepath"],
                        row["class_index"],
                        row["class_name"],
                        split,
                        row.get("source_doc") or row["filepath"],
                    ]
                )
    return result


def build_dataset(rows: list[dict], batch_size: int, augment: bool, shuffle: bool) -> tf.data.Dataset:
    paths = [str(REPO_ROOT / row["filepath"]) for row in rows]
    labels = [row["class_index"] for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.cast(tf.image.resize(image, (binary_train.IMG_SIZE, binary_train.IMG_SIZE)), tf.uint8)
        return image, label

    dataset = dataset.map(load, num_parallel_calls=4)
    if shuffle:
        dataset = dataset.shuffle(min(len(rows), 2000), seed=0, reshuffle_each_iteration=True)
    if augment:
        def apply_augmentation(image, label):
            image = tf.numpy_function(binary_train._albumentations_augment, [image], tf.uint8)
            image.set_shape((binary_train.IMG_SIZE, binary_train.IMG_SIZE, 3))
            return image, label

        dataset = dataset.map(apply_augmentation, num_parallel_calls=4)
    dataset = dataset.map(
        lambda image, label: (tf.cast(image, tf.float32), label),
        num_parallel_calls=4,
    )
    return dataset.batch(batch_size).prefetch(1)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    cached = IMAGENET_WEIGHTS
    if not cached.exists():
        cached = Path.home() / ".keras" / "models" / IMAGENET_WEIGHTS.name
    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape=(binary_train.IMG_SIZE, binary_train.IMG_SIZE, 3),
        include_top=False,
        weights=None if cached.exists() else "imagenet",
        include_preprocessing=True,
        pooling="avg",
    )
    if cached.exists():
        backbone.load_weights(cached)
    backbone.trainable = False
    features = tf.keras.layers.Dropout(0.25)(backbone.output)
    output = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="document_class")(features)
    return tf.keras.Model(backbone.input, output, name="invoice_verifier_large"), backbone


def class_weights(rows: list[dict]) -> dict[int, float]:
    counts = Counter(row["class_index"] for row in rows)
    total = len(rows)
    return {index: total / (len(CLASS_NAMES) * counts[index]) for index in range(len(CLASS_NAMES))}


def callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(CHECKPOINT_PATH),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=4, restore_best_weights=True
        ),
    ]


def predict(model: tf.keras.Model, rows: list[dict], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = build_dataset(rows, batch_size, augment=False, shuffle=False)
    true_classes = np.array([row["class_index"] for row in rows])
    probabilities = model.predict(dataset, verbose=0)
    return true_classes, probabilities


def threshold_for_recall(true_classes: np.ndarray, probabilities: np.ndarray, recall: float) -> float:
    bill_true = np.isin(true_classes, BILL_CLASS_INDICES)
    bill_scores = probabilities[:, BILL_CLASS_INDICES].sum(axis=1)
    positive_scores = np.sort(bill_scores[bill_true])
    allowed_misses = int(np.floor((1.0 - recall) * len(positive_scores) + 1e-9))
    return float(np.nextafter(positive_scores[allowed_misses], -np.inf))


def metrics_for(
    true_classes: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted_classes = probabilities.argmax(axis=1)
    bill_true = np.isin(true_classes, BILL_CLASS_INDICES).astype(int)
    bill_scores = probabilities[:, BILL_CLASS_INDICES].sum(axis=1)
    bill_predicted = (bill_scores >= threshold).astype(int)
    return {
        "n": int(len(true_classes)),
        "multiclass_accuracy": float(accuracy_score(true_classes, predicted_classes)),
        "multiclass_macro_f1": float(f1_score(true_classes, predicted_classes, average="macro")),
        "bill_accuracy": float(accuracy_score(bill_true, bill_predicted)),
        "bill_precision": float(precision_score(bill_true, bill_predicted)),
        "bill_recall": float(recall_score(bill_true, bill_predicted)),
        "bill_auc": float(roc_auc_score(bill_true, bill_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=12)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--unfreeze-frac", type=float, default=0.25)
    parser.add_argument("--target-validation-recall", type=float, default=0.99)
    parser.add_argument("--target-hard-negative-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    binary_train.set_seeds(args.seed)
    binary_train.configure_memory_growth()
    rows = load_rows()
    splits = make_splits(rows, args.seed)
    for split, split_rows in splits.items():
        print(f"{split}: {len(split_rows)} {dict(Counter(row['class_name'] for row in split_rows))}")
    if args.dry_run:
        return

    original_train_rows = splits["train"]
    target_rows = [row for row in original_train_rows if row.get("source") == "target_hard_negative"]
    train_rows = original_train_rows + target_rows * (args.target_hard_negative_repeat - 1)
    weights = class_weights(train_rows)
    print(f"Effective target hard negatives: {len(target_rows) * args.target_hard_negative_repeat}")
    print(f"Class weights: {weights}")

    train_dataset = build_dataset(train_rows, args.batch_size, augment=True, shuffle=True)
    validation_dataset = build_dataset(splits["val"], args.batch_size, augment=False, shuffle=False)
    model, backbone = build_model()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    print("\n=== Phase 1: train multiclass head ===")
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs_head,
        class_weight=weights,
        callbacks=callbacks(),
    )

    backbone.trainable = True
    freeze_before = int(len(backbone.layers) * (1.0 - args.unfreeze_frac))
    trainable_layers = 0
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= freeze_before and not isinstance(layer, tf.keras.layers.BatchNormalization)
        trainable_layers += int(layer.trainable)
    print(f"\n=== Phase 2: fine-tune {trainable_layers}/{len(backbone.layers)} backbone layers ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_finetune),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs_finetune,
        class_weight=weights,
        callbacks=callbacks(),
    )
    model.load_weights(CHECKPOINT_PATH)

    val_true, val_probabilities = predict(model, splits["val"], args.batch_size)
    threshold = threshold_for_recall(
        val_true, val_probabilities, args.target_validation_recall
    )
    test_true, test_probabilities = predict(model, splits["test"], args.batch_size)
    results = {
        "architecture": "MobileNetV3Large",
        "class_names": list(CLASS_NAMES),
        "bill_class_indices": list(BILL_CLASS_INDICES),
        "threshold": threshold,
        "threshold_selection": f"validation recall >= {args.target_validation_recall}",
        "validation": metrics_for(val_true, val_probabilities, threshold),
        "test": metrics_for(test_true, test_probabilities, threshold),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as output:
        json.dump(results, output, indent=2)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(json.dumps(results, indent=2))
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
