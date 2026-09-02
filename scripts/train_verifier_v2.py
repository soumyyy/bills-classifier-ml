"""Train a recall-constrained, multi-task MobileNetV3-Large bill verifier.

This experiment keeps the seven-way document taxonomy as an auxiliary task,
but adds a dedicated bill/non-bill head.  Checkpoints are selected by the
false-positive rate achieved at a requested validation recall rather than by
generic validation loss.  The resize mode is configurable so experiments can
match the production preprocessing path exactly.

The existing production artifacts are never overwritten.  Outputs go under
``models/experiments/verifier_v2`` and ``logs/experiments/verifier_v2``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "verifier_v2_aligned_splits.csv"
EXPERIMENT_NAME = "verifier_v2"
MODEL_DIR = REPO_ROOT / "models" / "experiments" / EXPERIMENT_NAME
LOG_DIR = REPO_ROOT / "logs" / "experiments" / EXPERIMENT_NAME
CHECKPOINT_PATH = MODEL_DIR / "best.weights.h5"
MODEL_PATH = MODEL_DIR / "model.keras"
METRICS_PATH = LOG_DIR / "metrics.json"
HISTORY_PATH = LOG_DIR / "history.json"
IMAGENET_WEIGHTS = (
    REPO_ROOT
    / "models"
    / "checkpoints"
    / "weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5"
)

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
MACOS_DATALESS_FLAG = 0x40000000

# The original seven-class trainer put every app-specific negative into the
# structured-document bucket.  These two observed failures are visibly a
# handwritten clinical note and a multi-panel app advertisement, so retaining
# the blanket label would make the auxiliary task teach the wrong distinction.
TARGET_CLASS_OVERRIDES = {
    "Negative/3096da16-2470-4c67-92c2-9b3125b0911f.jpg": "narrative_document",
    "Negative/35df1041-4635-4252-9158-a89ae3de1a46.jpg": "advertisement_presentation",
}


def make_augment() -> A.Compose:
    white = (255, 255, 255)
    return A.Compose(
        [
            A.Affine(
                scale=(0.78, 1.05),
                translate_percent=(-0.08, 0.08),
                rotate=(-10, 10),
                shear=(-3, 3),
                border_mode=cv2.BORDER_CONSTANT,
                fill=white,
                p=0.75,
            ),
            A.Perspective(scale=(0.02, 0.07), fill=white, p=0.30),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.65
            ),
            A.RandomGamma(gamma_limit=(65, 140), p=0.25),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.20
            ),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.03, 0.16),
                hole_width_range=(0.03, 0.16),
                fill=white,
                p=0.18,
            ),
            A.GaussNoise(std_range=(0.01, 0.06), p=0.18),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.ImageCompression(quality_range=(45, 95), p=0.35),
        ]
    )


AUGMENT = make_augment()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_splits(data_root: Path, splits_path: Path = SPLITS_PATH) -> dict[str, list[dict]]:
    with open(splits_path) as source:
        rows = list(csv.DictReader(source))
    result = {"train": [], "val": [], "test": []}
    missing = []
    offloaded = []
    for row in rows:
        row = dict(row)
        override = TARGET_CLASS_OVERRIDES.get(row["filepath"])
        if override is not None:
            row["class_name"] = override
            row["class_index"] = str(CLASS_NAMES.index(override))
        row["class_index"] = int(row["class_index"])
        row["bill_label"] = int(row["class_index"] in BILL_CLASS_INDICES)
        row["absolute_path"] = str(data_root / row["filepath"])
        absolute_path = Path(row["absolute_path"])
        if not absolute_path.is_file():
            missing.append(row["absolute_path"])
        elif getattr(absolute_path.stat(), "st_flags", 0) & MACOS_DATALESS_FLAG:
            offloaded.append(row["absolute_path"])
        result[row["split"]].append(row)
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} dataset files are missing; first paths:\n{preview}")
    if offloaded:
        preview = "\n".join(offloaded[:10])
        raise RuntimeError(
            f"{len(offloaded)} dataset files are iCloud placeholders. Download them "
            f"before training; first paths:\n{preview}"
        )
    return result


def describe_splits(splits: dict[str, list[dict]]) -> None:
    for split, rows in splits.items():
        classes = Counter(row["class_name"] for row in rows)
        bills = Counter(row["bill_label"] for row in rows)
        print(f"{split}: n={len(rows)} bill_labels={dict(bills)} classes={dict(classes)}")


def letterbox(image: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = tf.minimum(image_size / shape[0], image_size / shape[1])
    resized_shape = tf.maximum(1, tf.cast(tf.round(shape * scale), tf.int32))
    image = tf.image.resize(image, resized_shape, antialias=True)
    pad_h = image_size - resized_shape[0]
    pad_w = image_size - resized_shape[1]
    top = pad_h // 2
    left = pad_w // 2
    image = tf.pad(
        image,
        [[top, pad_h - top], [left, pad_w - left], [0, 0]],
        constant_values=255.0,
    )
    image.set_shape((image_size, image_size, 3))
    return image


def resize_image(image: tf.Tensor, image_size: int, resize_mode: str) -> tf.Tensor:
    if resize_mode == "letterbox":
        return letterbox(image, image_size)
    if resize_mode == "direct":
        image = tf.image.resize(
            tf.cast(image, tf.float32), (image_size, image_size), antialias=True
        )
        image.set_shape((image_size, image_size, 3))
        return image
    raise ValueError(f"Unsupported resize mode: {resize_mode}")


def augment_numpy(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0, 255).astype(np.uint8)
    return AUGMENT(image=image)["image"]


def sample_weights(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    bill_labels = np.array([row["bill_label"] for row in rows], dtype=np.int32)
    bill_counts = np.bincount(bill_labels, minlength=2)
    binary_weight = len(rows) / (2.0 * bill_counts)

    class_indices = np.array([row["class_index"] for row in rows], dtype=np.int32)
    class_counts = np.bincount(class_indices, minlength=len(CLASS_NAMES))
    class_weight = len(rows) / (len(CLASS_NAMES) * class_counts)

    bill_weights = binary_weight[bill_labels]
    document_weights = class_weight[class_indices]
    for index, row in enumerate(rows):
        relative = row["filepath"]
        if relative.startswith("Negative/"):
            # The tiny app-specific set represents costly observed failures.
            bill_weights[index] *= 2.0
            document_weights[index] *= 1.5
        elif relative.startswith("data/raw/public_hard_negatives/"):
            bill_weights[index] *= 1.35
    # Keep a few examples from dominating an entire mini-batch.
    return np.minimum(bill_weights, 8.0), np.minimum(document_weights, 5.0)


def build_dataset(
    rows: list[dict],
    data_root: Path,
    image_size: int,
    resize_mode: str,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    include_targets: bool = True,
) -> tf.data.Dataset:
    paths = [str(data_root / row["filepath"]) for row in rows]
    bill_labels = np.array([row["bill_label"] for row in rows], dtype=np.float32)
    class_labels = np.array([row["class_index"] for row in rows], dtype=np.int32)
    bill_weights, document_weights = sample_weights(rows)
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, bill_labels, class_labels, bill_weights, document_weights)
    )

    def load(path, bill_label, class_label, bill_weight, document_weight):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape((None, None, 3))
        image = resize_image(image, image_size, resize_mode)
        return image, bill_label, class_label, bill_weight, document_weight

    dataset = dataset.map(load, num_parallel_calls=4)
    if shuffle:
        dataset = dataset.shuffle(
            min(2500, len(rows)), seed=0, reshuffle_each_iteration=True
        )
    if augment:
        def apply_augment(image, bill_label, class_label, bill_weight, document_weight):
            image = tf.numpy_function(augment_numpy, [image], tf.uint8)
            image.set_shape((image_size, image_size, 3))
            return image, bill_label, class_label, bill_weight, document_weight

        dataset = dataset.map(apply_augment, num_parallel_calls=4)

    if include_targets:
        def prepare(image, bill_label, class_label, bill_weight, document_weight):
            targets = {
                "bill_prob": tf.reshape(bill_label, (1,)),
                "document_class": tf.one_hot(class_label, len(CLASS_NAMES)),
            }
            weights = {
                "bill_prob": bill_weight,
                "document_class": document_weight,
            }
            return tf.cast(image, tf.float32), targets, weights

        dataset = dataset.map(prepare, num_parallel_calls=4)
    else:
        dataset = dataset.map(
            lambda image, bill_label, class_label, bill_weight, document_weight: tf.cast(
                image, tf.float32
            ),
            num_parallel_calls=4,
        )
    return dataset.batch(batch_size).prefetch(1)


def build_model(image_size: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    backbone = tf.keras.applications.MobileNetV3Large(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=None,
        include_preprocessing=True,
        pooling="avg",
    )
    backbone.load_weights(IMAGENET_WEIGHTS)
    backbone.trainable = False
    features = tf.keras.layers.Dropout(0.28, name="shared_dropout")(backbone.output)
    bill_prob = tf.keras.layers.Dense(1, activation="sigmoid", name="bill_prob")(features)
    document_class = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="document_class"
    )(features)
    model = tf.keras.Model(
        backbone.input,
        {"bill_prob": bill_prob, "document_class": document_class},
        name="invoice_verifier_v2",
    )
    return model, backbone


def repeat_target_negatives(rows: list[dict], repeat: int) -> list[dict]:
    if repeat < 1:
        raise ValueError("target-negative-repeat must be at least 1")
    target_rows = [row for row in rows if row["filepath"].startswith("Negative/")]
    print(
        f"Target hard-negative replay: {len(target_rows)} unique x {repeat} "
        f"= {len(target_rows) * repeat} augmented examples/epoch"
    )
    return rows + target_rows * (repeat - 1)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-5, clipnorm=1.0
        ),
        loss={
            "bill_prob": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "document_class": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        },
        loss_weights={"bill_prob": 0.78, "document_class": 0.22},
        metrics={
            "bill_prob": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="roc_auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            ],
            "document_class": [tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        },
    )


def threshold_at_recall(true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    positive_scores = np.sort(scores[true == 1])
    allowed_misses = int(np.floor((1.0 - target_recall) * len(positive_scores) + 1e-12))
    allowed_misses = min(max(allowed_misses, 0), len(positive_scores) - 1)
    return float(np.nextafter(positive_scores[allowed_misses], -np.inf))


def unpack_predictions(predictions) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(predictions, dict):
        return predictions["bill_prob"].ravel(), predictions["document_class"]
    if isinstance(predictions, (list, tuple)):
        # Keras follows model.output_names for list outputs.
        by_name = dict(zip(("bill_prob", "document_class"), predictions))
        return by_name["bill_prob"].ravel(), by_name["document_class"]
    raise TypeError(f"Unexpected prediction type: {type(predictions)}")


class RecallConstrainedCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        validation_images: tf.data.Dataset,
        true_labels: np.ndarray,
        target_recall: float,
        filepath: Path,
    ) -> None:
        super().__init__()
        self.validation_images = validation_images
        self.true_labels = true_labels
        self.target_recall = target_recall
        self.filepath = filepath
        self.best_fpr = np.inf
        self.best_auc = -np.inf
        self.best_epoch = -1
        self.records: list[dict] = []

    def on_epoch_end(self, epoch, logs=None):
        bill_scores, _ = unpack_predictions(
            self.model.predict(self.validation_images, verbose=0)
        )
        threshold = threshold_at_recall(
            self.true_labels, bill_scores, self.target_recall
        )
        predicted = bill_scores >= threshold
        negatives = self.true_labels == 0
        fpr = float(predicted[negatives].mean())
        recall = float(predicted[self.true_labels == 1].mean())
        auc = float(roc_auc_score(self.true_labels, bill_scores))
        record = {
            "epoch": int(epoch),
            "threshold": threshold,
            "recall": recall,
            "fpr": fpr,
            "auc": auc,
        }
        self.records.append(record)
        improved = fpr < self.best_fpr - 1e-12 or (
            abs(fpr - self.best_fpr) <= 1e-12 and auc > self.best_auc
        )
        if improved:
            self.best_fpr = fpr
            self.best_auc = auc
            self.best_epoch = int(epoch)
            self.model.save_weights(self.filepath)
        print(
            f"\nrecall-constrained val: recall={recall:.4f} fpr={fpr:.4f} "
            f"threshold={threshold:.6f} auc={auc:.4f}"
            + (" [saved]" if improved else "")
        )


def set_trainable_fraction(backbone: tf.keras.Model, fraction: float) -> int:
    backbone.trainable = True
    first_trainable = int(len(backbone.layers) * (1.0 - fraction))
    count = 0
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= first_trainable and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
        count += int(layer.trainable)
    return count


def evaluate(
    model: tf.keras.Model,
    images: tf.data.Dataset,
    rows: list[dict],
    threshold: float,
) -> dict:
    bill_scores, class_probabilities = unpack_predictions(model.predict(images, verbose=0))
    true = np.array([row["bill_label"] for row in rows], dtype=np.int32)
    true_classes = np.array([row["class_index"] for row in rows], dtype=np.int32)
    predicted = (bill_scores >= threshold).astype(np.int32)
    predicted_classes = class_probabilities.argmax(axis=1)
    negative = true == 0
    return {
        "n": int(len(rows)),
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted)),
        "fpr": float(predicted[negative].mean()),
        "auc": float(roc_auc_score(true, bill_scores)),
        "multiclass_accuracy": float(accuracy_score(true_classes, predicted_classes)),
        "multiclass_macro_f1": float(
            f1_score(true_classes, predicted_classes, average="macro")
        ),
    }


def fit_phase(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    epochs: int,
    learning_rate: float,
    checkpoint: RecallConstrainedCheckpoint,
    phase_name: str,
) -> dict:
    print(f"\n=== {phase_name} ===")
    compile_model(model, learning_rate)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=2,
    )
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--splits-path", type=Path, default=SPLITS_PATH)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--resize-mode",
        choices=("direct", "letterbox"),
        default="direct",
        help="Direct resize matches the production pipeline and preserves text pixels.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--lr-head", type=float, default=8e-4)
    parser.add_argument("--lr-finetune", type=float, default=1.5e-5)
    parser.add_argument("--unfreeze-frac", type=float, default=0.35)
    parser.add_argument("--target-validation-recall", type=float, default=0.99)
    parser.add_argument("--target-negative-repeat", type=int, default=8)
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="Optional compatible .keras or weights file for a short continuation run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Output subdirectory under models/experiments and logs/experiments.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    configure_tensorflow()
    model_dir = REPO_ROOT / "models" / "experiments" / args.experiment_name
    log_dir = REPO_ROOT / "logs" / "experiments" / args.experiment_name
    checkpoint_path = model_dir / "best.weights.h5"
    model_path = model_dir / "model.keras"
    metrics_path = log_dir / "metrics.json"
    history_path = log_dir / "history.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.data_root.resolve(), args.splits_path.resolve())
    describe_splits(splits)
    if args.dry_run:
        return

    train_rows = repeat_target_negatives(
        splits["train"], args.target_negative_repeat
    )
    train_dataset = build_dataset(
        train_rows,
        args.data_root,
        args.image_size,
        args.resize_mode,
        args.batch_size,
        True,
        True,
    )
    validation_dataset = build_dataset(
        splits["val"],
        args.data_root,
        args.image_size,
        args.resize_mode,
        args.batch_size,
        False,
        False,
    )
    validation_images = build_dataset(
        splits["val"],
        args.data_root,
        args.image_size,
        args.resize_mode,
        args.batch_size,
        False,
        False,
        include_targets=False,
    )
    test_images = build_dataset(
        splits["test"],
        args.data_root,
        args.image_size,
        args.resize_mode,
        args.batch_size,
        False,
        False,
        include_targets=False,
    )
    validation_true = np.array(
        [row["bill_label"] for row in splits["val"]], dtype=np.int32
    )

    model, backbone = build_model(args.image_size)
    if args.initial_weights is not None:
        print(f"Loading initial weights from {args.initial_weights}")
        model.load_weights(args.initial_weights)
    checkpoint = RecallConstrainedCheckpoint(
        validation_images,
        validation_true,
        args.target_validation_recall,
        checkpoint_path,
    )
    histories = {}
    if args.epochs_head > 0:
        histories["head"] = fit_phase(
            model,
            train_dataset,
            validation_dataset,
            args.epochs_head,
            args.lr_head,
            checkpoint,
            "Phase 1: train multi-task heads",
        )
    else:
        histories["head"] = {}
    trainable_count = set_trainable_fraction(backbone, args.unfreeze_frac)
    print(
        f"Fine-tuning {trainable_count}/{len(backbone.layers)} backbone layers; "
        "BatchNorm remains frozen"
    )
    histories["finetune"] = fit_phase(
        model,
        train_dataset,
        validation_dataset,
        args.epochs_finetune,
        args.lr_finetune,
        checkpoint,
        "Phase 2: fine-tune high-level visual features",
    )

    model.load_weights(checkpoint_path)
    validation_scores, _ = unpack_predictions(model.predict(validation_images, verbose=0))
    threshold = threshold_at_recall(
        validation_true, validation_scores, args.target_validation_recall
    )
    results = {
        "architecture": "MobileNetV3Large multi-task",
        "image_size": args.image_size,
        "resize_mode": args.resize_mode,
        "preprocessing": f"{args.resize_mode} resize; raw RGB float [0,255]",
        "class_names": list(CLASS_NAMES),
        "bill_class_indices": list(BILL_CLASS_INDICES),
        "threshold": threshold,
        "threshold_selection": (
            f"minimum validation FPR at recall >= {args.target_validation_recall}"
        ),
        "best_epoch": checkpoint.best_epoch,
        "validation": evaluate(model, validation_images, splits["val"], threshold),
        "test": evaluate(model, test_images, splits["test"], threshold),
        "recall_constrained_history": checkpoint.records,
        "training": vars(args)
        | {
            "data_root": str(args.data_root),
            "splits_path": str(args.splits_path),
            "initial_weights": (
                str(args.initial_weights) if args.initial_weights is not None else None
            ),
        },
    }
    model.save(model_path)
    with open(metrics_path, "w") as output:
        json.dump(results, output, indent=2)
    with open(history_path, "w") as output:
        json.dump(histories, output, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved experiment model to {model_path}")


if __name__ == "__main__":
    main()
