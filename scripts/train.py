"""Train a lightweight binary invoice/bill classifier (Step 2 of the README).

MobileNetV3-Small backbone (ImageNet pretrained) + sigmoid head. Two-phase
schedule: freeze the backbone and train the head first, then unfreeze the
last ~25% of backbone layers and fine-tune at a low learning rate.

Sized for an 8GB-RAM machine:
  - small batch size, small shuffle buffer
  - decoded (resized, unaugmented) images are cached in RAM -- at ~150KB per
    224x224x3 uint8 image this comfortably fits for a dataset this size --
    but augmentation runs fresh every epoch, after the cache
  - no torch loaded in this process (labeling already happened separately)

Usage:
    python scripts/train.py --epochs-head 8 --epochs-finetune 12
"""

import argparse
import csv
import json
import random
from pathlib import Path

import albumentations as A
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

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "labeled" / "manifest.csv"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
FINAL_MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
CURVES_PATH = REPO_ROOT / "logs" / "training_curves.png"
METRICS_PATH = REPO_ROOT / "logs" / "final_metrics.json"

IMG_SIZE = 224
# Max decoded images to hold in the .cache() RAM buffer for a split, to
# stay well inside the 8GB budget alongside the model/TF runtime.
MAX_CACHE_BYTES = 1_000_000_000

AUGMENT = A.Compose(
    [
        A.Rotate(limit=15, border_mode=0, p=0.7),
        A.Perspective(scale=(0.03, 0.08), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.25),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.05, 0.2), hole_width_range=(0.05, 0.2), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ImageCompression(quality_range=(40, 90), p=0.4),
    ]
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def make_splits(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Stratified 80/10/10 split, grouped by `source_doc` so a real document
    and any synthetic composites derived from it always land in the same
    split -- otherwise the model could be tested on a warped/composited
    near-duplicate of something it trained on."""
    labels = [int(r["invoice_label"]) for r in rows]
    groups = [r.get("source_doc") or r["filepath"] for r in rows]

    splitter1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(splitter1.split(rows, labels, groups))
    train_rows = [rows[i] for i in train_idx]
    temp_rows = [rows[i] for i in temp_idx]
    temp_labels = [labels[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]

    splitter2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_idx, test_idx = next(splitter2.split(temp_rows, temp_labels, temp_groups))
    val_rows = [temp_rows[i] for i in val_idx]
    test_rows = [temp_rows[i] for i in test_idx]

    # Sanity check: no source_doc group should ever straddle two splits.
    split_of_group: dict[str, str] = {}
    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        for r in split_rows:
            g = r.get("source_doc") or r["filepath"]
            prev = split_of_group.get(g)
            assert prev is None or prev == split_name, f"leakage: group {g} appears in both {prev} and {split_name}"
            split_of_group[g] = split_name

    with open(SPLITS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "invoice_label", "split", "source_doc"])
        for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            for r in split_rows:
                writer.writerow([r["filepath"], r["invoice_label"], split_name, r.get("source_doc") or r["filepath"]])

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def _albumentations_augment(image: np.ndarray) -> np.ndarray:
    return AUGMENT(image=image)["image"]


def build_dataset(
    rows: list[dict], batch_size: int, augment: bool, shuffle: bool
) -> tf.data.Dataset:
    paths = [str(REPO_ROOT / r["filepath"]) for r in rows]
    labels = [float(r["invoice_label"]) for r in rows]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.uint8)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    approx_bytes = len(rows) * IMG_SIZE * IMG_SIZE * 3
    if approx_bytes <= MAX_CACHE_BYTES:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(rows), 2000), seed=0, reshuffle_each_iteration=True)

    if augment:
        def _augment(img, label):
            img = tf.numpy_function(_albumentations_augment, [img], tf.uint8)
            img.set_shape((IMG_SIZE, IMG_SIZE, 3))
            return img, label

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda img, label: (tf.cast(img, tf.float32), label), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,  # model expects raw [0,255] float input
        pooling="avg",
    )
    base.trainable = False

    x = tf.keras.layers.Dropout(0.2)(base.output)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="invoice_prob")(x)
    model = tf.keras.Model(base.input, outputs, name="invoice_classifier")
    return model, base


def compute_class_weight(rows: list[dict]) -> dict[int, float]:
    labels = np.array([int(r["invoice_label"]) for r in rows])
    n = len(labels)
    n_pos = labels.sum()
    n_neg = n - n_pos
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}


def metrics_list() -> list:
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def plot_curves(histories: list[tuple[str, tf.keras.callbacks.History]]) -> None:
    import matplotlib.pyplot as plt

    CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    offset = 0
    for phase_name, history in histories:
        epochs = [offset + e for e in range(len(history.history["loss"]))]
        axes[0].plot(epochs, history.history["loss"], label=f"{phase_name} train")
        axes[0].plot(epochs, history.history["val_loss"], label=f"{phase_name} val", linestyle="--")
        axes[1].plot(epochs, history.history["accuracy"], label=f"{phase_name} train")
        axes[1].plot(epochs, history.history["val_accuracy"], label=f"{phase_name} val", linestyle="--")
        offset += len(history.history["loss"])

    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=120)
    print(f"Saved training curves to {CURVES_PATH}")


def evaluate_on_test(model: tf.keras.Model, test_rows: list[dict], batch_size: int) -> dict:
    test_ds = build_dataset(test_rows, batch_size=batch_size, augment=False, shuffle=False)
    y_true = np.array([int(r["invoice_label"]) for r in test_rows])
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "n_test": len(y_true),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=12)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-5)
    parser.add_argument("--unfreeze-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="cap total rows (debug/smoke test)")
    args = parser.parse_args()

    set_seeds(args.seed)
    configure_memory_growth()

    rows = load_manifest()
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]

    splits = make_splits(rows, args.seed)
    print(
        f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    train_ds = build_dataset(splits["train"], args.batch_size, augment=True, shuffle=True)
    val_ds = build_dataset(splits["val"], args.batch_size, augment=False, shuffle=False)

    class_weight = compute_class_weight(splits["train"])
    print(f"Class weight: {class_weight}")

    model, base = build_model()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        str(CHECKPOINT_DIR / "best.weights.h5"),
        save_best_only=True,
        save_weights_only=True,
        monitor="val_auc",
        mode="max",
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=4, restore_best_weights=True
    )

    print("\n=== Phase 1: train head (backbone frozen) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_head),
        loss="binary_crossentropy",
        metrics=metrics_list(),
    )
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weight,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    print("\n=== Phase 2: fine-tune last layers of backbone ===")
    base.trainable = True
    n_layers = len(base.layers)
    n_frozen = int(n_layers * (1 - args.unfreeze_frac))
    for layer in base.layers[:n_frozen]:
        layer.trainable = False
    print(f"Unfroze {n_layers - n_frozen}/{n_layers} backbone layers")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr_finetune),
        loss="binary_crossentropy",
        metrics=metrics_list(),
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        class_weight=class_weight,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    model.load_weights(str(CHECKPOINT_DIR / "best.weights.h5"))

    plot_curves([("head", history_head), ("finetune", history_finetune)])

    print("\n=== Final evaluation on held-out test split ===")
    metrics = evaluate_on_test(model, splits["test"], args.batch_size)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved final metrics to {METRICS_PATH}")

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    print(f"Saved final model to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
