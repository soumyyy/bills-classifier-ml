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

from _common import require_file, run_fingerprint

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
TARGET_HARD_NEGATIVES = REPO_ROOT / "data" / "labeled" / "target_hard_negatives.csv"
PUBLIC_HARD_NEGATIVES = REPO_ROOT / "data" / "labeled" / "public_hard_negatives.csv"
MINED_HARD_NEGATIVES = REPO_ROOT / "data" / "labeled" / "mined_hard_negatives.csv"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
FINAL_MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
CURVES_PATH = REPO_ROOT / "logs" / "training_curves.png"
METRICS_PATH = REPO_ROOT / "logs" / "final_metrics.json"
DEPLOYMENT_METRICS_PATH = REPO_ROOT / "logs" / "deployment_metrics.json"
# Used only when no deployment record exists yet; the app ships 0.15.
DEFAULT_DECISION_THRESHOLD = 0.15

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
    with open(require_file(MANIFEST, "python scripts/build_dataset.py")) as f:
        rows = list(csv.DictReader(f))

    # Small, app-specific regression sets are maintained separately from
    # the reproducible public-data manifest. Their explicit split prevents
    # a user photo from silently moving between train and test after a seed
    # or dataset change.
    if TARGET_HARD_NEGATIVES.exists():
        with open(TARGET_HARD_NEGATIVES) as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "filepath": row["filepath"],
                        "source": "target_hard_negative",
                        "predicted_class": "user_labeled_non_bill",
                        "invoice_label": "0",
                        "confidence": "1.0000",
                        "source_doc": row["filepath"],
                        "forced_split": row["split"],
                    }
                )
    if PUBLIC_HARD_NEGATIVES.exists():
        with open(PUBLIC_HARD_NEGATIVES) as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "filepath": row["filepath"],
                        "source": row["source"],
                        "predicted_class": "public_non_bill",
                        "invoice_label": "0",
                        "confidence": "1.0000",
                        "source_doc": row["source_doc"],
                        "forced_split": row["split"],
                    }
                )
    return rows


def make_splits(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Stratified 80/10/10 split, grouped by `source_doc` so a real document
    and any synthetic composites derived from it always land in the same
    split -- otherwise the model could be tested on a warped/composited
    near-duplicate of something it trained on."""
    forced_rows = [r for r in rows if r.get("forced_split")]
    rows = [r for r in rows if not r.get("forced_split")]
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

    rows_by_split = {"train": train_rows, "val": val_rows, "test": test_rows}
    for row in forced_rows:
        split = row["forced_split"]
        if split not in rows_by_split:
            raise ValueError(f"Invalid forced split {split!r} for {row['filepath']}")
        if not (REPO_ROOT / row["filepath"]).exists():
            raise FileNotFoundError(REPO_ROOT / row["filepath"])
        rows_by_split[split].append(row)

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

    return rows_by_split


def _albumentations_augment(image: np.ndarray) -> np.ndarray:
    return AUGMENT(image=image)["image"]


def build_dataset(
    rows: list[dict], batch_size: int, augment: bool, shuffle: bool, seed: int = 0
) -> tf.data.Dataset:
    paths = [str(REPO_ROOT / r["filepath"]) for r in rows]
    labels = [float(r["invoice_label"]) for r in rows]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        # Target-domain examples include phone screenshots (PNG) as well as
        # camera JPEGs, so decode by content instead of assuming JPEG.
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape((None, None, 3))
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.uint8)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    approx_bytes = len(rows) * IMG_SIZE * IMG_SIZE * 3
    if approx_bytes <= MAX_CACHE_BYTES:
        ds = ds.cache()

    if shuffle:
        # seed comes from --seed. It was hardcoded to 0, so two runs with
        # different seeds got different splits but identical batch ordering -
        # a seed sweep meant to estimate run-to-run variance was only
        # measuring half of it.
        ds = ds.shuffle(buffer_size=min(len(rows), 2000), seed=seed, reshuffle_each_iteration=True)

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
    cached_weights = (
        Path.home()
        / ".keras"
        / "models"
        / "weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5"
    )
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        # Loading an existing cache explicitly also supports offline retrains;
        # Keras otherwise attempts a network request even when this compatible
        # weight file is already present locally.
        weights=None if cached_weights.exists() else "imagenet",
        include_preprocessing=True,  # model expects raw [0,255] float input
        pooling="avg",
    )
    if cached_weights.exists():
        base.load_weights(cached_weights)
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


def repeat_target_hard_negatives(rows: list[dict], repeat: int) -> list[dict]:
    """Oversample scarce app-specific errors so each receives fresh image
    augmentation several times per epoch without duplicating files."""
    if repeat < 1:
        raise ValueError("hard-negative-repeat must be at least 1")
    hard_rows = [r for r in rows if r.get("source") == "target_hard_negative"]
    return rows + hard_rows * (repeat - 1)


def sample_public_hard_negatives(
    rows: list[dict], fraction: float, seed: int
) -> list[dict]:
    """Keep a deterministic, source-balanced fraction of the new public
    negatives in training. Validation and test retain every imported image;
    only the training mixture is thinned to avoid overwhelming positives and
    shifting the deployed score calibration."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("public-negative-train-fraction must be in (0, 1]")

    public_by_source: dict[str, list[dict]] = {}
    retained = []
    for row in rows:
        source = row.get("source", "")
        if source == "hiertext_text_scene" or source.startswith("rvl_targeted_"):
            public_by_source.setdefault(source, []).append(row)
        else:
            retained.append(row)

    for source, source_rows in sorted(public_by_source.items()):
        random.Random(f"{seed}:{source}").shuffle(source_rows)
        keep = max(1, round(len(source_rows) * fraction))
        retained.extend(source_rows[:keep])
    return retained


def repeat_mined_hard_negatives(rows: list[dict], repeat: int) -> list[dict]:
    """Replay the public training negatives that the previous model scored
    highest. Mining only from the training split preserves validation/test
    independence while teaching a sharper document-vs-bill boundary."""
    if repeat < 1:
        raise ValueError("mined-negative-repeat must be at least 1")
    if not MINED_HARD_NEGATIVES.exists():
        return rows
    with open(MINED_HARD_NEGATIVES) as f:
        mined_paths = {r["filepath"] for r in csv.DictReader(f)}
    mined_rows = [r for r in rows if r["filepath"] in mined_paths]
    return rows + mined_rows * (repeat - 1)


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


def deployment_threshold() -> float:
    """The operating point the app actually ships.

    Metrics used to be reported at a hardcoded 0.5, which is not a threshold
    this project deploys at - so the headline accuracy and recall in
    readme.md described behaviour no user ever saw. Reading the deployed
    value keeps the reported numbers and the shipped ones the same thing.
    """
    if DEPLOYMENT_METRICS_PATH.exists():
        with open(DEPLOYMENT_METRICS_PATH) as f:
            recorded = json.load(f).get("threshold")
            if recorded is not None:
                return float(recorded)
    return DEFAULT_DECISION_THRESHOLD


def evaluate_on_test(
    model: tf.keras.Model, test_rows: list[dict], batch_size: int, threshold: float,
) -> dict:
    test_ds = build_dataset(test_rows, batch_size=batch_size, augment=False, shuffle=False)
    y_true = np.array([int(r["invoice_label"]) for r in test_rows])
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
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
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=None,
        help=(
            "Decision threshold for the final test metrics. Defaults to the "
            "deployed operating point in logs/deployment_metrics.json, so the "
            "reported numbers describe what the app actually does."
        ),
    )
    parser.add_argument(
        "--hard-negative-repeat",
        type=int,
        default=20,
        help="effective copies of each target hard negative per training epoch",
    )
    parser.add_argument(
        "--mined-negative-repeat",
        type=int,
        default=1,
        help="effective copies of each model-mined public hard negative (1 disables replay)",
    )
    parser.add_argument(
        "--public-negative-train-fraction",
        type=float,
        default=0.35,
        help="source-balanced fraction of imported public hard negatives used for training",
    )
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

    train_rows = sample_public_hard_negatives(
        splits["train"], args.public_negative_train_fraction, args.seed
    )
    n_public_train = sum(
        r.get("source") == "hiertext_text_scene"
        or r.get("source", "").startswith("rvl_targeted_")
        for r in train_rows
    )
    print(
        f"Public hard negatives retained for training: {n_public_train} "
        f"({args.public_negative_train_fraction:.0%} source-balanced sample)"
    )
    train_rows = repeat_target_hard_negatives(train_rows, args.hard_negative_repeat)
    train_rows = repeat_mined_hard_negatives(train_rows, args.mined_negative_repeat)
    n_hard_train = sum(r.get("source") == "target_hard_negative" for r in splits["train"])
    print(
        f"Target hard negatives: {n_hard_train} train originals, "
        f"{sum(r.get('source') == 'target_hard_negative' for r in splits['val'])} val, "
        f"{sum(r.get('source') == 'target_hard_negative' for r in splits['test'])} test; "
        f"effective train copies={n_hard_train * args.hard_negative_repeat}"
    )
    n_mined = 0
    if MINED_HARD_NEGATIVES.exists():
        with open(MINED_HARD_NEGATIVES) as f:
            mined_paths = {r["filepath"] for r in csv.DictReader(f)}
        n_mined = sum(r["filepath"] in mined_paths for r in splits["train"])
    print(f"Mined public hard negatives: {n_mined}; effective copies={n_mined * args.mined_negative_repeat}")

    train_ds = build_dataset(train_rows, args.batch_size, augment=True, shuffle=True, seed=args.seed)
    val_ds = build_dataset(splits["val"], args.batch_size, augment=False, shuffle=False)

    class_weight = compute_class_weight(train_rows)
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
    n_trainable = 0
    for index, layer in enumerate(base.layers):
        # Updating small-batch BatchNorm statistics can destroy useful
        # ImageNet features during fine-tuning. Keep every BN layer frozen
        # while unfreezing only the tail's convolutional/dense parameters.
        layer.trainable = index >= n_frozen and not isinstance(layer, tf.keras.layers.BatchNormalization)
        n_trainable += int(layer.trainable)
    print(f"Unfroze {n_trainable}/{n_layers} non-BatchNorm backbone layers")

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
    eval_threshold = (
        deployment_threshold() if args.eval_threshold is None else args.eval_threshold
    )
    print(f"Evaluating at threshold {eval_threshold:.4f} (the deployed operating point)")
    metrics = evaluate_on_test(model, splits["test"], args.batch_size, eval_threshold)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # Recorded alongside the metrics so a result can be traced back to the
    # exact inputs that produced it. splits.csv is rewritten in place by every
    # run, so without this a model and the split file beside it can silently
    # stop corresponding - which is how splits.csv came to hold 920 test rows
    # while the recorded metrics said 733, with nothing detecting it.
    metrics["run"] = run_fingerprint(
        seed=args.seed,
        manifest=MANIFEST,
        splits=SPLITS_PATH,
        args={
            "batch_size": args.batch_size,
            "hard_negative_repeat": getattr(args, "hard_negative_repeat", None),
            "public_negative_train_fraction": getattr(args, "public_negative_train_fraction", None),
            "mined_negative_repeat": getattr(args, "mined_negative_repeat", None),
            "limit": getattr(args, "limit", None),
        },
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved final metrics to {METRICS_PATH}")
    print(
        f"  run: seed={metrics['run']['seed']} git={metrics['run']['git_revision']} "
        f"splits={metrics['run']['splits_sha256'][:12]}"
    )

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    print(f"Saved final model to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
