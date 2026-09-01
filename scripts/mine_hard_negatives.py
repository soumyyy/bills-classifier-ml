"""Find the highest-scoring non-bills in the existing training split.

Only training rows are eligible, so the validation and test sets remain
untouched. The resulting CSV is consumed by train.py for modest replay
oversampling on the next run.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "labeled" / "mined_hard_negatives.csv"
IMG_SIZE = 224


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(SPLITS_PATH) as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["split"] == "train"
            and row["invoice_label"] == "0"
            and not row["filepath"].startswith("Negative/")
        ]

    model = tf.keras.models.load_model(MODEL_PATH)
    scored: list[tuple[float, str]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = [
            np.array(
                Image.open(REPO_ROOT / row["filepath"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                dtype=np.float32,
            )
            for row in batch
        ]
        scores = model.predict(np.stack(images), verbose=0).ravel()
        scored.extend((float(score), row["filepath"]) for score, row in zip(scores, batch))

    selected = sorted(scored, reverse=True)[: args.top_k]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "score"])
        for score, filepath in selected:
            writer.writerow([filepath, f"{score:.8f}"])

    print(
        f"Saved {len(selected)} mined training negatives to {OUTPUT_PATH}; "
        f"score range={selected[-1][0]:.4f}..{selected[0][0]:.4f}"
    )


if __name__ == "__main__":
    main()
