"""Assemble a raw image pool and validate its labels with a teacher model.

Runs as two phases so the heavy ML libraries are only loaded when needed
(the machine this trains on has 8GB of RAM, so we keep peak memory low by
never holding more than one image / one small batch in memory at a time):

  assemble  - streams images from HF datasets straight to disk (no torch
              needed), writing data/raw/pool_manifest.csv
  label     - loads the DiT teacher model once, runs it over the pool in
              small batches, and records its prediction as label metadata;
              the dataset's known source label remains the training target
  reconcile - repairs an existing manifest produced by an older version
              that incorrectly treated the teacher prediction as truth
  all       - runs both phases in sequence (default)

Usage:
    python scripts/build_dataset.py all --target-per-class 3000
"""

import argparse
import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "pool"
POOL_MANIFEST = REPO_ROOT / "data" / "raw" / "pool_manifest.csv"
LABELED_MANIFEST = REPO_ROOT / "data" / "labeled" / "manifest.csv"
SPLITS_PATH = REPO_ROOT / "data" / "labeled" / "splits.csv"
MAX_SIDE = 512  # cap stored image size to keep disk/IO light

# These datasets have ground-truth semantics at the source level. In
# particular, DiT/RVL-CDIP has a separate "invoice" class but no receipt
# class, so using `predicted_class == "invoice"` as the target mislabeled
# more than half of CORD/SROIE receipts as negatives. The teacher prediction
# is still useful metadata for auditing difficult samples, but it must not
# override known ground truth.
SOURCE_LABELS = {
    "cord_receipts": 1,
    "sroie_receipts": 1,
    "rvlcdip_invoice": 1,
    "rvlcdip_other": 0,
    "coco_photos": 0,
}


def label_for_source(source: str) -> int:
    try:
        return SOURCE_LABELS[source]
    except KeyError as exc:
        raise ValueError(f"No ground-truth label configured for source {source!r}") from exc


@dataclass
class Source:
    name: str
    hf_id: str
    split: str
    pool_label: str  # "positive" or "negative"
    max_images: int
    quota_frac: float  # fixed share of the class's target_per_class, independent of other sources
    label_field: Optional[str] = None  # ClassLabel column name, if any
    exclude_label_name: Optional[str] = None  # skip rows whose label matches this
    config: Optional[str] = None


def default_sources() -> list[Source]:
    return [
        # Positive-leaning: RVL-CDIP invoice class + real photographed receipts.
        Source("cord_receipts", "naver-clova-ix/cord-v2", "train", "positive", max_images=1200, quota_frac=0.2),
        Source(
            "sroie_receipts",
            "arvindrajan92/sroie_document_understanding",
            "train",
            "positive",
            max_images=1200,
            quota_frac=0.2,
        ),
        Source(
            "rvlcdip_invoice", "chainyo/rvl-cdip-invoice", "train", "positive", max_images=10_000, quota_frac=0.6
        ),
        # Negative-leaning: other RVL-CDIP document classes + generic photos (hard negatives).
        Source(
            "rvlcdip_other",
            "jinhybr/rvl_cdip_400_train_val_test",
            "train",
            "negative",
            max_images=10_000,
            quota_frac=0.6,
            label_field="label",
            exclude_label_name="invoice",
        ),
        Source(
            "coco_photos", "detection-datasets/coco", "train", "negative", max_images=10_000, quota_frac=0.4
        ),
    ]


def iter_source_images(source: Source) -> Iterator[Image.Image]:
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.config, split=source.split, streaming=True)
    features = ds.features
    exclude_idx = None
    if source.label_field and source.exclude_label_name:
        exclude_idx = features[source.label_field].str2int(source.exclude_label_name)

    for example in ds:
        if exclude_idx is not None and example[source.label_field] == exclude_idx:
            continue
        yield example["image"]


def _stream_and_save(source: Source, quota: int, start_idx: int, writer) -> int:
    """Stream up to `quota` new images from `source`, saving to disk and
    writing manifest rows as they arrive. Returns the number saved."""
    if quota <= 0:
        return 0
    print(f"[{source.name}] streaming up to {quota} images (starting at index {start_idx})...")
    saved = 0
    try:
        for img in iter_source_images(source):
            if saved >= quota:
                break
            try:
                img = img.convert("RGB")
                img.thumbnail((MAX_SIDE, MAX_SIDE))
                idx = start_idx + saved
                fname = f"{source.pool_label[:3]}_{source.name}_{idx:05d}.jpg"
                fpath = RAW_DIR / fname
                img.save(fpath, format="JPEG", quality=88)
            except Exception as e:  # noqa: BLE001 - one bad image shouldn't kill the run
                print(f"  skip (decode/save error): {e}")
                continue

            writer.writerow([str(fpath.relative_to(REPO_ROOT)), source.name, source.pool_label])
            saved += 1
            if saved % 200 == 0:
                print(f"  {source.name}: {saved}/{quota}")
    except Exception as e:  # noqa: BLE001 - network hiccups shouldn't kill the whole run
        print(f"[{source.name}] stopped early after {saved} images due to: {e}")

    print(f"[{source.name}] done: saved {saved} images")
    return saved


def assemble_pool(target_per_class: int, seed: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    POOL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    sources = default_sources()
    random.seed(seed)
    counts = {"positive": 0, "negative": 0}
    saved_per_source: dict[str, int] = {}

    with open(POOL_MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "source", "pool_label"])

        # Pass 1: each source gets its own fixed quota (a fraction of the
        # class target), independent of how much other sources in the same
        # class yield. This guarantees e.g. hard-negative photos aren't
        # crowded out just because a document-negative source has plenty
        # of supply.
        for source in sources:
            quota = min(source.max_images, round(target_per_class * source.quota_frac))
            saved = _stream_and_save(source, quota, start_idx=0, writer=writer)
            saved_per_source[source.name] = saved
            counts[source.pool_label] += saved
            f.flush()

        # Pass 2: top up any class that fell short of its target (e.g. a
        # source had less supply than its quota_frac assumed), pulling the
        # shortfall from sources in that class with remaining headroom.
        for pool_label in ("positive", "negative"):
            shortfall = target_per_class - counts[pool_label]
            if shortfall <= 0:
                continue
            class_sources = [s for s in sources if s.pool_label == pool_label]
            for source in class_sources:
                if shortfall <= 0:
                    break
                headroom = source.max_images - saved_per_source[source.name]
                if headroom <= 0:
                    continue
                extra_quota = min(headroom, shortfall)
                extra_saved = _stream_and_save(
                    source, extra_quota, start_idx=saved_per_source[source.name], writer=writer
                )
                saved_per_source[source.name] += extra_saved
                counts[pool_label] += extra_saved
                shortfall -= extra_saved
                f.flush()

    print(f"\nPool assembled: positive={counts['positive']} negative={counts['negative']}")
    print(f"Manifest: {POOL_MANIFEST}")


def label_pool(batch_size: int, device: str) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    MODEL_NAME = "microsoft/dit-base-finetuned-rvlcdip"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    id2label = model.config.id2label
    invoice_id = next(i for i, name in id2label.items() if name.lower() == "invoice")

    with open(POOL_MANIFEST) as f:
        rows = list(csv.DictReader(f))

    LABELED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    counts = {0: 0, 1: 0}
    teacher_agreements = 0
    teacher_disagreements = 0

    with open(LABELED_MANIFEST, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["filepath", "source", "predicted_class", "invoice_label", "confidence", "source_doc"])

        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            images, valid_rows = [], []
            for row in batch:
                try:
                    images.append(Image.open(REPO_ROOT / row["filepath"]).convert("RGB"))
                    valid_rows.append(row)
                except Exception as e:  # noqa: BLE001
                    print(f"  skip unreadable {row['filepath']}: {e}")

            if not images:
                continue

            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            for row, prob in zip(valid_rows, probs):
                top_idx = int(torch.argmax(prob))
                teacher_label = int(top_idx == invoice_id)
                invoice_label = label_for_source(row["source"])
                if teacher_label == invoice_label:
                    teacher_agreements += 1
                else:
                    teacher_disagreements += 1
                writer.writerow(
                    [
                        row["filepath"],
                        row["source"],
                        id2label[top_idx],
                        invoice_label,
                        f"{float(prob[top_idx]):.4f}",
                        row["filepath"],  # source_doc: a real image is its own lineage root
                    ]
                )
                counts[invoice_label] += 1

            out_f.flush()
            done = min(batch_start + batch_size, len(rows))
            print(f"  labeled {done}/{len(rows)}")

    print(f"\nLabeling done: invoice=1 -> {counts[1]}, invoice=0 -> {counts[0]}")
    print(
        "Teacher agreement with source ground truth: "
        f"{teacher_agreements}/{teacher_agreements + teacher_disagreements} "
        f"({teacher_disagreements} disagreements retained as audit metadata)"
    )
    print(f"Manifest: {LABELED_MANIFEST}")


def reconcile_manifest_labels() -> None:
    """Repair targets in an existing manifest without rerunning the teacher.

    Synthetic examples inherit the corrected target of their lineage root.
    This also repairs old "hard negatives" synthesized from receipts that
    the teacher had incorrectly labeled as non-invoices.
    """
    with open(LABELED_MANIFEST) as f:
        rows = list(csv.DictReader(f))

    base_rows = {r["filepath"]: r for r in rows if not r["source"].startswith("synthetic_")}
    changed = 0
    counts = {0: 0, 1: 0}

    for row in rows:
        if row["source"].startswith("synthetic_"):
            source_doc = base_rows.get(row["source_doc"])
            if source_doc is None:
                raise ValueError(f"Missing lineage root {row['source_doc']!r} for {row['filepath']!r}")
            label = label_for_source(source_doc["source"])
            row["source"] = "synthetic_hard_pos" if label else "synthetic_hard_neg"
        else:
            label = label_for_source(row["source"])

        if row["invoice_label"] != str(label):
            changed += 1
        row["invoice_label"] = str(label)
        counts[label] += 1

    temp_path = LABELED_MANIFEST.with_suffix(".csv.tmp")
    with open(temp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(LABELED_MANIFEST)

    # Preserve existing split membership for before/after comparisons while
    # keeping its copied labels consistent with the repaired manifest.
    if SPLITS_PATH.exists():
        label_by_path = {r["filepath"]: r["invoice_label"] for r in rows}
        with open(SPLITS_PATH) as f:
            split_rows = list(csv.DictReader(f))
        for split_row in split_rows:
            split_row["invoice_label"] = label_by_path[split_row["filepath"]]
        split_temp_path = SPLITS_PATH.with_suffix(".csv.tmp")
        with open(split_temp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=split_rows[0].keys())
            writer.writeheader()
            writer.writerows(split_rows)
        split_temp_path.replace(SPLITS_PATH)

    print(
        f"Reconciled {len(rows)} rows ({changed} labels corrected): "
        f"invoice=1 -> {counts[1]}, invoice=0 -> {counts[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["assemble", "label", "reconcile", "all"], nargs="?", default="all")
    parser.add_argument("--target-per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()

    if args.phase in ("assemble", "all"):
        assemble_pool(args.target_per_class, args.seed)
    if args.phase in ("label", "all"):
        label_pool(args.batch_size, args.device)
    if args.phase == "reconcile":
        reconcile_manifest_labels()


if __name__ == "__main__":
    main()
