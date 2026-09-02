"""Import targeted public non-bill images for classifier retraining.

The original RVL-CDIP stream is class-sorted, so taking its first N rows
mostly yielded letters, forms, emails, and handwriting. This importer samples
the document classes that resemble the app's false positives and combines
them with text-heavy natural scenes from Google's HierText validation archive.

Downloaded pixels remain under ignored ``data/raw``. The small provenance and
split manifest is written to ``data/labeled/public_hard_negatives.csv``.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "public_hard_negatives"
OUTPUT_CSV = REPO_ROOT / "data" / "labeled" / "public_hard_negatives.csv"
MAX_SIDE = 512

RVL_REPO = "jinhybr/rvl_cdip_400_train_val_test"
RVL_TRAIN_FILES = (
    "data/train-00000-of-00002-fdc52038be08b4c4.parquet",
    "data/train-00001-of-00002-f58eaf6bb83fd3ad.parquet",
)
RVL_LABELS = {
    5: "scientific_report",
    6: "scientific_publication",
    7: "specification",
    9: "news_article",
    10: "budget",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}

HIERTEXT_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/hiertext/"
    "main/gt/validation.jsonl.gz"
)
HIERTEXT_IMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz"
RECEIPT_TERMS = {
    "amount due",
    "balance due",
    "cashier",
    "change due",
    "grand total",
    "invoice",
    "invoice no",
    "payment method",
    "receipt",
    "subtotal",
    "tax invoice",
    "total due",
    "transaction id",
}


def split_for(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def save_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(destination, format="JPEG", quality=90)


def download_file(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(temporary, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    temporary.replace(destination)
    return destination


def import_rvl(cache_dir: Path, per_class: int) -> list[dict[str, str]]:
    counts: Counter[int] = Counter()
    rows: list[dict[str, str]] = []
    for filename in RVL_TRAIN_FILES:
        parquet_path = hf_hub_download(
            repo_id=RVL_REPO,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=64, columns=["image", "label"]):
            for record in batch.to_pylist():
                label = int(record["label"])
                if label not in RVL_LABELS or counts[label] >= per_class:
                    continue
                image_record = record["image"]
                payload = image_record.get("bytes") if isinstance(image_record, dict) else None
                if not payload:
                    continue
                digest = hashlib.sha256(payload).hexdigest()[:16]
                relative = Path("data/raw/public_hard_negatives/rvl") / f"{RVL_LABELS[label]}_{digest}.jpg"
                destination = REPO_ROOT / relative
                if not destination.exists():
                    save_image(Image.open(io.BytesIO(payload)), destination)
                source_doc = f"rvl:{RVL_LABELS[label]}:{digest}"
                rows.append(
                    {
                        "filepath": relative.as_posix(),
                        "source": f"rvl_targeted_{RVL_LABELS[label]}",
                        "split": split_for(source_doc),
                        "source_doc": source_doc,
                    }
                )
                counts[label] += 1
            if all(counts[label] >= per_class for label in RVL_LABELS):
                break
        if all(counts[label] >= per_class for label in RVL_LABELS):
            break

    missing = {RVL_LABELS[label]: per_class - counts[label] for label in RVL_LABELS if counts[label] < per_class}
    if missing:
        raise RuntimeError(f"RVL-CDIP did not provide the requested samples: {missing}")
    print(f"Imported {len(rows)} targeted RVL-CDIP negatives: {dict(counts)}")
    return rows


def annotation_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    if "annotations" in value:
        return value["annotations"]
    if "image_id" in value:
        return [value]
    return []


def annotation_text(record: dict) -> str:
    words: list[str] = []
    for paragraph in record.get("paragraphs", []):
        for line in paragraph.get("lines", []):
            text = line.get("text")
            if text:
                words.append(text)
            else:
                words.extend(word.get("text", "") for word in line.get("words", []))
    return " ".join(words).lower()


def import_hiertext(cache_dir: Path, requested: int, seed: int) -> list[dict[str, str]]:
    annotation_path = download_file(HIERTEXT_ANNOTATIONS_URL, cache_dir / "hiertext_validation.jsonl.gz")
    archive_path = download_file(HIERTEXT_IMAGES_URL, cache_dir / "hiertext_validation.tgz")

    eligible: list[str] = []
    for record in annotation_records(annotation_path):
        text = annotation_text(record)
        if len(text.split()) < 20:
            continue
        if any(term in text for term in RECEIPT_TERMS):
            continue
        eligible.append(str(record["image_id"]))
    random.Random(seed).shuffle(eligible)
    selected = set(eligible[:requested])
    if len(selected) < requested:
        raise RuntimeError(f"Only {len(selected)} eligible HierText images for requested {requested}")

    rows: list[dict[str, str]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            image_id = Path(member.name).stem
            if image_id not in selected:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            relative = Path("data/raw/public_hard_negatives/hiertext") / f"{image_id}.jpg"
            destination = REPO_ROOT / relative
            if not destination.exists():
                save_image(Image.open(extracted), destination)
            source_doc = f"hiertext:{image_id}"
            rows.append(
                {
                    "filepath": relative.as_posix(),
                    "source": "hiertext_text_scene",
                    "split": split_for(source_doc),
                    "source_doc": source_doc,
                }
            )

    if len(rows) != requested:
        raise RuntimeError(f"Extracted {len(rows)}/{requested} selected HierText images")
    print(f"Imported {len(rows)} HierText negatives")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rvl-per-class", type=int, default=150)
    parser.add_argument("--hiertext", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/bill-classifier-public-data"))
    args = parser.parse_args()

    rows = import_rvl(args.cache_dir, args.rvl_per_class)
    rows.extend(import_hiertext(args.cache_dir, args.hiertext, args.seed))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["filepath", "source", "split", "source_doc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} public hard-negative rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
