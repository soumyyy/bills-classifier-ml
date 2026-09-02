"""Download a compact, episode-disjoint AITW screenshot-negative sample.

The upstream Android-in-the-Wild WebShopping release is stored as many small
gzip TFRecord shards.  This script streams shards one at a time, extracts only
the screenshots we need, deletes each temporary shard, exact-deduplicates the
images, and assigns an episode to exactly one split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import tensorflow as tf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "screenshot_negatives_aitw"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
BUCKET = "gresearch"
PREFIX = "android-in-the-wild/web_shopping/"
FIELDS = (
    "filepath",
    "split",
    "source",
    "source_object",
    "episode_id",
    "step_id",
    "current_activity",
    "goal",
    "sha256",
    "width",
    "height",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--max-shard-mb", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def list_objects() -> list[dict]:
    objects = []
    page_token = None
    while True:
        query = {"prefix": PREFIX, "maxResults": 1000}
        if page_token:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
            + urllib.parse.urlencode(query)
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        objects.extend(page.get("items", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            return objects


def feature_bytes(example: tf.train.Example, name: str) -> bytes:
    return example.features.feature[name].bytes_list.value[0]


def feature_int(example: tf.train.Example, name: str) -> int:
    return int(example.features.feature[name].int64_list.value[0])


def decode_text(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").replace("\x00", "")


def episode_split(episode_id: str) -> str:
    bucket = int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def save_image(
    blob: bytes, width: int, height: int, channels: int, destination_stem: Path
) -> Path | None:
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        destination = destination_stem.with_suffix(".png")
        destination.write_bytes(blob)
        return destination
    if blob.startswith(b"\xff\xd8\xff"):
        destination = destination_stem.with_suffix(".jpg")
        destination.write_bytes(blob)
        return destination
    expected = width * height * channels
    if len(blob) != expected or channels not in (1, 3, 4):
        return None
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
    image = Image.frombytes(mode, (width, height), blob)
    if mode != "RGB":
        image = image.convert("RGB")
    destination = destination_stem.with_suffix(".jpg")
    image.save(destination, format="JPEG", quality=90, optimize=True)
    return destination


def existing_rows(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with open(manifest_path, newline="") as source:
        return list(csv.DictReader(source))


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    with open(manifest_path, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def download_object(name: str, destination: Path) -> None:
    quoted = urllib.parse.quote(name, safe="/")
    url = f"https://storage.googleapis.com/{BUCKET}/{quoted}"
    request = urllib.request.Request(url, headers={"User-Agent": "bill-classifier/1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        with open(destination, "wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    rows = existing_rows(manifest_path)
    seen_digests = {row["sha256"] for row in rows}
    seen_objects = {row["source_object"] for row in rows}
    split_targets = {
        "train": round(args.target * 0.70),
        "val": round(args.target * 0.15),
        "test": args.target - round(args.target * 0.70) - round(args.target * 0.15),
    }
    counts = {split: sum(row["split"] == split for row in rows) for split in split_targets}
    print(f"Existing rows={len(rows)} counts={counts} targets={split_targets}")

    objects = [
        item
        for item in list_objects()
        if item["name"] != PREFIX
        and 21 < int(item.get("size", 0)) <= args.max_shard_mb * 1024 * 1024
        and item["name"] not in seen_objects
    ]
    random.Random(args.seed).shuffle(objects)
    print(f"Eligible upstream shards={len(objects)}")

    for shard_index, item in enumerate(objects, start=1):
        if all(counts[key] >= split_targets[key] for key in split_targets):
            break
        name = item["name"]
        size_mb = int(item["size"]) / (1024 * 1024)
        with tempfile.NamedTemporaryFile(prefix="aitw-", suffix=".tfrecord") as shard:
            print(
                f"[{shard_index}/{len(objects)}] download {name.rsplit('/', 1)[-1]} "
                f"{size_mb:.1f}MB rows={len(rows)}"
            )
            download_object(name, Path(shard.name))
            dataset = tf.data.TFRecordDataset([shard.name], compression_type="GZIP")
            added = 0
            for serialized in dataset.as_numpy_iterator():
                example = tf.train.Example.FromString(serialized)
                episode_id = decode_text(feature_bytes(example, "episode_id"))
                split = episode_split(episode_id)
                if counts[split] >= split_targets[split]:
                    continue
                blob = feature_bytes(example, "image/encoded")
                digest = hashlib.sha256(blob).hexdigest()
                if digest in seen_digests:
                    continue
                width = feature_int(example, "image/width")
                height = feature_int(example, "image/height")
                channels = feature_int(example, "image/channels")
                destination = save_image(
                    blob, width, height, channels, output_dir / digest[:24]
                )
                if destination is None:
                    continue
                relative = destination.relative_to(REPO_ROOT)
                row = {
                    "filepath": str(relative),
                    "split": split,
                    "source": "google_android_in_the_wild_web_shopping",
                    "source_object": name,
                    "episode_id": episode_id,
                    "step_id": feature_int(example, "step_id"),
                    "current_activity": decode_text(feature_bytes(example, "current_activity")),
                    "goal": decode_text(feature_bytes(example, "goal_info")),
                    "sha256": digest,
                    "width": width,
                    "height": height,
                }
                rows.append(row)
                seen_digests.add(digest)
                counts[split] += 1
                added += 1
            print(f"  extracted={added} counts={counts}")
            write_manifest(manifest_path, rows)

    if not all(counts[key] >= split_targets[key] for key in split_targets):
        raise RuntimeError(f"Could not reach requested split targets: {counts}")
    print(f"Saved {len(rows)} exact-deduplicated screenshots to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
