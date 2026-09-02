"""Download a targeted, leakage-safe DocLayNet hard-negative subset.

DocLayNet's official archive is roughly 28 GiB.  This script uses HTTP byte
ranges to read its ZIP central directory, official COCO split metadata, and
only the selected PNG members.  Selection is stratified by document category,
caps pages per upstream document, and retains DocLayNet's official
train/validation/test split.

All selected pages are non-bills.  Financial reports and tenders map to the
``structured_document`` auxiliary class; scientific articles, laws, manuals,
and patents map to ``narrative_document``.  The upstream category remains in
the manifest for auditing and future remapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import struct
import threading
import zipfile
import zlib
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_URL = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.0/DocLayNet_core.zip"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/raw/doclaynet_hard_negatives"
DEFAULT_MANIFEST = REPO_ROOT / "data/labeled/doclaynet_hard_negatives.csv"
OFFICIAL_SOURCE = "https://github.com/DS4SD/DocLayNet"
LICENSE_NAME = "Community Data License Agreement – Permissive – Version 1.0"
LICENSE_URL = "https://github.com/DS4SD/DocLayNet/blob/main/LICENSE"
CATEGORIES = (
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
)
CLASS_MAPPING = {
    "financial_reports": (2, "structured_document"),
    "government_tenders": (2, "structured_document"),
    "scientific_articles": (3, "narrative_document"),
    "laws_and_regulations": (3, "narrative_document"),
    "manuals": (3, "narrative_document"),
    "patents": (3, "narrative_document"),
}
FIELDS = (
    "filepath",
    "split",
    "source_doc",
    "class_index",
    "class_name",
    "bill_label",
    "source",
    "doc_category",
    "collection",
    "doc_name",
    "page_no",
    "upstream_file",
    "sha256",
)


class HTTPRangeReader(io.RawIOBase):
    """A seekable, block-cached HTTP reader suitable for ``zipfile``."""

    def __init__(
        self,
        url: str,
        block_size: int = 2 * 1024 * 1024,
        cached_blocks: int = 32,
        timeout: int = 90,
    ) -> None:
        self.url = url
        self.block_size = block_size
        self.cached_blocks = cached_blocks
        self.timeout = timeout
        self.position = 0
        self.session = requests.Session()
        response = self.session.head(url, timeout=timeout)
        response.raise_for_status()
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("DocLayNet server does not advertise byte-range support")
        self.length = int(response.headers["Content-Length"])
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        print(f"Remote archive: {self.length / (1024**3):.2f} GiB with byte ranges")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.length)
        return self.position

    def _block(self, index: int) -> bytes:
        cached = self.cache.pop(index, None)
        if cached is not None:
            self.cache[index] = cached
            return cached
        start = index * self.block_size
        end = min(start + self.block_size, self.length) - 1
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206 for range {start}-{end}, got {response.status_code}"
            )
        value = response.content
        expected = end - start + 1
        if len(value) != expected:
            raise IOError(f"Short HTTP range: expected {expected}, received {len(value)}")
        self.cache[index] = value
        while len(self.cache) > self.cached_blocks:
            self.cache.popitem(last=False)
        return value

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        pieces = []
        remaining = size
        while remaining:
            block_index = self.position // self.block_size
            block_offset = self.position % self.block_size
            block = self._block(block_index)
            take = min(remaining, len(block) - block_offset)
            pieces.append(block[block_offset : block_offset + take])
            self.position += take
            remaining -= take
        return b"".join(pieces)

    def close(self) -> None:
        self.session.close()
        super().close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-per-category", type=int, default=500)
    parser.add_argument("--val-per-category", type=int, default=100)
    parser.add_argument("--test-per-category", type=int, default=100)
    parser.add_argument("--max-pages-per-document", type=int, default=5)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def member_ending(names: list[str], ending: str) -> str:
    normalized = ending.lstrip("/")
    matches = [name for name in names if name.endswith(normalized)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZIP member ending {ending!r}, found {matches}")
    return matches[0]


def load_split_metadata(archive: zipfile.ZipFile, names: list[str], split: str) -> list[dict]:
    member = member_ending(names, f"/COCO/{split}.json")
    print(f"Reading {member}")
    with archive.open(member) as compressed:
        payload = json.load(io.TextIOWrapper(compressed, encoding="utf-8"))
    images = payload["images"]
    unique_documents = Counter()
    seen = defaultdict(set)
    for record in images:
        if record.get("doc_category") in CATEGORIES:
            seen[record["doc_category"]].add(record["doc_name"])
    unique_documents.update({category: len(documents) for category, documents in seen.items()})
    print(
        f"{split}: {len(images)} official image records; "
        f"unique documents={dict(unique_documents)}"
    )
    return images


def select_records(
    records: list[dict],
    split: str,
    per_category: int,
    seed: int,
    max_pages_per_document: int,
) -> list[dict]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        category = record.get("doc_category")
        if category not in CATEGORIES:
            continue
        grouped[category][record["doc_name"]].append(record)

    selected = []
    for category in CATEGORIES:
        rng = random.Random(f"{seed}:{split}:{category}")
        documents = sorted(grouped[category])
        rng.shuffle(documents)
        pages_by_document = {}
        for document in documents:
            pages = list(grouped[category][document])
            rng.shuffle(pages)
            pages_by_document[document] = pages[:max_pages_per_document]
        category_rows = []
        for page_index in range(max_pages_per_document):
            for document in documents:
                pages = pages_by_document[document]
                if page_index < len(pages):
                    category_rows.append(pages[page_index])
                    if len(category_rows) == per_category:
                        break
            if len(category_rows) == per_category:
                break
        if len(category_rows) < per_category:
            print(
                f"Warning: {split}/{category} yielded {len(category_rows)} pages "
                f"with cap={max_pages_per_document}; requested {per_category}"
            )
        selected.extend(category_rows)
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_THREAD_LOCAL = threading.local()


def thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def range_request(url: str, start: int, end: int, timeout: int = 90) -> bytes:
    response = thread_session().get(
        url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Expected HTTP 206 for {start}-{end}, got {response.status_code}")
    return response.content


def download_zip_member(url: str, info: zipfile.ZipInfo, destination: Path) -> None:
    if destination.exists():
        with Image.open(destination) as image:
            image.verify()
        return

    # Fetch the local header plus the compressed member in one exact-ish
    # request.  Central and local extra fields are normally identical; a 4 KiB
    # allowance covers any local-only field.  A precise fallback handles the
    # rare case where it does not.
    guessed_overhead = 30 + len(info.filename.encode("utf-8")) + len(info.extra) + 4096
    start = info.header_offset
    blob = range_request(
        url, start, start + guessed_overhead + info.compress_size - 1
    )
    if len(blob) < 30:
        raise IOError(f"Short local header for {info.filename}")
    signature, _, flags, compression, _, _, _, _, _, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", blob[:30]
    )
    if signature != 0x04034B50:
        raise zipfile.BadZipFile(f"Bad local header signature for {info.filename}")
    if flags & 0x1:
        raise RuntimeError(f"Encrypted ZIP member is unsupported: {info.filename}")
    data_offset = 30 + name_len + extra_len
    compressed = blob[data_offset : data_offset + info.compress_size]
    if len(compressed) != info.compress_size:
        absolute_start = start + data_offset
        compressed = range_request(
            url, absolute_start, absolute_start + info.compress_size - 1
        )
    if compression == zipfile.ZIP_STORED:
        payload = compressed
    elif compression == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP compression {compression}: {info.filename}")
    if len(payload) != info.file_size:
        raise IOError(
            f"Size mismatch for {info.filename}: expected {info.file_size}, got {len(payload)}"
        )
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    with Image.open(temporary) as image:
        image.verify()
    temporary.replace(destination)


def write_source_note(output_dir: Path, counts: Counter) -> None:
    note = (
        "DocLayNet targeted hard-negative subset\n"
        "========================================\n\n"
        f"Official source: {OFFICIAL_SOURCE}\n"
        f"License: {LICENSE_NAME}\n"
        f"License text: {LICENSE_URL}\n"
        f"Archive: {ARCHIVE_URL}\n\n"
        "Selection: capped pages per upstream document, stratified by official split "
        "and document category. Every image is labeled non-bill.\n\n"
        f"Counts: {dict(counts)}\n"
    )
    (output_dir / "SOURCE.txt").write_text(note)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    limits = {
        "train": args.train_per_category,
        "val": args.val_per_category,
        "test": args.test_per_category,
    }

    with HTTPRangeReader(args.archive_url) as remote:
        with zipfile.ZipFile(remote) as archive:
            names = archive.namelist()
            print(f"ZIP members: {len(names)}")
            image_members = {
                Path(name).name: name
                for name in names
                if "/PNG/" in f"/{name}" and name.lower().endswith(".png")
            }
            print(f"PNG members: {len(image_members)}")
            selected = []
            for split, per_category in limits.items():
                metadata = load_split_metadata(archive, names, split)
                selected.extend(
                    {**record, "split": split}
                    for record in select_records(
                        metadata,
                        split,
                        per_category,
                        args.seed,
                        args.max_pages_per_document,
                    )
                )
            counts = Counter((row["split"], row["doc_category"]) for row in selected)
            print(f"Selected {len(selected)} pages: {dict(counts)}")
            if args.metadata_only:
                return

            tasks = []
            for record in selected:
                upstream_file = record["file_name"]
                member = image_members.get(Path(upstream_file).name)
                if member is None:
                    raise FileNotFoundError(f"PNG member not found: {upstream_file}")
                identity = (
                    f"{record['split']}:{record['doc_category']}:"
                    f"{record['doc_name']}:{record['page_no']}"
                )
                short_hash = hashlib.sha256(identity.encode()).hexdigest()[:20]
                destination = output_dir / f"doclaynet_{record['split']}_{short_hash}.png"
                tasks.append((record, archive.getinfo(member), destination))

            completed = 0
            with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
                futures = {
                    executor.submit(
                        download_zip_member, args.archive_url, info, destination
                    ): destination
                    for _, info, destination in tasks
                }
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed % 100 == 0 or completed == len(tasks):
                        print(f"Downloaded/verified {completed}/{len(tasks)}")

            manifest_rows = []
            for record, info, destination in tasks:
                upstream_file = record["file_name"]
                class_index, class_name = CLASS_MAPPING[record["doc_category"]]
                try:
                    filepath = str(destination.relative_to(REPO_ROOT))
                except ValueError:
                    filepath = str(destination)
                manifest_rows.append(
                    {
                        "filepath": filepath,
                        "split": record["split"],
                        "source_doc": f"doclaynet:{record['doc_name']}",
                        "class_index": class_index,
                        "class_name": class_name,
                        "bill_label": 0,
                        "source": "doclaynet",
                        "doc_category": record["doc_category"],
                        "collection": record["collection"],
                        "doc_name": record["doc_name"],
                        "page_no": record["page_no"],
                        "upstream_file": upstream_file,
                        "sha256": sha256(destination),
                    }
                )

    with args.manifest.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_source_note(output_dir, counts)
    print(f"Saved manifest: {args.manifest} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
