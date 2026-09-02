"""Append split-preserving screenshot-bill positives to screenshot-negative V2 data."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = REPO_ROOT / "data/labeled/verifier_v2_screenshot_splits.csv"
DEFAULT_POSITIVES = REPO_ROOT / "data/raw/screenshot_positive_bills/manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/labeled/verifier_v2_screenshot_balanced_splits.csv"
FIELDS = ("filepath", "class_index", "class_name", "split", "source_doc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    args = parse_args()
    base_rows = read_rows(args.base)
    positive_manifest = read_rows(args.positives)
    positive_rows = [
        {
            "filepath": row["filepath"],
            "class_index": row["class_index"],
            "class_name": row["class_name"],
            "split": row["split"],
            "source_doc": f"screenshot:{row['context']}:{row['source_doc']}",
        }
        for row in positive_manifest
    ]
    filepaths = [row["filepath"] for row in base_rows + positive_rows]
    if len(filepaths) != len(set(filepaths)):
        raise RuntimeError("Balanced split contains duplicate filepaths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(base_rows + positive_rows)

    for split in ("train", "val", "test"):
        rows = [row for row in base_rows + positive_rows if row["split"] == split]
        counts = Counter(
            "bill" if int(row["class_index"]) in (0, 1) else "not_bill" for row in rows
        )
        screenshot_positive = sum(
            row["filepath"].startswith("data/raw/screenshot_positive_bills/") for row in rows
        )
        screenshot_negative = sum(
            row["filepath"].startswith("data/raw/screenshot_negatives_aitw/") for row in rows
        )
        print(
            f"{split}: n={len(rows)} labels={dict(counts)} "
            f"screenshot_bill={screenshot_positive} screenshot_not_bill={screenshot_negative}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
