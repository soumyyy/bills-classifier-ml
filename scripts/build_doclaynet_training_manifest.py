"""Add mined DocLayNet train negatives without contaminating its holdouts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = REPO_ROOT / "data/labeled/verifier_v2_screenshot_balanced_splits.csv"
DEFAULT_MINED = REPO_ROOT / "data/labeled/doclaynet_mined_splits.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/labeled/verifier_v3_doclaynet_splits.csv"
DEFAULT_REPORT = REPO_ROOT / "logs/experiments/doclaynet_training_manifest.json"
FIELDS = ["filepath", "class_index", "class_name", "split", "source_doc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mined", type=Path, default=DEFAULT_MINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def counts(rows: list[dict[str, str]]) -> dict[str, int]:
    values = Counter((row["split"], row["class_name"]) for row in rows)
    return {
        f"{split}/{class_name}": count
        for (split, class_name), count in sorted(values.items())
    }


def main() -> None:
    args = parse_args()
    base = read_csv(args.base)
    mined = read_csv(args.mined)
    additions = [
        {field: row[field] for field in FIELDS}
        for row in mined
        if row["split"] == "train"
    ]
    if not additions:
        raise ValueError("No DocLayNet training rows found")
    if any(row["selection"] == "external_holdout" for row in mined if row["split"] == "train"):
        raise ValueError("An external holdout was marked for training")

    base_paths = {row["filepath"] for row in base}
    overlap = sorted(base_paths.intersection(row["filepath"] for row in additions))
    if overlap:
        raise ValueError(f"DocLayNet paths already exist in the base manifest: {overlap[:3]}")

    combined = [{field: row[field] for field in FIELDS} for row in base] + additions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined)

    report = {
        "base_manifest": str(args.base),
        "mined_manifest": str(args.mined),
        "output_manifest": str(args.output),
        "base_rows": len(base),
        "doclaynet_train_added": len(additions),
        "combined_rows": len(combined),
        "doclaynet_external_holdout_rows": sum(
            row["split"] in {"val", "test"} for row in mined
        ),
        "base_counts": counts(base),
        "combined_counts": counts(combined),
        "holdout_policy": (
            "DocLayNet validation and test pages are excluded from this training manifest."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
