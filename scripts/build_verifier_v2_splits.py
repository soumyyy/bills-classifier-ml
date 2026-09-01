"""Build one leakage-free, cleaned split shared with the binary gate.

The legacy verifier created a second random split, which meant most of its
test rows had already been seen by the binary gate.  This builder preserves
the binary model's immutable group split and only attaches verifier classes.

It also corrects conservative, high-confidence label noise in the RVL invoice
source.  Rows the independent RVL teacher called a non-invoice are moved to
that negative document class, and synthetic descendants inherit the fix.
Six manually reviewed source errors are listed explicitly below.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY_SPLITS = REPO_ROOT / "data" / "labeled" / "splits.csv"
LEGACY_VERIFIER_SPLITS = REPO_ROOT / "data" / "labeled" / "verifier_large_splits.csv"
MANIFEST = REPO_ROOT / "data" / "labeled" / "manifest.csv"
OUTPUT = REPO_ROOT / "data" / "labeled" / "verifier_v2_aligned_splits.csv"

CLASS_NAMES = (
    "receipt",
    "invoice",
    "structured_document",
    "narrative_document",
    "advertisement_presentation",
    "text_scene",
    "natural_photo",
)

STRUCTURED_LABELS = {
    "budget",
    "file folder",
    "form",
    "questionnaire",
    "specification",
}
NARRATIVE_LABELS = {
    "email",
    "handwritten",
    "letter",
    "memo",
    "news article",
    "resume",
    "scientific publication",
    "scientific report",
}
ADVERTISEMENT_LABELS = {"advertisement", "presentation"}

# Reviewed from the lowest-scoring positive tail.  These are letters, a
# registration form, a voucher request, a shipping label, and a financial
# statement—not receipts or invoices for the app's purpose.
MANUAL_RVL_NON_BILLS = {
    "data/raw/pool/pos_rvlcdip_invoice_00622.jpg": "form",
    "data/raw/pool/pos_rvlcdip_invoice_00700.jpg": "letter",
    "data/raw/pool/pos_rvlcdip_invoice_00868.jpg": "letter",
    "data/raw/pool/pos_rvlcdip_invoice_01001.jpg": "form",
    "data/raw/pool/pos_rvlcdip_invoice_01202.jpg": "budget",
    "data/raw/pool/pos_rvlcdip_invoice_01657.jpg": "form",
}


def negative_class(label: str) -> str:
    label = label.strip().lower().replace("_", " ")
    if label in NARRATIVE_LABELS:
        return "narrative_document"
    if label in ADVERTISEMENT_LABELS:
        return "advertisement_presentation"
    if label == "text scene":
        return "text_scene"
    if label == "natural photo":
        return "natural_photo"
    if label in STRUCTURED_LABELS:
        return "structured_document"
    return "structured_document"


def read_rows(path: Path) -> list[dict]:
    with open(path) as source:
        return list(csv.DictReader(source))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--all-target-negatives-in-train",
        action="store_true",
        help=(
            "Production fine-tune mode: train on every known Negative/ example. "
            "Use a separate untouched folder to measure generalization."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_rows(MANIFEST)
    metadata = {row["filepath"]: row for row in manifest_rows}
    legacy_classes = {
        row["filepath"]: row["class_name"] for row in read_rows(LEGACY_VERIFIER_SPLITS)
    }

    cleaned_roots: dict[str, str] = {}
    for row in manifest_rows:
        if row["source"] != "rvlcdip_invoice":
            continue
        reviewed_label = MANUAL_RVL_NON_BILLS.get(row["filepath"])
        predicted = row["predicted_class"].strip().lower()
        if reviewed_label is not None:
            cleaned_roots[row["filepath"]] = negative_class(reviewed_label)
        elif predicted != "invoice":
            cleaned_roots[row["filepath"]] = negative_class(predicted)

    binary_rows = read_rows(BINARY_SPLITS)
    output_rows = []
    changed = []
    for row in binary_rows:
        filepath = row["filepath"]
        split = row["split"]
        if args.all_target_negatives_in_train and filepath.startswith("Negative/"):
            split = "train"
        class_name = legacy_classes[filepath]
        cleaned_class = cleaned_roots.get(filepath)
        manifest_row = metadata.get(filepath)
        if cleaned_class is None and manifest_row is not None:
            cleaned_class = cleaned_roots.get(manifest_row.get("source_doc", ""))
        if cleaned_class is not None and class_name in {"receipt", "invoice"}:
            changed.append((filepath, class_name, cleaned_class))
            class_name = cleaned_class
        output_rows.append(
            {
                "filepath": filepath,
                "class_index": CLASS_NAMES.index(class_name),
                "class_name": class_name,
                "split": split,
                "source_doc": row["source_doc"],
            }
        )

    if len(output_rows) != len(binary_rows):
        raise AssertionError("Aligned split lost rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["filepath", "class_index", "class_name", "split", "source_doc"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} aligned rows to {args.output}")
    print(f"Relabeled {len(changed)} noisy positive rows (including descendants)")
    for split in ("train", "val", "test"):
        counts = Counter(
            row["class_name"] for row in output_rows if row["split"] == split
        )
        bills = sum(counts[name] for name in ("receipt", "invoice"))
        print(f"{split}: n={sum(counts.values())} bills={bills} classes={dict(counts)}")


if __name__ == "__main__":
    main()
