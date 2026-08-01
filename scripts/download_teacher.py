"""Download and cache the teacher model used for auto-labeling (Step 1).

Model: microsoft/dit-base-finetuned-rvlcdip
Fine-tuned on RVL-CDIP (16 document classes, including "invoice").
"""

import argparse
import sys

from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "microsoft/dit-base-finetuned-rvlcdip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional custom Hugging Face cache directory.",
    )
    args = parser.parse_args()

    print(f"Downloading processor + weights for {MODEL_NAME} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    model.eval()

    num_labels = model.config.num_labels
    id2label = model.config.id2label
    print(f"Loaded {MODEL_NAME}: {num_labels} classes")
    for idx in sorted(id2label):
        print(f"  {idx}: {id2label[idx]}")

    invoice_ids = [i for i, name in id2label.items() if name.lower() == "invoice"]
    if not invoice_ids:
        print("ERROR: no 'invoice' class found in this model's label set.", file=sys.stderr)
        sys.exit(1)
    print(f"\n'invoice' class id: {invoice_ids[0]}")
    print("Teacher model ready.")


if __name__ == "__main__":
    main()
