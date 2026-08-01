"""Quick end-to-end sanity check for the teacher model, before building the
full dataset pipeline. Runs it on a few real RVL-CDIP sample images (2
invoices, 1 letter) plus a synthetic photo-like negative, and confirms
predictions look sane.
"""

import glob

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "microsoft/dit-base-finetuned-rvlcdip"


def make_photo_like_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(600, 800, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img = img.resize((200, 150)).resize((800, 600))
    return img


def main() -> None:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    id2label = model.config.id2label
    invoice_id = 11

    samples = {}
    for path in sorted(glob.glob("/tmp/sample_*.jpg")):
        samples[path.split("/")[-1]] = Image.open(path).convert("RGB")
    samples["synthetic_photo"] = make_photo_like_image()

    for name, img in samples.items():
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs))
        print(
            f"{name:25s} -> predicted={id2label[top_idx]!r:25s} "
            f"confidence={probs[top_idx]:.3f} "
            f"invoice_prob={probs[invoice_id]:.3f}"
        )


if __name__ == "__main__":
    main()
