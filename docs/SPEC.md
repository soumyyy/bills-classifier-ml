# Task: Build a lightweight on-device invoice/bill image classifier

## Goal
Build a binary image classifier ("invoice/bill" vs "not invoice/bill") that runs
on-device on a phone. Deliver: training pipeline, exported TFLite (Android) and
Core ML (iOS) models, and a labeled dataset generated via a teacher model.

## Step 1 — Auto-label a dataset using a teacher model
- Use `microsoft/dit-base-finetuned-rvlcdip` (Hugging Face, transformers library)
  as a teacher model. It's a ViT/BEiT-based model fine-tuned on RVL-CDIP (16
  document classes, including "invoice").
- Write a script that:
  - Takes a folder of images (mixed: documents + non-documents + invoices).
  - Runs each through the teacher model.
  - Labels an image `invoice=1` if predicted class is "invoice", else `invoice=0`.
  - Saves results as a CSV/JSON manifest (filepath, predicted_class, invoice_label,
    confidence score).
- Also write a script to assemble a raw image pool before labeling:
  - Positive-leaning sources: pull sample images from the RVL-CDIP dataset
    (huggingface.co/datasets/aharley/rvl_cdip) filtered to class "invoice",
    plus any receipt/invoice datasets you can find (e.g. CORD, SROIE, FUNSD).
  - Negative sources: other RVL-CDIP classes (letter, form, resume, email, etc.),
    plus generic non-document images (sample from COCO or similar) to cover
    selfies/photos/screenshots as hard negatives.
  - Aim for at least a few thousand images per class after labeling; log final
    class balance.

## Step 2 — Train a lightweight binary classifier
- Architecture: MobileNetV3-Small (or MobileNetV2 as fallback), pretrained on
  ImageNet, binary sigmoid head.
- Framework: PyTorch or TensorFlow/Keras — pick whichever has cleaner conversion
  paths for step 3 (TF/Keras usually simplifies TFLite export).
- Training approach:
  - Freeze backbone, train head first.
  - Then unfreeze last 20-30% of backbone layers, fine-tune at low LR.
  - Use standard augmentation: rotation, brightness/contrast jitter, blur, JPEG
    compression artifacts (to simulate phone camera photos of paper documents).
  - Binary cross-entropy loss, track accuracy/F1/precision/recall on held-out
    val split (80/10/10 train/val/test).
- Output: trained model checkpoint + training curves + final eval metrics
  printed/logged.

## Step 3 — Export for on-device deployment
- Convert to TFLite:
  - Post-training int8 quantization.
  - Verify quantized model size (target: under 3MB) and run a quick inference
    speed benchmark (CPU, no GPU delegate) on a few sample images.
- Convert to Core ML using coremltools:
  - Verify the conversion loads and runs correctly.
- Save both exported models with a short README documenting: input shape/
  preprocessing (resize, normalization values), output format (single sigmoid
  probability), and label mapping.

## Step 4 — Sanity test
- Write a small inference script (Python) that loads the TFLite model and runs
  it on a handful of test images (a few real invoice photos, a few clearly
  non-invoice images) to confirm predictions look sane before calling it done.

## Deliverables
1. `scripts/build_dataset.py` — assembles + auto-labels the image pool
2. `scripts/train.py` — trains the MobileNetV3-Small binary classifier
3. `scripts/export_tflite.py` and `scripts/export_coreml.py`
4. `scripts/test_inference.py`
5. `models/invoice_classifier.tflite` and `models/invoice_classifier.mlmodel`
6. `README.md` documenting dataset composition, final metrics, model size,
   inference latency, and how to plug the model into an iOS/Android app

## Constraints
- Keep the final on-device model under ~3MB after quantization.
- No internet/API calls required at inference time — this must run fully
  offline on-device.
- Prefer well-established libraries (transformers, torch/tensorflow, timm,
  coremltools, tflite) over custom implementations.
