# Usage & build log

## Setup

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 specifically -- best compatibility across torch/tensorflow/coremltools.
Requires ~3GB for dependencies (torch, tensorflow, coremltools, transformers).

## Pipeline, step by step

### 1. Assemble + auto-label the dataset

```
python scripts/build_dataset.py assemble --target-per-class 3000
python scripts/build_dataset.py label --batch-size 16
```

(`build_dataset.py all` runs both phases in sequence.) Streams images from
5 public sources (RVL-CDIP invoices, CORD receipts, SROIE receipts, other
RVL-CDIP document classes, COCO photos), saves them to `data/raw/pool/`,
then runs each through the `microsoft/dit-base-finetuned-rvlcdip` teacher
model to assign a real `invoice_label` -- not just trusting the source.
Writes `data/labeled/manifest.csv`.

### 2. Synthesize hard examples

```
python scripts/synthesize_hard_examples.py --n-positive 900 --n-negative 400
```

Composites real document images onto real photo backgrounds with
perspective warp, occlusion, and harsh lighting -- built specifically
because the first trained model failed on real handheld phone photos of
bills. See "Challenges" below. Appends to the same manifest.

### 3. Train

```
python scripts/train.py --epochs-head 8 --epochs-finetune 12 --batch-size 16
```

MobileNetV3-Small, freeze-then-finetune schedule, class-weighted loss,
leakage-safe grouped train/val/test split. Outputs
`models/invoice_classifier.keras`, `logs/training_curves.png`,
`logs/final_metrics.json`.

### 4. Tune the decision threshold

```
python scripts/tune_threshold.py --target-recall 1.0
```

The default 0.5 cutoff is precision-optimized; this app's priority is to
never miss a real bill. Prints the precision/recall tradeoff at several
thresholds and recommends one. Current recommendation: **0.15** (see
`models/README.md`).

### 5. Export for on-device use

```
python scripts/export_tflite.py   # -> models/invoice_classifier.tflite
python scripts/export_coreml.py   # -> models/invoice_classifier.mlpackage
```

### 6. Sanity check

```
python scripts/test_inference.py
python scripts/test_inference.py --images path/to/your/photo.jpg
```

## Results

- **Dataset**: 7,300 images (6,000 real across 5 sources + 1,300 synthetic
  hard examples), 44%/56% class balance.
- **Model**: 82% test accuracy, 90.0% AUC on a leakage-checked 730-image
  held-out split.
- **At the deployment threshold (0.15)**: 98.8% recall -- essentially
  never misses a real bill, at the cost of a higher false-positive rate
  (acceptable for a "flag for review" UX).
- **TFLite**: 1.05MB, ~1.5ms CPU inference, recall preserved.
- **Core ML**: 1.98MB, predictions verified to match the source Keras
  model closely.
- **Manual testing on real handheld photos**: 7/7 real bills correctly
  caught; non-bills correctly rejected except for two known failure
  categories (below).

## Challenges

- **Real-world generalization gap.** The first trained model (clean
  scans + well-lit photos only) missed real handheld phone photos of
  bills -- dim lighting, tilt, hand/glass occlusion. Fixed by
  synthesizing hard examples that composite real documents onto photo
  backgrounds with perspective warp, occlusion, and harsh lighting.
- **Train/test leakage.** Synthetic examples are derived from real
  source documents; a plain random split could put a document in train
  and its synthetic derivative in test, inflating test metrics. Fixed
  with lineage tracking + a grouped split (`StratifiedGroupKFold`) so a
  document and all its derivatives always land in the same split.
- **TFLite/Core ML converter bugs.** Both `TFLiteConverter.from_keras_model()`
  and `coremltools.convert()` crash on this TensorFlow 2.16.2/Keras 3
  combination (an MLIR bug, reproducible even on a freshly-built,
  never-serialized model). Worked around by routing through an explicit,
  single-signature `tf.saved_model.save()` instead of the frameworks'
  default conversion bridges.
- **Full-integer quantization silently broke recall.** It shrank the
  TFLite model the same amount as gentler quantization, but collapsed
  recall from 99.1% to 69.0% on the test set -- MobileNetV3's
  hard-swish/squeeze-excite blocks are sensitive to naive int8
  activation quantization. Caught by evaluating on the full test set
  rather than a few spot checks; switched to dynamic-range quantization
  (int8 weights, float32 activations), which preserves recall almost
  exactly.
- **8GB RAM on the training machine.** Shaped most of the pipeline design:
  streaming dataset assembly (never holding the full pool in memory),
  small batch sizes, capped `tf.data` caching, and running the PyTorch
  teacher and TensorFlow student in separate processes rather than
  concurrently.
- **Legacy Hugging Face dataset loaders.** The "obvious" source repos
  (`aharley/rvl_cdip`, several SROIE mirrors) ship as legacy loader
  scripts the current `datasets` library refuses to run. Found
  parquet-based community mirrors instead.

## Known limitations

- Two failure modes found via manual testing, not yet fixed: tabular/grid
  documents (exam papers, mark sheets) and financial-chart screenshots
  both get occasionally misclassified as bills -- both resemble a bill's
  row-of-numbers layout. Worth targeted hard-negative mining if this
  matters for the app's real usage.
- The deployment threshold (0.15) trades precision for recall
  deliberately; roughly 4 in 10 images flagged as "bill" may not be one.
  Fine for a "flag for review" UX, not for silent auto-filing.
