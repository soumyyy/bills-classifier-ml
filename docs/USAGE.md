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
then runs each through the `microsoft/dit-base-finetuned-rvlcdip` teacher.
Known source semantics provide the target label (CORD/SROIE are receipts,
RVL-CDIP invoice is positive, and non-invoice/COCO are negative); teacher
predictions are retained as audit metadata. Writes `data/labeled/manifest.csv`.

### 2. Synthesize hard examples

```
python scripts/synthesize_hard_examples.py --n-positive 900 --n-negative 400
```

Composites real document images onto real photo backgrounds with
perspective warp, occlusion, and harsh lighting -- built specifically
because the first trained model failed on real handheld phone photos of
bills. See "Challenges" below. Appends to the same manifest.

### 3. Train

An optional targeted-negative import is available before training:

```
python scripts/import_public_hard_negatives.py
```

It adds 1,950 reproducible negatives: 1,350 samples from nine confusing
RVL-CDIP document classes (reports, specifications, budgets, questionnaires,
presentations, resumes, and similar pages) plus 600 text-heavy natural scenes
from HierText/Open Images. Downloaded pixels stay ignored under `data/raw/`;
the tracked `data/labeled/public_hard_negatives.csv` records source, lineage,
and fixed 80/10/10 splits. See the official [RVL-CDIP dataset page](https://adamharley.com/rvl-cdip/)
and [HierText repository](https://github.com/google-research-datasets/hiertext)
for source details and usage terms.

```
python scripts/train.py --epochs-head 8 --epochs-finetune 12 --batch-size 16 \
  --hard-negative-repeat 20 --public-negative-train-fraction 0.35
```

MobileNetV3-Small, freeze-then-finetune schedule, class-weighted loss,
leakage-safe grouped train/val/test split. Outputs
`models/invoice_classifier.keras`, `logs/training_curves.png`,
`logs/final_metrics.json`.

### 4. Tune the decision threshold

```
python scripts/tune_threshold.py --target-recall 0.99
```

The default 0.5 cutoff is precision-optimized; this app's priority is to
never miss a real bill. The script selects a threshold on validation, prints
the precision/recall tradeoff, then evaluates it once on the untouched test
split. Calibration remains diagnostic; the app's regression-tested operating
point is **0.15** (see `models/README.md`).

### Optional: train the MobileNetV3-Large multiclass verifier

```
python scripts/train_verifier_large.py --epochs-head 8 \
  --epochs-finetune 12 --batch-size 8 \
  --target-validation-recall 0.99 --target-hard-negative-repeat 8
python scripts/export_verifier_large.py
```

This produces a seven-class verifier for a two-stage cascade without changing
the existing high-recall binary gate. Outputs are
`models/invoice_verifier_large.keras`,
`models/invoice_verifier_large.tflite`, `logs/verifier_large_metrics.json`,
and `logs/verifier_large_deployment_metrics.json`. The export script evaluates
the TFLite cascade on the held-out split and all three app regression folders.
Use `--skip-convert` to evaluate an existing export.

### 5. Export for on-device use

```
python scripts/export_tflite.py --threshold 0.15  # -> models/invoice_classifier.tflite
python scripts/export_coreml.py   # -> models/invoice_classifier.mlpackage
```

### 6. Sanity check

```
python scripts/test_inference.py
python scripts/test_inference.py --images path/to/your/photo.jpg
```

## Results

- **Dataset**: 7,317 images (6,000 real across 5 sources + 1,300 synthetic
  hard examples + 17 app-specific hard negatives).
- **Model**: 83.8% test accuracy, 93.4% AUC on a leakage-checked 733-image
  held-out split.
- **At the deployment threshold (0.15)**: TFLite held-out recall is 96.75%,
  precision is 80.12%, and accuracy is 85.13%.
- **TFLite**: 1.05MB, ~1.7ms CPU inference, 93.64% AUC.
- **Core ML**: 1.98MB, predictions verified to match the source Keras
  model closely.
- **App regression folders**: 17/17 real bills detected and 11/17 supplied
  non-bills rejected at 0.15, improved from 7/17 negative rejections.
- **Targeted public-data experiment**: the first candidate improved broad-test
  precision but reduced receipt recall to 88.25%; a source-balanced retry
  reached 93.75%. Both failed the 95% recall release gate, so neither replaced
  the deployed model. Full measurements are in
  `logs/public_hard_negative_experiment.json`.

## Challenges

- **Teacher-label mismatch for receipts.** RVL-CDIP has an invoice class but
  no receipt class, so treating its top class as ground truth mislabeled many
  genuine CORD/SROIE receipts as advertisements or forms. Source semantics now
  provide the training targets, teacher outputs remain audit metadata, and 818
  existing targets plus their synthetic lineages were repaired.
- **Threshold leakage.** The original deployment cutoff was selected on the
  test set. Threshold calibration now uses validation only and the fixed result
  is evaluated once on the untouched test split.
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

- Text-heavy forms, bill-like medical documents, summaries, and screenshots
  can still be false positives. The three untouched app-specific negative
  holdouts all remain false positives; they must not be folded into training
  merely to make the same regression folder pass.
- The deployment threshold (0.15) still trades precision for recall;
  approximately 2 in 10 positives are false alarms on the broad held-out set.
  Fine for a "flag for review" UX, not for silent auto-filing.
