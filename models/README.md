# invoice_classifier

Binary invoice/bill image classifier. MobileNetV3-Small backbone (ImageNet
pretrained), sigmoid head, fine-tuned per `scripts/train.py`.

## Exported files

| File | Format | Size | Use |
|---|---|---|---|
| `invoice_classifier.keras` | Keras 3 | 9.9MB | Source of truth for re-export/fine-tuning |
| `invoice_classifier.tflite` | TFLite, dynamic-range quantized | 1.05MB | Android |
| `invoice_classifier.mlpackage` | Core ML (mlprogram) | 1.98MB | iOS |

**Quantization note:** the TFLite export uses *dynamic-range* quantization
(int8 weights, float32 activations), not full-integer. Full-integer
quantization was tested and rejected -- it collapsed recall from 99.1% to
69.0% on the held-out test set (MobileNetV3's hard-swish/squeeze-excite
blocks are sensitive to naive activation quantization), which directly
violates this app's "never miss a real bill" requirement. Dynamic-range
lands at essentially the same size (~1.1MB either way) while preserving
recall (98.8%) almost exactly. See `scripts/export_tflite.py`'s docstring
and its built-in `evaluate_on_test_split()` check, which fails loudly if a
future retrain/re-export drops recall below 95%.

The Core ML export needed a workaround too: `coremltools` can't convert a
Keras model handed to it directly (Keras 3's `model.export()` emits
multiple serving signatures; coremltools' TF2 loader only accepts one), so
both exports go through a manually wrapped single-signature
`tf.saved_model.save()` instead. See the export scripts for details.

## Input / preprocessing
- Resize to 224x224 RGB.
- Feed raw pixel values in **[0, 255]** as float32 -- do **not** manually
  normalize. `include_preprocessing=True` on the MobileNetV3Small backbone
  bakes the correct rescaling into the model itself.
- **TFLite**: input tensor `[1, 224, 224, 3]` float32, values in `[0, 255]`.
- **Core ML**: input is declared as `ImageType` (name `x`), so pass a
  224x224 RGB image directly (`UIImage`/`CVPixelBuffer`) -- no manual
  array conversion or normalization needed, Core ML handles the resize-to-
  tensor step itself as part of the `ImageType` input.

## Output
- Single sigmoid probability in [0, 1]: `invoice_label = 1` means "is a
  bill/invoice", `0` means "is not".

## Decision threshold: use 0.15, not 0.5

The default 0.5 cutoff is precision-optimized. This app's priority is to
**never miss a real bill** (false negatives are much more costly than false
positives), so the recommended operating point trades precision for recall:

| Threshold | Recall | Precision |
|---|---|---|
| 0.50 (default) | ~81% | ~79% |
| 0.152 | 99.1% | 60.0% |
| 0.107 | 100% | 57.3% |

Measured on a 730-image held-out test split (323 positives), leakage-checked
(synthetic examples grouped with their source document so no near-duplicate
crosses the train/test boundary -- see `scripts/tune_threshold.py`).

**Recommendation: threshold = 0.15.** Missing about 1% of held-out bills
in exchange for far fewer than 0.5's false negatives, while keeping the
false-positive rate manageable (~40%, i.e. roughly 4 in 10 flagged images
are not actually bills -- acceptable for a "flag for review" UX). Rerun
`scripts/tune_threshold.py` after any retrain to reconfirm this number.

## Label mapping
`0` = not invoice/bill, `1` = invoice/bill.
