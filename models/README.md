# invoice_classifier

Binary invoice/receipt classifier using a MobileNetV3-Small backbone and a
sigmoid head. The August 2026 model was retrained after correcting 818 noisy
targets: CORD and SROIE are receipt datasets, but the old RVL-CDIP teacher has
no receipt class and had incorrectly labeled many of them as advertisements or
forms. Source ground truth is now the target; teacher predictions remain audit
metadata.

## Exported files

| File | Format | Size | Use |
|---|---|---:|---|
| `invoice_classifier.keras` | Keras 3 | 9.9 MB | Re-export/fine-tuning source |
| `invoice_classifier.tflite` | Dynamic-range quantized TFLite | 1.05 MB | On-device app model |
| `invoice_classifier.mlpackage` | Core ML mlprogram | 1.98 MB | Native iOS integration |

Dynamic-range quantization keeps float activations and int8 weights. Full-int8
activation quantization was rejected because it substantially reduced recall.

## Input and output

- Input: `[1, 224, 224, 3]` float32 RGB in raw `[0, 255]` values.
- Resize directly to 224×224; do not normalize. MobileNetV3 preprocessing is
  embedded in the model.
- Output: one sigmoid probability, where `1` is invoice/receipt and `0` is not.

## Deployment threshold: 0.15

The app uses 0.15 as its operating point after regression testing showed that
the old recall-maximizing 0.054 cutoff admitted too many non-bills. The fixed
0.15 cutoff is evaluated on the untouched test split and on the app-specific
`TestPhotos` and `Negative` folders. Metrics are in
`logs/deployment_metrics.json`.

| Operating point | Accuracy | Precision | Recall | AUC |
|---|---:|---:|---:|---:|
| TFLite, threshold 0.15 | 85.13% | 80.12% | 96.75% | 93.64% |
| Keras, threshold 0.5 | 83.77% | 90.49% | 78.50% | 93.38% |

The app still favors recall because candidates pass through OCR and keyword
validation. Re-run calibration and both app-specific regression folders after
every retrain; do not change the app cutoff from calibration alone.

The current TFLite model detects all 17/17 `TestPhotos` receipts and rejects
11/17 `Negative` images at 0.15 (the prior model rejected 7/17). Three negative
images were reserved as an untouched hard-negative holdout; all three remain
false positives, so broader negative data is still required before claiming
that failure mode is solved.

## Label mapping

`0` = not invoice/receipt, `1` = invoice/receipt.

## Legacy MobileNetV3-Large verifier

`invoice_verifier_large.tflite` is the original second-stage, seven-class
verifier retained for reproducibility. It ran only after the binary model's
0.15 gate, using receipt + invoice probability at threshold
`0.008874409832060335`. The production app no longer bundles this artifact.

The verifier output order is receipt, invoice, structured document, narrative
document, advertisement/presentation, text scene, and natural photo. The
dynamic-range-quantized TFLite artifact is 3,278,952 bytes and averaged 4.51 ms
per inference on the training Mac's CPU.

On the leakage-safe 921-image test split, the cascade reached 84.15% accuracy,
74.29% precision, 97.26% recall, and 95.97% AUC. This improves the same binary
gate's 80.24% accuracy and 69.45% precision while retaining nearly all of its
97.51% recall. App regressions were 16/17 `TestPhotos`, 13/17 `Negative`, and
8/13 `N2`. Because it was too permissive, this model was replaced in the app
by `experiments/verifier_v2_screenshot_balanced/model.tflite`. Full legacy
measurements and per-image results are in
`logs/verifier_large_deployment_metrics.json`.

## Production balanced MobileNetV3-Large

The app now uses
`experiments/verifier_v2_screenshot_balanced/model_single_score.tflite` as a
single-pass candidate filter. It is exported from the trained two-head model,
with the calibrated `0.8 * direct + 0.2 * (receipt + invoice)` fusion baked into
one scalar TFLite output. The threshold remains `0.3006436387035705`. Keeping
one native output buffer avoids the cross-platform inference crash observed
with the otherwise equivalent two-output artifact.

On the 1,747-image combined held-out test, it reached 98.09% recall, 7.25%
false-positive rate, and 86.90% precision. See
`logs/experiments/verifier_v2_screenshot_balanced/calibration_balanced.json`.
