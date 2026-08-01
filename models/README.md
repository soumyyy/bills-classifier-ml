# invoice_classifier

Binary invoice/bill image classifier. MobileNetV3-Small backbone (ImageNet
pretrained), sigmoid head, fine-tuned per `scripts/train.py`.

## Input / preprocessing
- Resize to 224x224 RGB.
- Feed raw pixel values in **[0, 255]** as float32 -- do **not** manually
  normalize. `include_preprocessing=True` on the MobileNetV3Small backbone
  bakes the correct rescaling into the model itself.

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
