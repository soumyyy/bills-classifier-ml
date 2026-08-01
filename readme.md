# bill-classifier

A lightweight on-device binary image classifier that answers one question:
**is this photo a bill/invoice, or not?**

MobileNetV3-Small backbone, trained on an auto-labeled dataset built from
public document datasets plus synthetic "handheld phone photo" examples,
exported to both TFLite (Android) and Core ML (iOS).

## Status: done through export

All four steps of the original spec (see `docs/USAGE.md` for the full
build log) are complete:

| | |
|---|---|
| Dataset | 7,300 images, auto-labeled via `microsoft/dit-base-finetuned-rvlcdip` |
| Model | MobileNetV3-Small, 82% test accuracy / 90.0% AUC |
| Deployment threshold | 0.15 (recall-tuned -- see `models/README.md`) |
| TFLite | `models/invoice_classifier.tflite`, 1.05MB, ~1.5ms CPU inference |
| Core ML | `models/invoice_classifier.mlpackage`, 1.98MB |

## Quick start

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_dataset.py all --target-per-class 3000
python scripts/synthesize_hard_examples.py --n-positive 900 --n-negative 400
python scripts/train.py --epochs-head 8 --epochs-finetune 12
python scripts/tune_threshold.py --target-recall 1.0
python scripts/export_tflite.py
python scripts/export_coreml.py
python scripts/test_inference.py
```

See `docs/USAGE.md` for what each step does, why it's built this way, and
the challenges hit along the way. See `models/README.md` for exact
deployment details (preprocessing, input/output format, threshold).

## Original task spec

The task brief this project was built against is in `docs/SPEC.md`.
