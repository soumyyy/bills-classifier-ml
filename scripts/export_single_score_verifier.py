"""Export the balanced verifier with its calibrated ensemble as one output.

The trained model keeps both heads in Keras for training and analysis. The
production TFLite artifact folds the calibrated fusion into the graph so native
clients only need to copy one scalar output buffer.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT = REPO_ROOT / "models/experiments/verifier_v2_screenshot_balanced"
DEFAULT_KERAS = DEFAULT_EXPERIMENT / "model.keras"
DEFAULT_TFLITE = DEFAULT_EXPERIMENT / "model_single_score.tflite"
DEFAULT_REFERENCE = DEFAULT_EXPERIMENT / "model.tflite"
DIRECT_WEIGHT = 0.8
CLASS_WEIGHT = 0.2
THRESHOLD = 0.3006436387035705
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
DATALESS_FLAG = 0x40000000


def build_single_score_model(source: tf.keras.Model) -> tf.keras.Model:
    direct = source.get_layer("bill_prob").output
    classes = source.get_layer("document_class").output
    score = tf.keras.layers.Lambda(
        lambda values: DIRECT_WEIGHT * values[0]
        + CLASS_WEIGHT * tf.reduce_sum(values[1][:, :2], axis=1, keepdims=True),
        name="bill_ensemble_score",
        output_shape=(1,),
    )([direct, classes])
    return tf.keras.Model(source.input, score, name="invoice_verifier_single_score")


def export(model: tf.keras.Model, destination: Path) -> bytes:
    temp_root = Path(tempfile.mkdtemp(prefix="bill-verifier-single-score-"))
    saved_model = temp_root / "saved_model"
    try:
        model.export(str(saved_model))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = converter.convert()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(blob)
    return blob


def verify(
    source: tf.keras.Model,
    tflite_path: Path,
    samples: int,
    tolerance: float,
) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or list(inputs[0]["shape"]) != [1, 224, 224, 3]:
        raise RuntimeError(f"Unexpected input contract: {inputs}")
    if len(outputs) != 1 or list(outputs[0]["shape"]) != [1, 1]:
        raise RuntimeError(f"Unexpected output contract: {outputs}")

    random = np.random.default_rng(20260820)
    maximum_error = 0.0
    for _ in range(samples):
        batch = random.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)
        predictions = source(batch, training=False)
        direct = np.asarray(predictions["bill_prob"]).ravel()[0]
        classes = np.asarray(predictions["document_class"])[0]
        expected = DIRECT_WEIGHT * direct + CLASS_WEIGHT * classes[:2].sum()

        interpreter.set_tensor(inputs[0]["index"], batch)
        interpreter.invoke()
        actual = float(interpreter.get_tensor(outputs[0]["index"]).ravel()[0])
        maximum_error = max(maximum_error, abs(float(expected) - actual))

    if maximum_error > tolerance:
        raise RuntimeError(
            f"TFLite score differs from Keras by {maximum_error:.8f}; "
            f"tolerance is {tolerance:.8f}"
        )
    print(f"Verified {samples} samples; maximum score error={maximum_error:.8f}")


def invoke(interpreter: tf.lite.Interpreter, image: np.ndarray) -> list[np.ndarray]:
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], np.expand_dims(image, 0))
    interpreter.invoke()
    return [
        interpreter.get_tensor(detail["index"]).ravel()
        for detail in interpreter.get_output_details()
    ]


def verify_regression(
    production_path: Path,
    reference_path: Path,
    folders: list[Path],
) -> None:
    production = tf.lite.Interpreter(model_path=str(production_path))
    reference = tf.lite.Interpreter(model_path=str(reference_path))
    production.allocate_tensors()
    reference.allocate_tensors()
    checked = 0
    maximum_error = 0.0
    disagreements: list[str] = []

    for folder in folders:
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if getattr(path.stat(), "st_flags", 0) & DATALESS_FLAG:
                print(f"Skipping offloaded regression image: {path}")
                continue
            with Image.open(path) as opened:
                image = np.asarray(
                    opened.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
            reference_outputs = invoke(reference, image)
            direct = next(value for value in reference_outputs if value.size == 1)[0]
            classes = next(value for value in reference_outputs if value.size == 7)
            expected = DIRECT_WEIGHT * direct + CLASS_WEIGHT * classes[:2].sum()
            actual_outputs = invoke(production, image)
            if len(actual_outputs) != 1 or actual_outputs[0].size != 1:
                raise RuntimeError(f"Unexpected production outputs for {path}: {actual_outputs}")
            actual = float(actual_outputs[0][0])
            maximum_error = max(maximum_error, abs(float(expected) - actual))
            if (expected >= THRESHOLD) != (actual >= THRESHOLD):
                disagreements.append(str(path.relative_to(REPO_ROOT)))
            checked += 1

    if disagreements:
        raise RuntimeError(
            f"Single-output export changed {len(disagreements)} decisions: {disagreements}"
        )
    print(
        f"Regression checked {checked} folder images; no decision changes; "
        f"maximum reference TFLite score error={maximum_error:.8f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", type=Path, default=DEFAULT_KERAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--regression-folders",
        type=Path,
        nargs="*",
        default=[REPO_ROOT / "TestPhotos", REPO_ROOT / "Negative", REPO_ROOT / "N2"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = tf.keras.models.load_model(args.keras, compile=False)
    if args.skip_export:
        blob = args.output.read_bytes()
    else:
        production = build_single_score_model(source)
        blob = export(production, args.output)
    verify(source, args.output, args.samples, args.tolerance)
    verify_regression(args.output, args.reference, args.regression_folders)
    print(f"Saved {len(blob)} bytes to {args.output}")


if __name__ == "__main__":
    main()
