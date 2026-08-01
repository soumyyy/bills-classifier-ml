"""Export the trained Keras model to Core ML for on-device inference (Step
3, iOS path).

Since this machine is macOS, coremltools' own `model.predict()` runs the
*actual* Core ML runtime (not a mock), so the "verify it loads and runs"
requirement is checked directly against real predictions, cross-checked
against the Keras model's output on the same images.

Usage:
    python scripts/export_coreml.py
"""

import csv
import random
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import tensorflow as tf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "invoice_classifier.keras"
COREML_PATH = REPO_ROOT / "models" / "invoice_classifier.mlpackage"
IMG_SIZE = 224


class _ServingWrapper(tf.Module):
    """coremltools' TF2 loader requires exactly one concrete function, but
    Keras 3's `model.export()` emits multiple serving signatures -- so
    (like the TFLite export) we sidestep the built-in export path and save
    a plain tf.Module with a single explicit @tf.function signature."""

    def __init__(self, keras_model: tf.keras.Model):
        super().__init__()
        self.model = keras_model

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)])
    def serve(self, x):
        return self.model(x, training=False)


def convert() -> ct.models.MLModel:
    model = tf.keras.models.load_model(MODEL_PATH)
    wrapper = _ServingWrapper(model)

    export_dir = REPO_ROOT / "models" / "_tmp_savedmodel_coreml"
    tf.saved_model.save(wrapper, str(export_dir), signatures={"serving_default": wrapper.serve})

    mlmodel = ct.convert(
        str(export_dir),
        source="tensorflow",
        inputs=[ct.ImageType(name="x", shape=(1, IMG_SIZE, IMG_SIZE, 3), scale=1.0, bias=[0, 0, 0])],
        minimum_deployment_target=ct.target.iOS15,
    )

    shutil.rmtree(export_dir, ignore_errors=True)
    return mlmodel


def verify(mlmodel: ct.models.MLModel, paths: list[Path]) -> None:
    keras_model = tf.keras.models.load_model(MODEL_PATH)
    input_name = mlmodel.get_spec().description.input[0].name

    print(f"\nVerifying CoreML predictions against Keras on {len(paths)} images:")
    for p in paths:
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

        keras_prob = float(keras_model.predict(arr, verbose=0)[0][0])
        coreml_out = mlmodel.predict({input_name: img})
        coreml_prob = float(list(coreml_out.values())[0].ravel()[0])

        diff = abs(keras_prob - coreml_prob)
        flag = "" if diff < 0.02 else "  <-- diff > 0.02"
        print(f"  {p.name:40s} keras={keras_prob:.4f}  coreml={coreml_prob:.4f}{flag}")


def main() -> None:
    mlmodel = convert()
    COREML_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(COREML_PATH))

    size_bytes = int(subprocess.check_output(["du", "-sk", str(COREML_PATH)]).split()[0]) * 1024
    print(f"Saved {COREML_PATH} ({size_bytes / (1024 * 1024):.2f} MB)")

    with open(REPO_ROOT / "data" / "labeled" / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    random.Random(2).shuffle(rows)
    sample = [REPO_ROOT / r["filepath"] for r in rows[:10]]
    verify(mlmodel, sample)


if __name__ == "__main__":
    main()
