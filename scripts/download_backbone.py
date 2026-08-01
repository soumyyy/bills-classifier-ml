"""Pre-fetch ImageNet-pretrained MobileNetV3-Small weights for the student
classifier (Step 2), so training doesn't stall on a first-run download.
"""

import tensorflow as tf


def main() -> None:
    print("Downloading MobileNetV3Small ImageNet weights via tf.keras.applications ...")
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    print(f"Loaded backbone: {model.name}, {model.count_params():,} params")
    print("MobileNetV3-Small weights cached.")


if __name__ == "__main__":
    main()
