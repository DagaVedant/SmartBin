import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import load_images, load_manifest, split_by_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--model", default="models/smartbin.keras")
    parser.add_argument("--images", default="../datasets/raw")
    parser.add_argument("--out", default="models/smartbin-int8.tflite")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    train_rows, _, _ = split_by_session(rows)
    x, _ = load_images(train_rows[: args.samples], args.images)

    def representative():
        for image in x:
            yield [np.expand_dims(image.astype(np.float32), axis=0)]

    model = tf.keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    data = converter.convert()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)

    print("wrote", out, round(len(data) / 1024, 1), "KB")
    print("re-run evaluate.py against the quantised model before shipping it")


if __name__ == "__main__":
    main()
