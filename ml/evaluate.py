import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset import LABELS, load_images, load_manifest, split_by_frame


def predict(model, x):
    return np.argmax(model.predict(x, verbose=0), axis=1)


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def confusion(y_true, y_pred):
    table = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        table[true][pred] += 1
    return table


def print_confusion(table):
    width = max(len(name) for name in LABELS) + 1
    print("actual \\ predicted".rjust(width) + "".join(name[:7].rjust(9) for name in LABELS))
    for i, name in enumerate(LABELS):
        print(name.rjust(width) + "".join(str(value).rjust(9) for value in table[i]))


def recycle_precision(table):
    recycle = [LABELS.index(name) for name in ("pet_bottle", "metal_can", "paper_clean")]
    called = sum(table[t][p] for t in range(len(LABELS)) for p in recycle)
    correct = sum(table[t][p] for t in recycle for p in recycle)
    if called == 0:
        return 0.0
    return correct / called


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--model", default="models/smartbin.keras")
    parser.add_argument("--holdout", default="models/holdout.json")
    parser.add_argument("--images", default="../datasets/raw")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    model = tf.keras.models.load_model(args.model)

    holdout = set(json.loads(Path(args.holdout).read_text()))
    val_rows = [row for row in rows if row["session_id"] in holdout]
    x, y = load_images(val_rows, args.images)
    pred = predict(model, x)

    _, naive_rows = split_by_frame(rows)
    x_naive, y_naive = load_images(naive_rows, args.images)
    naive = accuracy(y_naive, predict(model, x_naive))

    honest = accuracy(y, pred)
    table = confusion(y, pred)

    print("session split (honest):", round(honest, 4))
    print("frame split (naive)   :", round(naive, 4))
    print("optimism gap          :", round(naive - honest, 4))
    print("recycle precision     :", round(recycle_precision(table), 4))
    print()
    print_confusion(table)


if __name__ == "__main__":
    main()
