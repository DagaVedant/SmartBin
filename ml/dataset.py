import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = ["pet_bottle", "metal_can", "paper_clean", "organics", "soiled", "unknown"]
IMAGE_SIZE = 160
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(str(path) + " is empty")

    for row in rows:
        if row["label"] not in LABELS:
            raise ValueError(str(path) + " has unknown label " + row["label"])

    return rows


def sessions(rows):
    return sorted({row["session_id"] for row in rows})


def split_by_session(rows, holdout_fraction=0.25, seed=0):
    names = sessions(rows)
    count = max(1, round(len(names) * holdout_fraction))
    holdout = set(random.Random(seed).sample(names, count))

    train = [row for row in rows if row["session_id"] not in holdout]
    val = [row for row in rows if row["session_id"] in holdout]
    return train, val, sorted(holdout)


def split_by_frame(rows, holdout_fraction=0.25, seed=0):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - holdout_fraction))
    return shuffled[:cut], shuffled[cut:]


def load_images(rows, root):
    root = Path(root)
    x = np.zeros((len(rows), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    y = np.zeros(len(rows), dtype=np.int64)

    for i, row in enumerate(rows):
        path = root / row["session_id"] / row["filename"]
        image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        x[i] = np.asarray(image)
        y[i] = LABELS.index(row["label"])

    return x, y
