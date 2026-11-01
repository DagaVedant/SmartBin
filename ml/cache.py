from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from student import CLASSES

SIZE = 224


def _load(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("ml/cache"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    index = {name: i for i, name in enumerate(CLASSES)}

    for split, rows in frame.groupby("split"):
        paths = [str(args.root / p) for p in rows.path]
        x = np.lib.format.open_memmap(
            args.out / f"{split}_x.npy", mode="w+", dtype=np.uint8, shape=(len(paths), SIZE, SIZE, 3)
        )
        with ProcessPoolExecutor(args.workers) as pool:
            for i, array in enumerate(pool.map(_load, paths, chunksize=64)):
                x[i] = array
        x.flush()
        np.save(args.out / f"{split}_y.npy", np.array([index[c] for c in rows.cls], dtype=np.int64))
        print(split, len(paths))


if __name__ == "__main__":
    main()
