from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "firmware"))
from smartbin.nn import forward, load_network, preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("firmware/models/smartbin.npz"))
    parser.add_argument("--cache", type=Path, default=Path("ml/cache"))
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()

    network = load_network(args.model)
    size = network.input_size
    print(f"model {network.version}, {len(network.layers)} layers, input {size}px")

    for split in ("val", "test_id", "ext_test"):
        x = np.load(args.cache / f"{split}_x.npy", mmap_mode="r")
        y = np.load(args.cache / f"{split}_y.npy")
        rng = np.random.default_rng(0)
        picks = rng.permutation(len(x))[: args.limit]
        hits, spent = 0, 0.0
        for i in picks:
            small = np.asarray(Image.fromarray(np.asarray(x[i])).resize((size, size), Image.BILINEAR))
            start = time.perf_counter()
            probs = forward(network, preprocess(network, small))
            spent += time.perf_counter() - start
            hits += int(probs.argmax() == y[i])
        print(f"{split}: {hits / len(picks):.4f} on {len(picks)} images, {spent / len(picks) * 1000:.0f} ms/image on this machine")


if __name__ == "__main__":
    main()
