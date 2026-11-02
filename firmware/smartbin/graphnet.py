from __future__ import annotations

import numpy as np


def same_pads(size: int, k: int, stride: int) -> tuple[int, int]:
    out = -(-size // stride)
    total = max((out - 1) * stride + k - size, 0)
    low = total // 2
    return low, total - low


def _padding(x: np.ndarray, k: int, stride: int, mode: str):
    if mode == "valid":
        return (0, 0), (0, 0)
    return same_pads(x.shape[0], k, stride), same_pads(x.shape[1], k, stride)


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, mode: str) -> np.ndarray:
    k, _, cin, cout = w.shape
    (pt, pb), (pl, pr) = _padding(x, k, stride, mode)
    if k == 1 and not (pt or pb or pl or pr):
        x = x[::stride, ::stride]
        h, wd, _ = x.shape
        out = x.reshape(h * wd, cin) @ w.reshape(cin, cout)
        return out.reshape(h, wd, cout) + b

    padded = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    h = (padded.shape[0] - k) // stride + 1
    wd = (padded.shape[1] - k) // stride + 1
    columns = [
        padded[dy : dy + h * stride : stride, dx : dx + wd * stride : stride, :]
        for dy in range(k)
        for dx in range(k)
    ]
    cols = np.concatenate(columns, axis=2).reshape(h * wd, k * k * cin)
    return (cols @ w.reshape(k * k * cin, cout)).reshape(h, wd, cout) + b


def depthwise(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, mode: str) -> np.ndarray:
    k = w.shape[0]
    (pt, pb), (pl, pr) = _padding(x, k, stride, mode)
    padded = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    h = (padded.shape[0] - k) // stride + 1
    wd = (padded.shape[1] - k) // stride + 1
    out = np.zeros((h, wd, x.shape[2]), dtype=np.float32)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy : dy + h * stride : stride, dx : dx + wd * stride : stride, :] * w[dy, dx]
    return out + b


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(x + 3.0, 0.0, 6.0) / 6.0


def activation(name: str, x: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "relu6":
        return np.clip(x, 0.0, 6.0)
    if name == "hard_sigmoid":
        return hard_sigmoid(x)
    if name == "hard_silu":
        return x * hard_sigmoid(x)
    raise ValueError(f"unknown activation {name!r}")


def run(ops, image: np.ndarray) -> np.ndarray:
    last_use: dict[str, int] = {}
    for index, op in enumerate(ops):
        for name in op["in"]:
            last_use[name] = index

    env = {"input": image}
    result = None
    for index, op in enumerate(ops):
        kind = op["op"]
        args = [env[name] for name in op["in"]]
        x = args[0]
        if kind == "conv":
            out = conv2d(x, op["w"], op["b"], op["stride"], op["pad"])
        elif kind == "dw":
            out = depthwise(x, op["w"], op["b"], op["stride"], op["pad"])
        elif kind == "act":
            out = activation(op["fn"], x)
        elif kind == "pad":
            out = np.pad(x, (tuple(op["pads"][0]), tuple(op["pads"][1]), (0, 0)))
        elif kind == "add":
            out = args[0] + args[1]
        elif kind == "mul":
            out = args[0] * args[1]
        elif kind == "gap":
            out = x.mean(axis=(0, 1), keepdims=bool(op["keepdims"]))
        elif kind == "rescale":
            out = x * np.float32(op["scale"]) + np.float32(op["offset"])
        elif kind == "dense":
            out = x @ op["w"] + op["b"]
        elif kind == "softmax":
            shifted = x - x.max()
            exp = np.exp(shifted)
            out = exp / exp.sum()
        else:
            raise ValueError(f"unknown op {kind!r}")

        env[op["out"]] = out.astype(np.float32, copy=False)
        result = env[op["out"]]
        for name in op["in"]:
            if last_use.get(name) == index and name != "input":
                env.pop(name, None)
    return result
