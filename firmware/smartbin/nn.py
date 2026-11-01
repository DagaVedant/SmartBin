from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Layer:
    kind: str
    weight: np.ndarray
    bias: np.ndarray
    stride: int = 1
    relu: bool = True


@dataclass(frozen=True)
class Network:
    classes: tuple[str, ...]
    input_size: int
    mean: np.ndarray
    std: np.ndarray
    layers: tuple[Layer, ...]
    version: str


def load_network(path: Path) -> Network:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    layers = []
    for index, spec in enumerate(meta["layers"]):
        layers.append(
            Layer(
                kind=spec["type"],
                weight=data[f"w{index}"].astype(np.float32),
                bias=data[f"b{index}"].astype(np.float32),
                stride=int(spec.get("stride", 1)),
                relu=bool(spec.get("relu", True)),
            )
        )
    return Network(
        classes=tuple(meta["classes"]),
        input_size=int(meta["input_size"]),
        mean=np.asarray(meta["mean"], dtype=np.float32),
        std=np.asarray(meta["std"], dtype=np.float32),
        layers=tuple(layers),
        version=str(meta.get("version", path.stem)),
    )


def preprocess(network: Network, image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return (x - network.mean) / network.std


def forward(network: Network, x: np.ndarray) -> np.ndarray:
    for layer in network.layers:
        if layer.kind == "conv":
            x = _conv(x, layer)
        elif layer.kind == "dw":
            x = _depthwise(x, layer)
        elif layer.kind == "fc":
            x = x.mean(axis=(0, 1)) @ layer.weight + layer.bias
            continue
        else:
            raise ValueError(f"unknown layer type {layer.kind!r}")
        if layer.relu:
            np.maximum(x, 0.0, out=x)
    return _softmax(x)


def _out_size(size: int, k: int, stride: int) -> int:
    return (size + 2 * (k // 2) - k) // stride + 1


def _conv(x: np.ndarray, layer: Layer) -> np.ndarray:
    k = layer.weight.shape[0]
    cin, cout = layer.weight.shape[2], layer.weight.shape[3]
    s = layer.stride
    if k == 1:
        x = x[::s, ::s]
        h, w, _ = x.shape
        out = x.reshape(h * w, cin) @ layer.weight.reshape(cin, cout)
        return out.reshape(h, w, cout) + layer.bias

    p = k // 2
    padded = np.pad(x, ((p, p), (p, p), (0, 0)))
    h = _out_size(x.shape[0], k, s)
    w = _out_size(x.shape[1], k, s)
    columns = [
        padded[dy : dy + h * s : s, dx : dx + w * s : s, :] for dy in range(k) for dx in range(k)
    ]
    cols = np.concatenate(columns, axis=2).reshape(h * w, k * k * cin)
    out = cols @ layer.weight.reshape(k * k * cin, cout)
    return out.reshape(h, w, cout) + layer.bias


def _depthwise(x: np.ndarray, layer: Layer) -> np.ndarray:
    k = layer.weight.shape[0]
    s = layer.stride
    p = k // 2
    padded = np.pad(x, ((p, p), (p, p), (0, 0)))
    h = _out_size(x.shape[0], k, s)
    w = _out_size(x.shape[1], k, s)
    out = np.zeros((h, w, x.shape[2]), dtype=np.float32)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy : dy + h * s : s, dx : dx + w * s : s, :] * layer.weight[dy, dx]
    return out + layer.bias


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
