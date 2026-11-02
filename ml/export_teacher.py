from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from student import CLASSES


def activation_name(layer) -> str:
    fn = getattr(layer, "activation", None)
    name = getattr(fn, "__name__", str(fn))
    return {"hard_swish": "hard_silu"}.get(name, name)


def convert(model, version: str) -> tuple[dict, list[dict]]:
    names: dict[int, str] = {}
    ops: list[dict] = []
    consumers: dict[str, int] = {}
    arrays: dict[str, np.ndarray] = {}

    def tensor_name(t) -> str:
        return names[id(t)]

    def emit(op: dict, layer, weights=None) -> None:
        ops.append(op)
        for name in op["in"]:
            consumers[name] = consumers.get(name, 0) + 1

    for tensor in model.inputs:
        names[id(tensor)] = "input"

    for layer in model.operations:
        node = layer._inbound_nodes[0]
        outputs = node.output_tensors
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        kind = layer.__class__.__name__
        if kind == "InputLayer":
            names[id(outputs[0])] = "input"
            continue
        inputs = [tensor_name(t) for t in node.input_tensors]
        out_name = layer.name
        if kind == "Dropout":
            names[id(outputs[0])] = inputs[0]
            continue

        names[id(outputs[0])] = out_name
        if kind == "Rescaling":
            scale, offset = float(layer.scale), float(layer.offset)
            emit({"op": "rescale", "in": inputs, "out": out_name, "scale": scale, "offset": offset}, layer)
        elif kind == "ZeroPadding2D":
            pads = [list(p) for p in layer.padding]
            emit({"op": "pad", "in": inputs, "out": out_name, "pads": pads}, layer)
        elif kind in ("Conv2D", "DepthwiseConv2D"):
            w = layer.get_weights()
            bias = w[1] if layer.use_bias else np.zeros(w[0].shape[-1] if kind == "Conv2D" else w[0].shape[2], np.float32)
            weight = w[0]
            if kind == "DepthwiseConv2D":
                weight = weight[:, :, :, 0]
            op = {
                "op": "conv" if kind == "Conv2D" else "dw",
                "in": inputs,
                "out": out_name,
                "stride": int(layer.strides[0]),
                "pad": layer.padding,
                "w": weight.astype(np.float32),
                "b": bias.astype(np.float32),
            }
            emit(op, layer)
            fn = activation_name(layer)
            if fn != "linear":
                emit({"op": "act", "in": [out_name], "out": out_name + "_act", "fn": fn}, layer)
                names[id(outputs[0])] = out_name + "_act"
        elif kind == "BatchNormalization":
            gamma, beta, mean, var = layer.get_weights()
            scale = gamma / np.sqrt(var + layer.epsilon)
            shift = beta - mean * scale
            producer = ops[-1]
            if producer["out"] != inputs[0] or producer["op"] not in ("conv", "dw") or consumers.get(inputs[0], 0) != 0:
                raise ValueError(f"cannot fold {layer.name}")
            producer["w"] = (producer["w"] * scale).astype(np.float32)
            producer["b"] = (producer["b"] * scale + shift).astype(np.float32)
            names[id(outputs[0])] = inputs[0]
        elif kind in ("Activation", "ReLU"):
            if kind == "ReLU":
                fn = "relu6" if layer.max_value == 6 else "relu"
            else:
                fn = activation_name(layer)
            emit({"op": "act", "in": inputs, "out": out_name, "fn": fn}, layer)
        elif kind in ("Add", "Multiply"):
            if len(inputs) == 1:
                constant = [a for a in node.arguments.args if isinstance(a, (int, float))]
                if len(constant) != 1:
                    raise ValueError(f"cannot read the constant for {layer.name}")
                scale, offset = (1.0, float(constant[0])) if kind == "Add" else (float(constant[0]), 0.0)
                emit({"op": "rescale", "in": inputs, "out": out_name, "scale": scale, "offset": offset}, layer)
            else:
                emit({"op": "add" if kind == "Add" else "mul", "in": inputs, "out": out_name}, layer)
        elif kind == "GlobalAveragePooling2D":
            emit({"op": "gap", "in": inputs, "out": out_name, "keepdims": bool(layer.keepdims)}, layer)
        elif kind == "Dense":
            w, b = layer.get_weights()
            emit({"op": "dense", "in": inputs, "out": out_name, "w": w.astype(np.float32), "b": b.astype(np.float32)}, layer)
            if activation_name(layer) == "softmax":
                emit({"op": "softmax", "in": [out_name], "out": out_name + "_softmax"}, layer)
        else:
            raise ValueError(f"unsupported layer {kind} ({layer.name})")

    return {"classes": list(CLASSES), "version": version}, ops


def pack(meta: dict, ops: list[dict], size: int) -> dict:
    arrays = {}
    specs = []
    for index, op in enumerate(ops):
        spec = {k: v for k, v in op.items() if k not in ("w", "b")}
        if "w" in op:
            arrays[f"w{index}"] = op["w"].astype(np.float16)
            arrays[f"b{index}"] = op["b"].astype(np.float32)
        specs.append(spec)
    meta = dict(meta, format="graph", input_size=size, ops=specs)
    arrays["meta"] = np.array(json.dumps(meta))
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("firmware/models/smartbin-teacher.npz"))
    parser.add_argument("--version", default="smartbin-teacher-1")
    args = parser.parse_args()

    from distill import load_teacher

    model = load_teacher(args.teacher)
    meta, ops = convert(model, args.version)
    arrays = pack(meta, ops, 224)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}: {len(ops)} ops, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
