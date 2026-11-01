from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

CLASSES = [
    "rigid_plastic", "soft_plastic_foam", "cardboard_paper", "metal",
    "glass", "ewaste_hazard", "no_object",
]
INPUT_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BLOCKS = [(32, 2), (64, 1), (96, 2), (96, 1), (160, 2), (160, 1), (256, 1)]
STEM = 24


def _round(channels: float) -> int:
    return max(8, int(round(channels / 8)) * 8)


def _conv_bn(cin, cout, k, stride, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Student(nn.Module):
    def __init__(self, width: float = 1.5, num_classes: int = len(CLASSES), dropout: float = 0.2):
        super().__init__()
        self.width = width
        stem = _round(STEM * width)
        layers = [_conv_bn(3, stem, 3, 2)]
        cin = stem
        for cout, stride in BLOCKS:
            cout = _round(cout * width)
            layers.append(_conv_bn(cin, cin, 3, stride, groups=cin))
            layers.append(_conv_bn(cin, cout, 1, 1))
            cin = cout
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(2, 3))
        return self.fc(self.dropout(x))


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    bias = bn.bias.detach() - bn.running_mean * scale
    return weight.cpu().numpy(), bias.cpu().numpy()


def export_npz(model: Student, path, version: str) -> None:
    model = model.eval().cpu()
    layers, arrays = [], {}
    for block in model.features:
        conv, bn = block[0], block[1]
        weight, bias = _fold(conv, bn)
        index = len(layers)
        if conv.groups > 1:
            kind = "dw"
            arrays[f"w{index}"] = np.ascontiguousarray(weight[:, 0].transpose(1, 2, 0))
        else:
            kind = "conv"
            arrays[f"w{index}"] = np.ascontiguousarray(weight.transpose(2, 3, 1, 0))
        arrays[f"b{index}"] = bias
        layers.append({"type": kind, "stride": conv.stride[0], "relu": True})

    index = len(layers)
    arrays[f"w{index}"] = model.fc.weight.detach().numpy().T.copy()
    arrays[f"b{index}"] = model.fc.bias.detach().numpy()
    layers.append({"type": "fc", "relu": False})

    meta = {
        "classes": CLASSES, "input_size": INPUT_SIZE, "mean": MEAN, "std": STD,
        "layers": layers, "version": version,
    }
    np.savez_compressed(
        path, meta=np.array(json.dumps(meta)),
        **{k: v.astype(np.float32) for k, v in arrays.items()},
    )
