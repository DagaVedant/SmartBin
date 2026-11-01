from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from . import nn
from .classes import WasteClass
from .hardware import Frame

CLASS_ORDER: tuple[WasteClass, ...] = (
    WasteClass.RIGID_PLASTIC,
    WasteClass.SOFT_PLASTIC_FOAM,
    WasteClass.CARDBOARD_PAPER,
    WasteClass.METAL,
    WasteClass.GLASS,
    WasteClass.EWASTE_HAZARD,
    WasteClass.NO_OBJECT,
)

PLAUSIBLE_GRAMS: dict[WasteClass, tuple[float, float]] = {
    WasteClass.RIGID_PLASTIC: (5.0, 700.0),
    WasteClass.SOFT_PLASTIC_FOAM: (1.0, 300.0),
    WasteClass.CARDBOARD_PAPER: (1.0, 400.0),
    WasteClass.METAL: (8.0, 600.0),
    WasteClass.GLASS: (30.0, 1500.0),
    WasteClass.EWASTE_HAZARD: (5.0, 1500.0),
    WasteClass.NO_OBJECT: (0.0, float("inf")),
}

OUT_OF_RANGE_FACTOR = 0.25


class Classifier(Protocol):
    version: str

    def predict(self, frame: Frame, weight_g: float | None = None) -> tuple[WasteClass, float]: ...


class FakeClassifier:
    def __init__(
        self,
        predictions: Sequence[tuple[WasteClass, float]] = (),
        default: tuple[WasteClass, float] = (WasteClass.NO_OBJECT, 0.0),
        version: str = "fake",
    ) -> None:
        self.predictions = deque(predictions)
        self.default = default
        self.version = version

    def predict(self, frame: Frame, weight_g: float | None = None) -> tuple[WasteClass, float]:
        if self.predictions:
            return self.predictions.popleft()
        return self.default


class NumpyClassifier:
    def __init__(self, model_path: Path, use_weight: bool = True) -> None:
        self.network = nn.load_network(model_path)
        if self.network.classes != tuple(str(c) for c in CLASS_ORDER):
            raise ValueError(
                f"{model_path}: model classes {self.network.classes} do not match CLASS_ORDER"
            )
        self.version = self.network.version
        self.use_weight = use_weight

    def predict(self, frame: Frame, weight_g: float | None = None) -> tuple[WasteClass, float]:
        image = prepare_frame(frame, self.network.input_size)
        probabilities = nn.forward(self.network, nn.preprocess(self.network, image))
        if self.use_weight and weight_g is not None:
            probabilities = fuse_weight(probabilities, weight_g)

        index = int(np.argmax(probabilities))
        return CLASS_ORDER[index], float(probabilities[index])


def prepare_frame(frame: Frame, size: int) -> np.ndarray:
    from PIL import Image

    height, width = frame.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = np.ascontiguousarray(frame[top : top + side, left : left + side, :3])
    image = Image.fromarray(square).resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def fuse_weight(probabilities: np.ndarray, weight_g: float) -> np.ndarray:
    adjusted = probabilities.copy()
    for index, waste_class in enumerate(CLASS_ORDER):
        low, high = PLAUSIBLE_GRAMS[waste_class]
        if not low <= weight_g <= high:
            adjusted[index] *= OUT_OF_RANGE_FACTOR

    total = adjusted.sum()
    return adjusted / total if total else probabilities
