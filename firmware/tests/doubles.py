from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np

from smartbin.classes import WasteClass


class StubTrigger:
    def __init__(self, triggers: int | None = None) -> None:
        self.remaining = triggers
        self.rearm_count = 0
        self.last_weight_g = 0.0

    def wait_for_item(self, timeout: float | None = None) -> bool:
        if self.remaining is not None and self.remaining <= 0:
            return False
        if self.remaining is not None:
            self.remaining -= 1
        return True

    def rearm(self) -> None:
        self.rearm_count += 1


class StubCamera:
    def capture(self) -> np.ndarray:
        return np.full((480, 640, 3), 128, dtype=np.uint8)

    def close(self) -> None:
        pass


class StubLoadCell:
    def __init__(self, weights: Iterable[float] = (), resting: float = 0.0) -> None:
        self.weights = deque(weights)
        self.resting = resting
        self.current = resting

    def read_grams(self) -> float:
        if self.weights:
            self.current = self.weights.popleft()
        return self.current

    def tare(self) -> None:
        self.current = self.resting

    def clear(self) -> None:
        self.current = self.resting
        self.weights.clear()


class StubPan:
    def __init__(self, load_cell: StubLoadCell | None = None, jam: bool = False) -> None:
        self.load_cell = load_cell
        self.jam = jam
        self.tilts: list[str] = []
        self.position = "level"

    def tilt(self, destination: str) -> None:
        self.tilts.append(destination)
        if self.load_cell is not None and not self.jam:
            self.load_cell.clear()
        self.position = "level"

    def hold_at_trash(self) -> None:
        self.position = "trash"

    def level(self) -> None:
        self.position = "level"


class StubClassifier:
    version = "stub"

    def __init__(self, predictions: Sequence[tuple[WasteClass, float]] = ()) -> None:
        self.predictions = deque(predictions)

    def predict(self, frame, weight_g=None) -> tuple[WasteClass, float]:
        if self.predictions:
            return self.predictions.popleft()
        return WasteClass.NO_OBJECT, 0.0
