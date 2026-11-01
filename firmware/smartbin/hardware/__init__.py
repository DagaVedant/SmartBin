from __future__ import annotations

from typing import Protocol

import numpy as np

Frame = np.ndarray  # HxWx3 uint8, RGB


class Trigger(Protocol):
    last_weight_g: float

    def wait_for_item(self, timeout: float | None = None) -> bool: ...

    def rearm(self) -> None: ...


class Camera(Protocol):
    def capture(self) -> Frame: ...

    def close(self) -> None: ...


class LoadCell(Protocol):
    def read_grams(self) -> float: ...

    def tare(self) -> None: ...


class Pan(Protocol):
    def tilt(self, destination: str) -> None: ...

    def hold_at_trash(self) -> None: ...

    def level(self) -> None: ...
