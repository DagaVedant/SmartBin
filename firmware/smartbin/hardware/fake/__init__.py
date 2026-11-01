from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .. import Frame


class FakeTrigger:
    def __init__(self, triggers: int | None = None, delay_s: float = 0.0) -> None:
        self.remaining = triggers
        self.delay_s = delay_s
        self.trigger_count = 0
        self.rearm_count = 0
        self.last_weight_g = 0.0

    def wait_for_item(self, timeout: float | None = None) -> bool:
        if self.remaining is not None and self.remaining <= 0:
            return False
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.remaining is not None:
            self.remaining -= 1
        self.trigger_count += 1
        return True

    def rearm(self) -> None:
        self.rearm_count += 1


class FakeCamera:
    def __init__(
        self, frames: Sequence[Frame] | None = None, size: tuple[int, int] = (480, 640)
    ) -> None:
        self.frames = deque(frames or [])
        self.size = size
        self.capture_count = 0
        self.closed = False

    @classmethod
    def from_fixtures(cls, directory: Path) -> FakeCamera:
        from PIL import Image

        paths = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
        return cls(frames)

    def capture(self) -> Frame:
        self.capture_count += 1
        if self.frames:
            return self.frames.popleft()
        height, width = self.size
        return np.full((height, width, 3), 128, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeLoadCell:
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


class FakePan:
    def __init__(
        self,
        load_cell: FakeLoadCell | None = None,
        sweep_s: float = 0.0,
        jam: bool = False,
    ) -> None:
        self.load_cell = load_cell
        self.sweep_s = sweep_s
        self.jam = jam
        self.tilts: list[str] = []
        self.position = "level"

    def tilt(self, destination: str) -> None:
        self.tilts.append(destination)
        if self.sweep_s:
            time.sleep(self.sweep_s)
        if self.load_cell is not None and not self.jam:
            self.load_cell.clear()
        self.position = "level"

    def hold_at_trash(self) -> None:
        self.position = "trash"

    def level(self) -> None:
        self.position = "level"
