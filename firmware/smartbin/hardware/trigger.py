from __future__ import annotations

import time

from . import LoadCell


class WeightTrigger:

    def __init__(
        self,
        load_cell: LoadCell,
        threshold_g: float = 8.0,
        settle_tolerance_g: float = 2.0,
        poll_s: float = 0.1,
        settle_timeout_s: float = 4.0,
    ) -> None:
        self.load_cell = load_cell
        self.threshold_g = threshold_g
        self.settle_tolerance_g = settle_tolerance_g
        self.poll_s = poll_s
        self.settle_timeout_s = settle_timeout_s
        self.last_weight_g = 0.0

    def wait_for_item(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            if self.load_cell.read_grams() >= self.threshold_g:
                self.last_weight_g = self._settled()
                return True
            time.sleep(self.poll_s)
        return False

    def rearm(self) -> None:
        self.load_cell.tare()

    def _settled(self) -> float:
        deadline = time.monotonic() + self.settle_timeout_s
        previous = self.load_cell.read_grams()
        while time.monotonic() < deadline:
            current = self.load_cell.read_grams()
            if abs(current - previous) <= self.settle_tolerance_g:
                return current
            previous = current
        return previous
