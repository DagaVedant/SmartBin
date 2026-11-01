from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from .. import Frame
from ...calibration import ScaleCalibration, ServoCalibration, load_scale, load_servo

TRASH_SIGN = -1
RECYCLE_SIGN = 1


class PiCamera3:
    def __init__(
        self, size: tuple[int, int] = (640, 480), lens_position: float | None = None
    ) -> None:
        from picamera2 import Picamera2

        self._camera = Picamera2()
        config = self._camera.create_still_configuration(main={"size": size, "format": "RGB888"})
        self._camera.configure(config)
        self._camera.start()
        if lens_position is not None:
            self._camera.set_controls({"AfMode": 0, "LensPosition": lens_position})
        else:
            self._camera.set_controls({"AfMode": 2})
        time.sleep(1.5)

    def capture(self) -> Frame:
        # picamera2 "RGB888" is BGR in memory; the classifier expects RGB
        return np.ascontiguousarray(self._camera.capture_array()[..., 2::-1])

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()


class LoadCellHX711:
    """HX711 read with lgpio. gpiozero is too slow to keep the clock high under 60 us."""

    MAX_COUNT = 0x7FFFFF

    def __init__(
        self,
        data_pin: int,
        clock_pin: int,
        calibration_path: Path | None = None,
        samples: int = 5,
    ) -> None:
        import lgpio

        self._lgpio = lgpio
        self._chip = lgpio.gpiochip_open(0)
        self._data = data_pin
        self._clock = clock_pin
        lgpio.gpio_claim_input(self._chip, data_pin)
        lgpio.gpio_claim_output(self._chip, clock_pin, 0)
        self._samples = samples
        self._calibration = (
            load_scale(calibration_path) if calibration_path else ScaleCalibration()
        )
        if calibration_path is None or not calibration_path.exists():
            self.tare()

    @property
    def calibration(self) -> ScaleCalibration:
        return self._calibration

    def read_grams(self) -> float:
        counts = self.read_counts()
        return (counts - self._calibration.offset) * self._calibration.grams_per_count

    def tare(self) -> None:
        self._calibration.offset = self.read_counts()

    def read_counts(self) -> float:
        """Median of several raw samples; a sample is dropped if the clock glitched."""
        readings = []
        for _ in range(self._samples * 2):
            value = self._read_raw()
            if value is not None:
                readings.append(value)
            if len(readings) == self._samples:
                break
        if not readings:
            raise RuntimeError("HX711 did not return a valid reading (check wiring and 5V)")
        return float(statistics.median(readings))

    def _read_raw(self) -> int | None:
        lgpio, chip = self._lgpio, self._chip

        deadline = time.monotonic() + 1.0
        while lgpio.gpio_read(chip, self._data):
            if time.monotonic() > deadline:
                return None
            time.sleep(0.001)

        value = 0
        for _ in range(24):
            lgpio.gpio_write(chip, self._clock, 1)
            value = (value << 1) | lgpio.gpio_read(chip, self._data)
            lgpio.gpio_write(chip, self._clock, 0)

        lgpio.gpio_write(chip, self._clock, 1)  # 25th pulse: channel A, gain 128
        lgpio.gpio_write(chip, self._clock, 0)

        if value & 0x800000:
            value -= 0x1000000
        if value in (-1, 0x7FFFFF, -0x800000):  # all ones or rails: bad read
            return None
        return value

    def close(self) -> None:
        self._lgpio.gpiochip_close(self._chip)


class HardwareServo:
    """MG90S driven by the Pi's hardware PWM through sysfs, so there is no software jitter.

    Needs `dtoverlay=pwm,pin=12,func=4` in /boot/firmware/config.txt (GPIO12 is
    PWM0 channel 0) and root, because /sys/class/pwm is root-only.
    """

    PERIOD_NS = 20_000_000
    MIN_US = 600.0
    MAX_US = 2400.0

    def __init__(
        self,
        calibration_path: Path | None = None,
        tilt_angle_deg: float = 30.0,
        sweep_s: float = 0.6,
        hold_s: float = 0.8,
        chip: int = 0,
        channel: int = 0,
    ) -> None:
        self._calibration: ServoCalibration = (
            load_servo(calibration_path) if calibration_path else ServoCalibration()
        )
        self._tilt_angle = tilt_angle_deg
        self._sweep_s = sweep_s
        self._hold_s = hold_s
        self._root = Path(f"/sys/class/pwm/pwmchip{chip}")
        self._pwm = self._root / f"pwm{channel}"
        self._current_us = self._calibration.center_us
        self._open(channel)
        self.level()

    @property
    def calibration(self) -> ServoCalibration:
        return self._calibration

    def _open(self, channel: int) -> None:
        if not self._root.exists():
            raise RuntimeError(
                f"{self._root} is missing: add 'dtoverlay=pwm,pin=12,func=4' to "
                "/boot/firmware/config.txt and reboot"
            )
        if not self._pwm.exists():
            (self._root / "export").write_text(str(channel))
            for _ in range(50):
                if self._pwm.exists():
                    break
                time.sleep(0.02)
        (self._pwm / "enable").write_text("0")
        (self._pwm / "period").write_text(str(self.PERIOD_NS))
        (self._pwm / "duty_cycle").write_text(str(int(self._current_us * 1000)))
        (self._pwm / "enable").write_text("1")

    def set_pulse_us(self, microseconds: float) -> None:
        clamped = min(max(microseconds, self.MIN_US), self.MAX_US)
        (self._pwm / "duty_cycle").write_text(str(int(clamped * 1000)))
        self._current_us = clamped

    def angle_to_us(self, degrees: float) -> float:
        sign = -1.0 if self._calibration.invert else 1.0
        return self._calibration.center_us + sign * degrees * self._calibration.us_per_deg

    def set_angle(self, degrees: float) -> None:
        self.set_pulse_us(self.angle_to_us(degrees))

    def move_to(self, degrees: float, duration_s: float | None = None) -> None:
        """Ease to the target in small steps instead of jumping, which jerks the pan."""
        duration = self._sweep_s if duration_s is None else duration_s
        start, end = self._current_us, self.angle_to_us(degrees)
        steps = max(int(duration / 0.02), 1)
        for step in range(1, steps + 1):
            t = step / steps
            eased = t * t * (3 - 2 * t)
            self.set_pulse_us(start + (end - start) * eased)
            time.sleep(duration / steps)

    def tilt(self, destination: str) -> None:
        sign = RECYCLE_SIGN if destination == "recycle" else TRASH_SIGN
        self.move_to(sign * self._tilt_angle)
        time.sleep(self._hold_s)
        self.level()

    def hold_at_trash(self) -> None:
        self.move_to(TRASH_SIGN * self._tilt_angle)

    def level(self) -> None:
        self.move_to(0.0)

    def close(self) -> None:
        (self._pwm / "enable").write_text("0")
