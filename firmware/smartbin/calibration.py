from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ServoCalibration:
    center_us: float = 1500.0
    us_per_deg: float = 11.1
    invert: bool = False


@dataclass
class ScaleCalibration:
    offset: float = 0.0
    grams_per_count: float = 1.0


def load_servo(path: Path) -> ServoCalibration:
    return ServoCalibration(**_read(path)) if path.exists() else ServoCalibration()


def save_servo(path: Path, calibration: ServoCalibration) -> None:
    _write(path, asdict(calibration))


def load_scale(path: Path) -> ScaleCalibration:
    return ScaleCalibration(**_read(path)) if path.exists() else ScaleCalibration()


def save_scale(path: Path, calibration: ScaleCalibration) -> None:
    _write(path, asdict(calibration))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
