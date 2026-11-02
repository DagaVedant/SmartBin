from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .classes import Bin, WasteClass


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Jurisdiction:
    name: str
    mapping: dict[WasteClass, Bin]
    confidence_threshold: float

    def bin_for(self, waste_class: WasteClass) -> Bin:
        return self.mapping.get(waste_class, Bin.TRASH)


@dataclass(frozen=True)
class PanSettings:
    tilt_angle_deg: float = 30.0
    settle_s: float = 0.4
    jam_retries: int = 2
    jam_residual_g: float = 5.0


@dataclass(frozen=True)
class TriggerSettings:
    threshold_g: float = 8.0
    settle_tolerance_g: float = 2.0


@dataclass(frozen=True)
class DeviceConfig:
    jurisdiction_path: Path
    model_path: Path
    database_path: Path
    captures_dir: Path
    api_host: str
    api_port: int
    pan: PanSettings
    trigger: TriggerSettings
    pins: dict[str, int]
    servo_calibration_path: Path
    scale_calibration_path: Path


def load_jurisdiction(path: Path) -> Jurisdiction:
    raw = _read_yaml(path)

    name = raw.get("jurisdiction")
    if not name:
        raise ConfigError(f"{path}: missing 'jurisdiction'")

    mapping: dict[WasteClass, Bin] = {}
    for key, value in (raw.get("mapping") or {}).items():
        try:
            waste_class = WasteClass(key)
        except ValueError as exc:
            raise ConfigError(f"{path}: unknown class {key!r}") from exc
        try:
            mapping[waste_class] = Bin(value)
        except ValueError as exc:
            raise ConfigError(f"{path}: unknown bin {value!r} for {key!r}") from exc

    missing = set(WasteClass) - set(mapping)
    if missing:
        raise ConfigError(f"{path}: no mapping for {sorted(c.value for c in missing)}")

    if mapping[WasteClass.NO_OBJECT] is Bin.RECYCLE:
        raise ConfigError(f"{path}: 'no_object' may never map to 'recycle'")

    threshold = raw.get("confidence_threshold")
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ConfigError(f"{path}: 'confidence_threshold' must be a number")
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError(f"{path}: 'confidence_threshold' must be in [0, 1]")

    return Jurisdiction(name=name, mapping=mapping, confidence_threshold=float(threshold))


def load_device_config(path: Path) -> DeviceConfig:
    raw = _read_yaml(path)
    root = path.parent

    storage = raw.get("storage") or {}
    api = raw.get("api") or {}
    pan = raw.get("pan") or {}
    trigger = raw.get("trigger") or {}
    calibration = raw.get("calibration") or {}

    return DeviceConfig(
        jurisdiction_path=_resolve(root, raw, "jurisdiction", path),
        model_path=_resolve(root, raw, "model", path),
        database_path=root / storage.get("database", "data/events.db"),
        captures_dir=root / storage.get("captures", "captures"),
        api_host=api.get("host", "0.0.0.0"),
        api_port=int(api.get("port", 8000)),
        pan=PanSettings(
            tilt_angle_deg=float(pan.get("tilt_angle_deg", 30.0)),
            settle_s=float(pan.get("settle_ms", 400)) / 1000.0,
            jam_retries=int(pan.get("jam_retries", 2)),
            jam_residual_g=float(pan.get("jam_residual_g", 5.0)),
        ),
        trigger=TriggerSettings(
            threshold_g=float(trigger.get("threshold_g", 8.0)),
            settle_tolerance_g=float(trigger.get("settle_tolerance_g", 2.0)),
        ),
        pins={key: int(value) for key, value in (raw.get("pins") or {}).items()},
        servo_calibration_path=root / calibration.get("servo", "servo_cal.json"),
        scale_calibration_path=root / calibration.get("scale", "scale_cal.json"),
    )


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return raw


def _resolve(root: Path, raw: dict, key: str, path: Path) -> Path:
    value = raw.get(key)
    if not value:
        raise ConfigError(f"{path}: missing {key!r}")
    return (root / value).resolve()
