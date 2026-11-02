from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from .api import create_app
from .classifier import Classifier, NumpyClassifier
from .config import DeviceConfig, load_device_config, load_jurisdiction
from .pipeline import SortPipeline
from .storage import EventStore

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "device.yaml"


def build_hardware(config: DeviceConfig):
    from .hardware.real import HardwareServo, LoadCellHX711, PiCamera3
    from .hardware.trigger import WeightTrigger

    pins = config.pins
    load_cell = LoadCellHX711(
        pins["load_cell_data"],
        pins["load_cell_clock"],
        calibration_path=config.scale_calibration_path,
    )
    trigger = WeightTrigger(
        load_cell,
        threshold_g=config.trigger.threshold_g,
        settle_tolerance_g=config.trigger.settle_tolerance_g,
    )
    pan = HardwareServo(
        calibration_path=config.servo_calibration_path,
        tilt_angle_deg=config.pan.tilt_angle_deg,
    )
    return trigger, PiCamera3(), load_cell, pan


def build_classifier(config: DeviceConfig) -> Classifier:
    return NumpyClassifier(config.model_path)


def main() -> int:
    parser = argparse.ArgumentParser(prog="smartbin")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_device_config(args.config)
    jurisdiction = load_jurisdiction(config.jurisdiction_path)
    store = EventStore(config.database_path)
    classifier = build_classifier(config)
    trigger, camera, load_cell, pan = build_hardware(config)

    pipeline = SortPipeline(
        trigger=trigger,
        camera=camera,
        load_cell=load_cell,
        pan=pan,
        classifier=classifier,
        jurisdiction=jurisdiction,
        store=store,
        settings=config.pan,
        captures_dir=config.captures_dir,
    )

    if not args.no_api:
        import uvicorn

        app = create_app(store, jurisdiction, classifier.version)
        server = uvicorn.Server(
            uvicorn.Config(app, host=config.api_host, port=config.api_port, log_level="warning")
        )
        threading.Thread(target=server.run, daemon=True).start()

    logging.info("running %s under %s", classifier.version, jurisdiction.name)
    pipeline.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
