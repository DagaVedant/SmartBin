from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import calibration as cal
from .__main__ import DEFAULT_CONFIG
from .config import load_device_config


def calibrate_servo(config) -> None:
    from .hardware.real import HardwareServo

    path = config.servo_calibration_path
    current = cal.load_servo(path)
    servo = HardwareServo(calibration_path=path, tilt_angle_deg=config.pan.tilt_angle_deg)
    try:
        print(f"Centre the pan by hand-nudging. Start: {current.center_us:.0f} us.")
        print("Type a change in microseconds (+20, -10), or Enter when the pan is level.")
        centre = current.center_us
        servo.set_pulse_us(centre)
        while True:
            answer = input(f"centre {centre:.0f} us > ").strip()
            if not answer:
                break
            centre += float(answer)
            servo.set_pulse_us(centre)
        current.center_us = centre

        test = 30.0
        servo.calibration.center_us = centre
        servo.calibration.invert = False
        servo.move_to(test)
        direction = input(f"Moved +{test:.0f} deg. Did the pan tip toward RECYCLE? [y/n] ").strip()
        current.invert = direction.lower().startswith("n")

        servo.calibration.invert = current.invert
        servo.move_to(test)
        measured = input(f"Measured tilt in degrees (Enter to keep {current.us_per_deg:.1f} us/deg) > ")
        if measured.strip():
            current.us_per_deg = current.us_per_deg * test / float(measured)

        cal.save_servo(path, current)
        print(f"saved {path}: {current}")
    finally:
        servo.level()
        time.sleep(0.5)
        servo.close()


def calibrate_scale(config, grams: float) -> None:
    from .hardware.real import LoadCellHX711

    path = config.scale_calibration_path
    cell = LoadCellHX711(
        config.pins["load_cell_data"], config.pins["load_cell_clock"], calibration_path=None
    )
    try:
        input("Empty the pan, then press Enter to zero the scale. ")
        offset = cell.read_counts()
        input(f"Place the {grams:g} g mass on the pan centre, then press Enter. ")
        loaded = cell.read_counts()
        if loaded == offset:
            raise SystemExit("No change in the reading; check the load cell wiring.")

        calibration = cal.ScaleCalibration(offset=offset, grams_per_count=grams / (loaded - offset))
        cal.save_scale(path, calibration)
        print(f"saved {path}: {calibration}")

        check = LoadCellHX711(
            config.pins["load_cell_data"], config.pins["load_cell_clock"], calibration_path=path
        )
        print(f"With the mass still on, the scale reads {check.read_grams():.1f} g.")
        check.close()
    finally:
        cell.close()


def main() -> int:
    parser = argparse.ArgumentParser(prog="smartbin.calibrate")
    parser.add_argument("target", choices=["servo", "scale"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--grams", type=float, default=47.0, help="reference mass for the scale")
    args = parser.parse_args()

    config = load_device_config(args.config)
    if args.target == "servo":
        calibrate_servo(config)
    else:
        calibrate_scale(config, args.grams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
