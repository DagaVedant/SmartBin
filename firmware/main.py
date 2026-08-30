import argparse
import signal
import sys
import time

import api
import rules
import storage
from classifier import Classifier
from hardware import Hardware

running = True


def stop(signum, frame):
    global running
    running = False


def sort_once(config, device, model, db):
    device.wait_for_item()
    time.sleep(config["settle_seconds"])

    weight = device.read_weight()

    device.light(True)
    image = device.capture()
    device.light(False)

    label, confidence = model.predict(image)
    bin_name, reason = rules.decide(config, label, confidence, weight)

    device.tilt(bin_name)
    device.level()

    storage.log_event(db, label, confidence, weight, bin_name, reason)
    print(label, round(confidence, 3), str(round(weight, 1)) + "g", "->", bin_name, "(" + reason + ")")
    return bin_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = rules.load(args.config)

    device = Hardware(config)
    model = Classifier(config["model"])
    db = storage.open_db(config["database"])

    api.serve(db, config["api_port"])

    print("api on port", config["api_port"])
    print("threshold:", config["confidence_threshold"], "| anything below goes to trash")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        while running:
            sort_once(config, device, model, db)
            if args.once:
                break
    finally:
        device.close()
        db.close()
        print("stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
