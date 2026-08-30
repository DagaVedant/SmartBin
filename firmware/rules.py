import json

TRASH = "trash"
RECYCLE = "recycle"
BINS = (TRASH, RECYCLE)

LABELS = ["pet_bottle", "metal_can", "paper_clean", "organics", "soiled", "unknown"]


def check(config):
    rules = config.get("rules")
    if not rules:
        raise ValueError("config has no rules")

    missing = [label for label in LABELS if label not in rules]
    if missing:
        raise ValueError("rules are missing " + ", ".join(missing))

    for label, bin_name in rules.items():
        if bin_name not in BINS:
            raise ValueError(label + " maps to " + str(bin_name) + " which is not a bin")

    if rules["unknown"] != TRASH:
        raise ValueError("unknown must map to trash, it maps to " + rules["unknown"])

    threshold = config.get("confidence_threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    return config


def load(path):
    with open(path) as f:
        return check(json.load(f))


def decide(config, label, confidence, weight_g):
    if label not in config["rules"]:
        return TRASH, "label not in rules"

    if confidence < config["confidence_threshold"]:
        return TRASH, "confidence " + str(round(confidence, 3)) + " below threshold"

    if label == "paper_clean" and weight_g > config["wet_paper_grams"]:
        return TRASH, "paper at " + str(round(weight_g, 1)) + " g is probably wet"

    return config["rules"][label], "rule"
