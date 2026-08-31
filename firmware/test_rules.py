import json

import pytest

import rules


def good_config():
    return {
        "confidence_threshold": 0.85,
        "wet_paper_grams": 40,
        "rules": {
            "pet_bottle": "recycle",
            "metal_can": "recycle",
            "paper_clean": "recycle",
            "organics": "trash",
            "soiled": "trash",
            "unknown": "trash",
        },
    }


def test_check_accepts_a_good_config():
    assert rules.check(good_config())


def test_unknown_may_not_go_to_recycling():
    config = good_config()
    config["rules"]["unknown"] = "recycle"
    with pytest.raises(ValueError):
        rules.check(config)


def test_every_label_must_have_a_rule():
    config = good_config()
    del config["rules"]["soiled"]
    with pytest.raises(ValueError):
        rules.check(config)


def test_bins_must_be_real_bins():
    config = good_config()
    config["rules"]["organics"] = "compost"
    with pytest.raises(ValueError):
        rules.check(config)


def test_threshold_must_be_a_fraction():
    for bad in (0, -0.1, 1.5, "high", None):
        config = good_config()
        config["confidence_threshold"] = bad
        with pytest.raises(ValueError):
            rules.check(config)


def test_confident_recyclable_goes_to_recycling():
    bin_name, reason = rules.decide(good_config(), "pet_bottle", 0.97, 22.0)
    assert bin_name == "recycle"
    assert reason == "rule"


def test_low_confidence_goes_to_trash():
    bin_name, reason = rules.decide(good_config(), "pet_bottle", 0.5, 22.0)
    assert bin_name == "trash"
    assert "below threshold" in reason


def test_confidence_exactly_at_threshold_is_accepted():
    bin_name, _ = rules.decide(good_config(), "metal_can", 0.85, 30.0)
    assert bin_name == "recycle"


def test_heavy_paper_is_treated_as_wet():
    bin_name, reason = rules.decide(good_config(), "paper_clean", 0.99, 120.0)
    assert bin_name == "trash"
    assert "wet" in reason


def test_light_paper_still_recycles():
    bin_name, _ = rules.decide(good_config(), "paper_clean", 0.99, 12.0)
    assert bin_name == "recycle"


def test_unrecognised_label_goes_to_trash():
    bin_name, reason = rules.decide(good_config(), "batteries", 0.99, 30.0)
    assert bin_name == "trash"
    assert reason == "label not in rules"


def test_shipped_config_passes_its_own_check(tmp_path):
    with open("config.json") as f:
        config = json.load(f)
    assert rules.check(config)
