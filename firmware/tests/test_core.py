import json
from pathlib import Path

import numpy as np
import pytest

from smartbin import nn
from smartbin.classes import Bin, WasteClass
from smartbin.classifier import CLASS_ORDER, FakeClassifier, NumpyClassifier, fuse_weight
from smartbin.config import ConfigError, PanSettings, load_device_config, load_jurisdiction
from smartbin.gate import decide
from smartbin.hardware.fake import FakeCamera, FakeLoadCell, FakePan, FakeTrigger
from smartbin.hardware.trigger import WeightTrigger
from smartbin.pipeline import SortPipeline
from smartbin.storage import EventStore

CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def jurisdiction():
    return load_jurisdiction(CONFIG / "jurisdiction.yaml")


def test_shipped_configs_load(jurisdiction):
    assert jurisdiction.bin_for(WasteClass.GLASS) is Bin.RECYCLE
    assert jurisdiction.bin_for(WasteClass.EWASTE_HAZARD) is Bin.TRASH
    config = load_device_config(CONFIG / "device.yaml")
    assert config.pins["servo"] == 12


def test_no_object_cannot_recycle(tmp_path):
    lines = [f"  {c.value}: trash" for c in WasteClass if c is not WasteClass.NO_OBJECT]
    path = tmp_path / "j.yaml"
    path.write_text(
        "jurisdiction: x\nconfidence_threshold: 0.5\nmapping:\n" + "\n".join(lines)
        + "\n  no_object: recycle\n"
    )
    with pytest.raises(ConfigError):
        load_jurisdiction(path)


def test_low_confidence_goes_to_trash(jurisdiction):
    assert decide(WasteClass.METAL, 0.9, jurisdiction).bin is Bin.RECYCLE
    low = decide(WasteClass.METAL, 0.2, jurisdiction)
    assert low.bin is Bin.TRASH and low.gated


def test_fuse_weight_demotes_implausible():
    probs = np.full(len(CLASS_ORDER), 1 / len(CLASS_ORDER))
    fused = fuse_weight(probs, 900.0)  # too heavy for a cardboard item
    assert fused[CLASS_ORDER.index(WasteClass.CARDBOARD_PAPER)] < probs[0]
    assert fused.sum() == pytest.approx(1.0)


def _tiny_model(path):
    rng = np.random.default_rng(0)
    layers = [
        {"type": "conv", "stride": 2},
        {"type": "dw", "stride": 1},
        {"type": "conv", "stride": 1},
        {"type": "fc", "relu": False},
    ]
    shapes = [(3, 3, 3, 4), (3, 3, 4), (1, 1, 4, 6), (6, len(CLASS_ORDER))]
    arrays = {}
    for i, shape in enumerate(shapes):
        arrays[f"w{i}"] = rng.normal(size=shape).astype(np.float32)
        arrays[f"b{i}"] = rng.normal(size=shape[-1]).astype(np.float32)
    meta = {
        "classes": [str(c) for c in CLASS_ORDER], "input_size": 16, "version": "tiny",
        "mean": [0.5] * 3, "std": [0.5] * 3, "layers": layers,
    }
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)


def test_conv_matches_naive():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 8, 3)).astype(np.float32)
    w = rng.normal(size=(3, 3, 3, 5)).astype(np.float32)
    b = rng.normal(size=5).astype(np.float32)
    layer = nn.Layer("conv", w, b, stride=2, relu=False)
    out = nn._conv(x, layer)
    padded = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    for i in range(4):
        for j in range(4):
            patch = padded[2 * i : 2 * i + 3, 2 * j : 2 * j + 3, :]
            expected = np.einsum("hwc,hwco->o", patch, w) + b
            assert np.allclose(out[i, j], expected, atol=1e-4)


def test_numpy_classifier_runs(tmp_path):
    _tiny_model(tmp_path / "m.npz")
    clf = NumpyClassifier(tmp_path / "m.npz")
    waste_class, confidence = clf.predict(np.full((48, 64, 3), 100, np.uint8), weight_g=50.0)
    assert waste_class in CLASS_ORDER and 0 < confidence <= 1


def test_weight_trigger_waits_for_settle():
    cell = FakeLoadCell(weights=[0, 0, 30, 60, 61, 61, 61])
    trigger = WeightTrigger(cell, threshold_g=8, poll_s=0)
    assert trigger.wait_for_item(timeout=2)
    assert trigger.last_weight_g == pytest.approx(61)
    cell2 = FakeLoadCell()
    assert not WeightTrigger(cell2, poll_s=0).wait_for_item(timeout=0.05)


def _pipeline(tmp_path, jurisdiction, predictions, jam=False):
    cell = FakeLoadCell(resting=0.0)
    pan = FakePan(load_cell=cell, jam=jam)
    store = EventStore(tmp_path / "e.db")
    trigger = FakeTrigger(triggers=1)
    pipeline = SortPipeline(
        trigger=trigger, camera=FakeCamera(), load_cell=cell, pan=pan,
        classifier=FakeClassifier(predictions), jurisdiction=jurisdiction, store=store,
        settings=PanSettings(settle_s=0.0, jam_retries=1),
    )
    return pipeline, pan, trigger, cell


def test_pipeline_recycles_metal(tmp_path, jurisdiction):
    pipeline, pan, trigger, _ = _pipeline(tmp_path, jurisdiction, [(WasteClass.METAL, 0.9)])
    event = pipeline.handle_event()
    assert pan.tilts == ["recycle"] and event.bin is Bin.RECYCLE and trigger.rearm_count == 1


def test_pipeline_ignores_confident_no_object(tmp_path, jurisdiction):
    pipeline, pan, _, _ = _pipeline(tmp_path, jurisdiction, [(WasteClass.NO_OBJECT, 0.95)])
    pipeline.handle_event()
    assert pan.tilts == []


def test_pipeline_flags_jam(tmp_path, jurisdiction):
    pipeline, pan, trigger, cell = _pipeline(
        tmp_path, jurisdiction, [(WasteClass.GLASS, 0.9)], jam=True
    )
    cell.current = 120.0
    event = pipeline.handle_event()
    assert not event.mech_ok and pan.position == "trash" and trigger.rearm_count == 0
