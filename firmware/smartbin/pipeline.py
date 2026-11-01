from __future__ import annotations

import logging
import time
from pathlib import Path

from .classes import Bin, WasteClass
from .classifier import Classifier
from .config import Jurisdiction, PanSettings
from .gate import decide
from .hardware import Camera, Frame, LoadCell, Pan, Trigger
from .storage import Event, EventStore

log = logging.getLogger(__name__)


class SortPipeline:
    def __init__(
        self,
        trigger: Trigger,
        camera: Camera,
        load_cell: LoadCell | None,
        pan: Pan,
        classifier: Classifier,
        jurisdiction: Jurisdiction,
        store: EventStore,
        settings: PanSettings,
        captures_dir: Path | None = None,
    ) -> None:
        self.trigger = trigger
        self.camera = camera
        self.load_cell = load_cell
        self.pan = pan
        self.classifier = classifier
        self.jurisdiction = jurisdiction
        self.store = store
        self.settings = settings
        self.captures_dir = captures_dir
        self.running = False

    def run_forever(self) -> None:
        self.running = True
        try:
            while self.running:
                if self.trigger.wait_for_item(timeout=1.0):
                    self.handle_event()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.running = False
        self.pan.hold_at_trash()
        self.camera.close()

    def handle_event(self) -> Event:
        time.sleep(self.settings.settle_s)

        weight_g = self.trigger.last_weight_g or (
            self.load_cell.read_grams() if self.load_cell else None
        )
        frame = self.camera.capture()

        waste_class, confidence = self.classifier.predict(frame, weight_g)
        decision = decide(waste_class, confidence, self.jurisdiction)

        if (
            decision.waste_class is WasteClass.NO_OBJECT
            and confidence >= self.jurisdiction.confidence_threshold
        ):
            mech_ok = True  # nothing on the pan: a false trigger, leave it level
        else:
            mech_ok = self._actuate(decision.bin)
        if mech_ok:
            self.trigger.rearm()
        image_path = self._save_frame(frame)

        event = Event(
            ts=int(time.time() * 1000),
            predicted=decision.waste_class,
            confidence=decision.confidence,
            bin=decision.bin,
            gated=decision.gated,
            weight_g=weight_g,
            image_path=image_path,
            mech_ok=mech_ok,
            model_version=self.classifier.version,
            jurisdiction=self.jurisdiction.name,
        )
        self.store.append(event)
        return event

    def _actuate(self, destination: Bin) -> bool:
        for attempt in range(self.settings.jam_retries + 1):
            self.pan.tilt(str(destination))
            if self._pan_cleared():
                return True
            log.warning("pan did not clear on attempt %d", attempt + 1)

        log.error("pan jammed; holding toward trash")
        self.pan.hold_at_trash()
        return False

    def _pan_cleared(self) -> bool:
        if self.load_cell is None:
            return True
        return self.load_cell.read_grams() < self.settings.jam_residual_g

    def _save_frame(self, frame: Frame) -> str | None:
        if self.captures_dir is None:
            return None

        from PIL import Image

        self.captures_dir.mkdir(parents=True, exist_ok=True)
        path = self.captures_dir / f"{int(time.time() * 1000)}.jpg"
        Image.fromarray(frame).save(path, quality=90)
        return str(path)
