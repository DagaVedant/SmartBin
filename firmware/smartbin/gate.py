from __future__ import annotations

from dataclasses import dataclass

from .classes import Bin, WasteClass
from .config import Jurisdiction


@dataclass(frozen=True)
class Decision:
    waste_class: WasteClass
    confidence: float
    bin: Bin
    gated: bool


def decide(
    waste_class: WasteClass,
    confidence: float,
    jurisdiction: Jurisdiction,
) -> Decision:
    mapped = jurisdiction.bin_for(waste_class)

    if confidence < jurisdiction.confidence_threshold:
        return Decision(waste_class, confidence, Bin.TRASH, gated=mapped is Bin.RECYCLE)

    return Decision(waste_class, confidence, mapped, gated=False)
