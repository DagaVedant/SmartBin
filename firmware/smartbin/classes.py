from enum import StrEnum


class WasteClass(StrEnum):
    RIGID_PLASTIC = "rigid_plastic"
    SOFT_PLASTIC_FOAM = "soft_plastic_foam"
    CARDBOARD_PAPER = "cardboard_paper"
    METAL = "metal"
    GLASS = "glass"
    EWASTE_HAZARD = "ewaste_hazard"
    NO_OBJECT = "no_object"


class Bin(StrEnum):
    TRASH = "trash"
    RECYCLE = "recycle"


DEFAULT_BIN = Bin.TRASH
