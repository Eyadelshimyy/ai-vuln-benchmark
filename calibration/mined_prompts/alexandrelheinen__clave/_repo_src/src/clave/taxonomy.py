"""The material taxonomy, as code.

This mirrors `docs/waste-taxonomy.md`, which is the authority. Identifiers are
append-only there: a retired class keeps its identifier rather than having it
reused, because the pick decision contract, the simulated bins, the training
labels and the confusion matrix all key off these strings.

Changing anything here is a breaking change to every one of those.
"""

from __future__ import annotations

from dataclasses import dataclass

REJECT_CHANNEL = "CH-REJECT"
"""The channel receiving residue and anything the system cannot route."""


@dataclass(frozen=True)
class MaterialClass:
    """One material class and the channel it routes to by default.

    Attributes:
        id: Stable identifier of the form `M-NN`, never reused.
        name: Human-readable name. Presentation only; the id carries meaning.
        channel: Default channel. A deployment may override the mapping.
    """

    id: str
    name: str
    channel: str


MATERIAL_CLASSES: tuple[MaterialClass, ...] = (
    MaterialClass("M-01", "PET", "CH-PET"),
    MaterialClass("M-02", "HDPE", "CH-HDPE"),
    MaterialClass("M-03", "PP", "CH-PP"),
    MaterialClass("M-04", "Other plastic", "CH-PLASTIC-OTHER"),
    MaterialClass("M-05", "Aluminum", "CH-ALU"),
    MaterialClass("M-06", "Ferrous metal", "CH-FERROUS"),
    MaterialClass("M-07", "Glass", "CH-GLASS"),
    MaterialClass("M-08", "Corrugated cardboard", "CH-FIBER"),
    MaterialClass("M-09", "Mixed paper", "CH-FIBER"),
    MaterialClass("M-10", "Beverage carton", "CH-FIBER"),
    MaterialClass("M-11", "Residue", REJECT_CHANNEL),
)

BY_ID: dict[str, MaterialClass] = {entry.id: entry for entry in MATERIAL_CLASSES}

CLASS_COUNT = len(MATERIAL_CLASSES)


def channel_of(class_id: str) -> str:
    """Return the default channel for a material class.

    Args:
        class_id: An identifier of the form `M-NN`.

    Returns:
        The channel identifier.

    Raises:
        KeyError: If the class is not in the taxonomy.
    """
    return BY_ID[class_id].channel
