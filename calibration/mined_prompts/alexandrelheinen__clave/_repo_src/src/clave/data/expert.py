"""The scripted expert.

Imitation learning needs a teacher. This is it: a policy that picks the
reachable object with least time remaining and routes it by its material class.

It is deliberately simple. Its job is to be explainable, not to be good. If a
learned policy matches it exactly, that says imitation worked, not that the
behavior is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clave.data.examples import ObjectLabel
from clave.taxonomy import channel_of


@dataclass(frozen=True)
class PickDecision:
    """What the expert decided for one object.

    Attributes:
        object_id: The object to pick.
        material_class: Its material class.
        channel: Where it routes.
        position: Where it was when the decision was taken.
    """

    object_id: int
    material_class: str
    channel: str
    position: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PickDecision):
            return NotImplemented
        return (
            self.object_id == other.object_id
            and self.material_class == other.material_class
            and self.channel == other.channel
            and bool(np.allclose(self.position, other.position, rtol=0.0, atol=1e-12))
        )


def decide(labels: tuple[ObjectLabel, ...], window_exit: float) -> PickDecision | None:
    """Choose an object to pick, or decline.

    Among the objects currently inside the reachable window, the one nearest the
    window exit has the least time remaining, so it is the one a real line would
    take first.

    Args:
        labels: The objects visible this frame.
        window_exit: Belt coordinate at which the reachable window ends.

    Returns:
        A decision, or None when nothing is reachable. Returning None rather
        than a decision naming nothing is what keeps a consumer from having to
        check for a sentinel object id.
    """
    reachable = [label for label in labels if label.in_reachable_window]
    if not reachable:
        return None
    # Nearest the exit first, with the object id breaking ties so the choice is
    # deterministic when two objects sit at the same coordinate.
    chosen = min(
        reachable, key=lambda label: (window_exit - label.position[0], label.object_id)
    )
    return PickDecision(
        object_id=chosen.object_id,
        material_class=chosen.material_class,
        channel=channel_of(chosen.material_class),
        position=chosen.position,
    )
