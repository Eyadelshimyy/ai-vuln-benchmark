"""Counting decodes, and naming every denominator they could be counted against.

A decode yield is a fraction, and the interesting question is what is under the
line. Of 45 objects crossing the gate on the shipped line, all 45 now fall
inside a code camera's band, but that was 29 before the optics were grounded in
real parts, and the two denominators gave 3.4 and 2.2 percent for the same two
decodes. Both denominators are reported, because a single figure misleads in
one direction or the other.

An object no camera could see did not fail to decode; it was never presented.
Counting it against the decoder blames optics on software, which is exactly the
confusion this project already made once when it read a gate covering 0.704 m of
a 1.00 m belt as covering 1.252 m of it.

Nothing here renders. The source is a protocol so the arithmetic is provable
against a fake, and the script that drives a real world lives outside this
module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from clave.tracker.codes import Decoder


@dataclass(frozen=True)
class Crossing:
    """One object passing the sensing gate.

    Attributes:
        frame: What the code camera saw, or None when none did.
        source_id: Which camera saw it, empty when none did.
        observed_at_nanos: When it crossed.
        material_class: What the object is, for the per-class breakdown.
        inside_a_code_band: Whether any code camera's measured lateral band
            contained it. An object outside every band was never presented to a
            decoder and is not a decoder failure.
    """

    frame: Any
    source_id: str
    observed_at_nanos: int
    material_class: str
    inside_a_code_band: bool


@dataclass(frozen=True)
class YieldReport:
    """What a decode sweep measured.

    Attributes:
        crossings: Objects that crossed the gate at all.
        readable: Objects inside a code camera's band.
        decoded: Symbols read, each having passed its own check digit.
        per_class: Crossings and decodes per material class.
    """

    crossings: int
    readable: int
    decoded: int
    per_class: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def per_readable(self) -> float | None:
        """Decodes per object a camera could actually see, or None."""
        return self.decoded / self.readable if self.readable else None

    @property
    def per_crossing(self) -> float | None:
        """Decodes per object on the belt, or None when nothing crossed.

        Zero decodes out of zero crossings is not a zero percent yield, and
        reporting it as one would put a number where there is no measurement.
        """
        return self.decoded / self.crossings if self.crossings else None

    def as_dict(self) -> dict[str, Any]:
        """Render as plain data, for the document that quotes the figure."""
        return {
            "crossings": self.crossings,
            "readable": self.readable,
            "decoded": self.decoded,
            "yield_per_readable_presentation": self.per_readable,
            "yield_per_object_on_the_belt": self.per_crossing,
            "per_class": {
                name: {"crossings": seen, "decoded": read}
                for name, (seen, read) in sorted(self.per_class.items())
            },
        }


class FrameSource(Protocol):
    """Anything that can enumerate gate crossings."""

    def __iter__(self) -> Any:
        """Yield one `Crossing` per object passing the gate."""


def measure_decode_yield(
    crossings: Iterable[Crossing], decoder: Decoder
) -> YieldReport:
    """Count what a decoder reads over a sweep of gate crossings.

    Args:
        crossings: Every object passing the gate, in any order.
        decoder: What reads a symbol.

    Returns:
        The report, with every denominator named. A symbol failing its own check
        digit never reaches the count, because a misread counted as a decode
        would inflate the published figure with readings that resolve to the
        wrong product.
    """
    seen = 0
    readable = 0
    decoded = 0
    per_class: dict[str, tuple[int, int]] = {}
    for crossing in crossings:
        seen += 1
        crossed, read = per_class.get(crossing.material_class, (0, 0))
        hit = 0
        if crossing.inside_a_code_band:
            readable += 1
            if decoder.decode(crossing.frame) is not None:
                decoded += 1
                hit = 1
        per_class[crossing.material_class] = (crossed + 1, read + hit)
    return YieldReport(
        crossings=seen, readable=readable, decoded=decoded, per_class=per_class
    )
