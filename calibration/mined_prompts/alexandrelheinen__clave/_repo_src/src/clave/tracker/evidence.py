"""What one sensor reading becomes, and the five shapes it can take.

An adapter converts one sensor reading into one or more `Evidence` values. The
envelope is common and the payload is a tagged union, which is what lets a
consumer read a fused record without knowing that a camera exists.

Two rules give the union its value.

A variant states what was **measured**, never what should be **done**. No
payload carries a channel, a grasp or an instruction, so no adapter can start
deciding things the routing policy and the effector own. A test walks the
fields of every variant and enforces it, because the rule is easy to agree with
and easy to break.

Adding a variant is additive. A consumer that does not know one keeps working,
because it reads the fused record rather than the evidence stream, and a
fusion rule that does not recognize a payload reports the miss rather than
raising.

`GroundTruth` sits in the same union as the rest on purpose. It is how
simulation supplies labels through the path real sensors use, so the training
pipeline cannot come to depend on a shape that exists only in simulation. A
runtime configured for hardware refuses it, and that refusal lives at the
intake rather than here, because a payload that cannot be constructed cannot be
tested.

`PixelMask` lives in this module rather than beside the adapter that builds it.
A mask is data, the adapter is behavior, and putting the type next to the
adapter would have `evidence` importing `adapters` while `adapters` constructs
`Evidence`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError
from clave.taxonomy import BY_ID, CLASS_COUNT
from clave.tracker.belt_frame import Footprint

OUTCOME_COUNT = CLASS_COUNT + 1
"""The eleven taxonomy classes plus reject, which is what a posterior spans.

Whether reject is a predicted class or an absence of confidence is a question
`docs/perception-contract.md` declines to settle. A twelfth slot is the
reading taken here, and the contract is what should be amended if that is
wrong, rather than this constant being read loosely.
"""


class EvidenceError(ClaveError):
    """A reading describes something no sensor could have measured."""


class Role(Enum):
    """What kind of thing a sensor reported.

    Fusion dispatches on the payload and adapters are selected by role, so this
    is how a line gains a sensor without any consumer learning its name.
    """

    DETECTION = "detection"
    CODE = "code"
    DEPTH = "depth"
    SPECTRAL = "spectral"
    GROUND_TRUTH = "ground_truth"


@dataclass(frozen=True)
class PixelMask:
    """Which pixels of one frame an instance covers.

    Run-length triples rather than an array, so the whole input surface of the
    detection adapter is a literal a test can write, and so a mask can be
    compared for equality without reaching for numpy.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        runs: One `(row, start_column, length)` triple per horizontal run.
    """

    width: int
    height: int
    runs: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        """Refuse a mask describing pixels of no frame.

        Raises:
            EvidenceError: If the frame has no extent, if the mask covers no
                pixels, or if any run leaves the frame.
        """
        if self.width <= 0 or self.height <= 0:
            raise EvidenceError(
                f"a {self.width} by {self.height} frame holds no pixels to mask"
            )
        if not self.runs:
            raise EvidenceError("the mask covers no pixels, so it detects nothing")
        for row, start, length in self.runs:
            if length <= 0:
                raise EvidenceError(f"a run of length {length} covers no pixels")
            if not 0 <= row < self.height or start < 0 or start + length > self.width:
                raise EvidenceError(
                    f"run ({row}, {start}, {length}) falls outside a "
                    f"{self.width} by {self.height} frame"
                )

    @property
    def pixel_count(self) -> int:
        """How many pixels the instance covers."""
        return sum(length for _, _, length in self.runs)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """The pixel bounds as `(x_min, y_min, x_max, y_max)`, inclusive."""
        rows = [row for row, _, _ in self.runs]
        return (
            min(start for _, start, _ in self.runs),
            min(rows),
            max(start + length - 1 for _, start, length in self.runs),
            max(rows),
        )


@dataclass(frozen=True)
class Detection:
    """An object the imaging sensor found, in the belt frame.

    Attributes:
        footprint: The oriented box it occupies.
        mask: The pixels it covers, for associating a code to an object rather
            than to a position.
        height: Estimated height above the belt surface, in meters, or None
            when nothing has estimated one. A nadir footprint cannot be scaled
            without it, which is why it travels with the box.
    """

    footprint: Footprint
    mask: PixelMask
    height: float | None = None


@dataclass(frozen=True)
class Material:
    """What some imaging sensor believes the object is made of.

    Attributes:
        posterior: A distribution over the eleven taxonomy classes and reject,
            in taxonomy order with reject last.
    """

    posterior: tuple[float, ...]

    def __post_init__(self) -> None:
        """Refuse anything that is not a distribution over the taxonomy.

        Raises:
            EvidenceError: If the posterior spans the wrong number of outcomes,
                carries a value outside the unit interval, or does not sum to
                one.
        """
        if len(self.posterior) != OUTCOME_COUNT:
            raise EvidenceError(
                f"a posterior spans {OUTCOME_COUNT} outcomes, being the "
                f"taxonomy plus reject, and this one spans {len(self.posterior)}"
            )
        for weight in self.posterior:
            if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise EvidenceError(f"{weight!r} is not a probability")
        total = math.fsum(self.posterior)
        if abs(total - 1.0) > 1e-9:
            raise EvidenceError(
                f"the posterior does not sum to one, it sums to {total}"
            )


@dataclass(frozen=True)
class Code:
    """A symbol read off the package, and where it sat on the belt.

    Attributes:
        symbology: What kind of symbol it is, as the decoder named it.
        digits: The decoded digits, already past their check digit.
        quad: The symbol's corners in the belt frame, which is what lets a code
            associate to an object rather than to a position. Empty when the
            decoder reported no geometry.
    """

    symbology: str
    digits: str
    quad: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        """Refuse a symbol that does not check out.

        A `Code` that exists is a code that passed, so no consumer has to ask.

        Raises:
            EvidenceError: If the digits are not digits, or the check digit
                fails.
        """
        if not self.digits.isdigit():
            raise EvidenceError(f"{self.digits!r} is not a decoded symbol")
        if not check_digit_holds(self.digits):
            raise EvidenceError(
                f"{self.digits} fails its check digit, so it was misread"
            )


@dataclass(frozen=True)
class Height:
    """A measured height of the object's upper surface.

    Attributes:
        top_surface: Where the top of the object sits, in belt frame meters,
            which is measured from the floor like every other `z` in this
            frame.
    """

    top_surface: float


@dataclass(frozen=True)
class GroundTruth:
    """What the simulator knows, offered through the path a sensor uses.

    Attributes:
        object_id: The simulator's own identity. It is a label and never an
            input to association, which `AC-TRACK-33` holds the learned tracker
            to.
        material_class: Taxonomy identifier of the form `M-NN`.
        position: Where the object is, in belt frame meters.
    """

    object_id: int
    material_class: str
    position: NDArray[np.float64]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroundTruth):
            return NotImplemented
        return (
            self.object_id == other.object_id
            and self.material_class == other.material_class
            and bool(np.allclose(self.position, other.position, rtol=0.0, atol=1e-12))
        )

    def __post_init__(self) -> None:
        """Refuse a class the taxonomy does not define, and coerce position.

        Raises:
            EvidenceError: If the material class is unknown, naming it.
        """
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64)
        )
        if self.material_class not in BY_ID:
            raise EvidenceError(
                f"object {self.object_id} carries material class "
                f"{self.material_class!r}, which the taxonomy does not define"
            )


Payload = Detection | Material | Code | Height | GroundTruth
"""Everything a sensor can report.

Adding a variant is additive for consumers, who read the fused record, and it
is a change here plus one fusion rule. The set is fixed here, so widening it
is a decision rather than an import.
"""

ROLE_OF: dict[type, Role] = {
    Detection: Role.DETECTION,
    Material: Role.SPECTRAL,
    Code: Role.CODE,
    Height: Role.DEPTH,
    GroundTruth: Role.GROUND_TRUTH,
}
"""Which role each variant belongs under.

A classifier on any imaging sensor produces `Material`, so its role reads
`SPECTRAL` rather than naming the sensor that ran it, which is the same reason
nothing downstream reads a `source_id`.
"""


def check_digit_holds(digits: str) -> bool:
    """Return whether a UPC-A or EAN-13 string passes its own check digit.

    The modulo ten weighting is computed here rather than trusted from whatever
    decoded the symbol, so a misread is refused at the boundary instead of
    resolving to some other product's bill of materials.

    Args:
        digits: The decoded digits, including the trailing check digit.

    Returns:
        Whether the check digit is consistent with the rest.
    """
    if len(digits) not in (8, 12, 13):
        return False
    body = [int(digit) for digit in reversed(digits[:-1])]
    total = sum(
        digit * (3 if index % 2 == 0 else 1) for index, digit in enumerate(body)
    )
    return (10 - total % 10) % 10 == int(digits[-1])


@dataclass(frozen=True)
class Evidence:
    """One reading, from one sensor, at one instant.

    Attributes:
        source_id: Which sensor reported it. Provenance and debugging only, and
            no fusion rule accepts it, which forbids source priority by
            making it unrepresentable.
        role: What kind of thing was reported.
        observed_at_nanos: The monotonic instant it was taken. Belt speed is
            known, so this is what lets the reading be propagated to any later
            instant.
        confidence: The adapter's own certainty, from 0.0 to 1.0.
        payload: What was measured.
    """

    source_id: str
    role: Role
    observed_at_nanos: int
    confidence: float
    payload: Payload

    def __post_init__(self) -> None:
        """Refuse an envelope that disagrees with what it carries.

        Raises:
            EvidenceError: If the confidence is not a probability, if the
                payload is not one the union carries, or if the role does not
                match it. A code read filed under the detection role would be
                folded by the wrong rule, and nothing downstream would say so.
        """
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise EvidenceError(
                f"confidence is {self.confidence!r}, which states no certainty"
            )
        belongs = ROLE_OF.get(type(self.payload))
        if belongs is None:
            raise EvidenceError(
                f"{type(self.payload).__name__} is not a payload this union "
                f"carries; adding one means adding it to Payload and to ROLE_OF"
            )
        if self.role is not belongs:
            raise EvidenceError(
                f"{type(self.payload).__name__} is reported under role "
                f"{belongs.value!r}, not {self.role.value!r}"
            )
