"""Reading one published decision, the way an outside consumer reads it.

The decision is CBOR, specified by `crates/clave-decision/contract/pick-decision.cddl`
and pinned by the golden vectors beside it. This module implements that
specification rather than mirroring the Rust source, which is the same position
any other consumer is in, and the tests read the committed vectors for exactly
that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError

CONTRACT_VERSION = 1
"""The contract version this decoder implements.

A message carrying any other version is refused whole, and no field past the
version is read. The contract states that rule and the `unknown-version.hex`
vector exists so a decoder can point its rejection path at a real message.
"""

NANOS_PER_SECOND = 1_000_000_000
"""Nanoseconds in a second, for the monotonic times the contract carries."""

WIRE_CLASSES: dict[str, str] = {
    "Pet": "M-01",
    "Hdpe": "M-02",
    "Pp": "M-03",
    "OtherPlastic": "M-04",
    "Aluminum": "M-05",
    "Ferrous": "M-06",
    "Glass": "M-07",
    "Cardboard": "M-08",
    "MixedPaper": "M-09",
    "BeverageCarton": "M-10",
    "Residue": "M-11",
}
"""The names the wire carries, and what the taxonomy calls them.

The contract carries a name rather than a number so that reordering the set
cannot silently remap a label. The table is written out rather than derived from
the taxonomy's order, because deriving it would make a reordering in either
place change what a published label means.
"""


class DecisionError(ClaveError):
    """Bytes that are not a decision this build can read."""


@dataclass(frozen=True)
class PublishedDecision:
    """One decision as it arrived on the wire.

    Attributes:
        version: The contract version the message was produced under.
        object_id: The identity the tracker assigned.
        material_class: Taxonomy identifier of the form `M-NN`.
        channel: The channel the operator's mapping resolved, including reject.
        confidence: Classifier confidence, 0.0 to 1.0.
        point: Pick point in belt frame meters.
        yaw_radians: Rotation about the belt normal.
        reference_time_nanos: The instant the pose is predicted for.
        window_start_nanos: Earliest instant the object is reachable.
        window_end_nanos: Latest instant the object is reachable.
    """

    version: int
    object_id: int
    material_class: str
    channel: int
    confidence: float
    point: NDArray[np.float64]
    yaw_radians: float
    reference_time_nanos: int
    window_start_nanos: int
    window_end_nanos: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", np.asarray(self.point, dtype=np.float64))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PublishedDecision):
            return NotImplemented
        return (
            self.version == other.version
            and self.object_id == other.object_id
            and self.material_class == other.material_class
            and self.channel == other.channel
            and self.confidence == other.confidence
            and bool(np.allclose(self.point, other.point, rtol=0.0, atol=1e-12))
            and self.yaw_radians == other.yaw_radians
            and self.reference_time_nanos == other.reference_time_nanos
            and self.window_start_nanos == other.window_start_nanos
            and self.window_end_nanos == other.window_end_nanos
        )

    @property
    def window_seconds(self) -> float:
        """How long the object stays reachable, in seconds.

        A subscriber uses this to tell a decision it can still act on from one
        whose window closed while the message was in transit.
        """
        return (self.window_end_nanos - self.window_start_nanos) / NANOS_PER_SECOND


def read_vector(path: Path) -> bytes:
    """Read one golden vector, ignoring comments and whitespace.

    Args:
        path: A `.hex` file from the contract's vector directory.

    Returns:
        The message bytes.
    """
    digits = "".join(
        line.split("#", 1)[0].strip() for line in path.read_text().splitlines()
    )
    return bytes.fromhex(digits)


def decode(payload: bytes) -> PublishedDecision:
    """Decode one published decision.

    Args:
        payload: The bytes of one CBOR message.

    Returns:
        The decision.

    Raises:
        DecisionError: If `cbor2` is not installed, the bytes are not a CBOR
            map, the version is one this build does not implement, a field is
            missing or of the wrong type, or the class is not one the contract
            defines. The caller reports it and carries on; one unreadable
            datagram is not a reason to stop a conveyor.
    """
    try:
        import cbor2
    except ImportError as error:  # pragma: no cover - exercised by absence
        raise DecisionError(
            "cbor2 is not installed, so a published decision cannot be read. "
            'Install it with: uv pip install "clave[ros]"'
        ) from error

    try:
        read = cbor2.loads(payload)
    except Exception as error:
        raise DecisionError(f"the payload is not CBOR: {error}") from error
    if not isinstance(read, dict):
        raise DecisionError(f"the payload is a {type(read).__name__}, not a map")

    version = read.get("version")
    if version != CONTRACT_VERSION:
        raise DecisionError(
            f"the message declares contract version {version}, and this build "
            f"implements version {CONTRACT_VERSION}. No field past the version "
            "was read."
        )

    wire_class = _require(read, "class")
    material_class = WIRE_CLASSES.get(str(wire_class))
    if material_class is None:
        raise DecisionError(f"{wire_class!r} is not a class the contract defines")

    pose = _require(read, "pose")
    point = _require(pose, "point")
    window = _require(read, "window")
    try:
        return PublishedDecision(
            version=int(version),
            object_id=int(_require(read, "object")),
            material_class=material_class,
            channel=int(_require(read, "channel")),
            confidence=float(_require(read, "confidence")),
            point=np.asarray(
                (
                    float(_require(point, "x_meters")),
                    float(_require(point, "y_meters")),
                    float(_require(point, "z_meters")),
                ),
                dtype=np.float64,
            ),
            yaw_radians=float(_require(pose, "yaw_radians")),
            reference_time_nanos=int(_require(pose, "reference_time")),
            window_start_nanos=int(_require(window, "earliest")),
            window_end_nanos=int(_require(window, "latest")),
        )
    except (TypeError, ValueError) as error:
        raise DecisionError(
            f"a field holds the wrong kind of value: {error}"
        ) from error


def _require(mapping: Any, key: str) -> Any:
    """Read a field, naming it when it is absent.

    Args:
        mapping: The decoded map.
        key: The field required.

    Returns:
        The value.

    Raises:
        DecisionError: If the value is not a map or the field is absent.
    """
    if not isinstance(mapping, dict) or key not in mapping:
        raise DecisionError(f"the message carries no {key!r} field")
    return mapping[key]
