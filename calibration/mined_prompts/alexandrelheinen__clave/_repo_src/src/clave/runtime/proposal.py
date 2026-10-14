"""The Python half of the inference boundary.

The format is stated once, in `crates/clave-safety/contract/proposal.md`. This
module encodes it and refuses to send a proposal it already knows the runtime
will refuse, because a producer that ships garbage and reads the complaint back
has learned nothing the check here could not have told it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError
from clave.taxonomy import BY_ID

PROPOSAL_VERSION = 1
"""The wire version this build speaks.

A runtime that meets a version it does not implement refuses the whole message
rather than reading fields under the wrong meaning.
"""

STOP = b""
"""The stop signal: a datagram with no version and no fields."""


class ProposalError(ClaveError):
    """A proposal the runtime would refuse, caught before it is sent."""


@dataclass(frozen=True)
class Proposal:
    """One pick inference proposes, before any check has run.

    Attributes:
        object_id: Identity of the object, carried unchanged into the decision.
        material_class: Taxonomy identifier of the form `M-NN`.
        confidence: Classifier confidence, 0.0 to 1.0 inclusive.
        point: Proposed pick point in belt frame meters.
        yaw_radians: Proposed rotation about the belt normal.
        reference_time_nanos: Instant the pose is predicted for.
        window_start_nanos: Earliest instant the object is reachable.
        window_end_nanos: Latest instant the object is reachable.
    """

    object_id: int
    material_class: str
    confidence: float
    point: NDArray[np.float64]
    yaw_radians: float
    reference_time_nanos: int
    window_start_nanos: int
    window_end_nanos: int

    def __post_init__(self) -> None:
        """Refuse a proposal the runtime would refuse, and coerce the point.

        Raises:
            ProposalError: Naming the field that is wrong.
        """
        object.__setattr__(self, "point", np.asarray(self.point, dtype=np.float64))
        if self.material_class not in BY_ID:
            raise ProposalError(
                f"material class {self.material_class!r} is not in the taxonomy"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ProposalError(
                f"confidence {self.confidence} is outside 0.0 to 1.0 inclusive"
            )
        if self.window_end_nanos < self.window_start_nanos:
            raise ProposalError(
                f"window {self.window_start_nanos} to {self.window_end_nanos} "
                "ends before it starts"
            )
        if not (
            self.window_start_nanos
            <= self.reference_time_nanos
            <= self.window_end_nanos
        ):
            raise ProposalError(
                f"reference time {self.reference_time_nanos} falls outside the window"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Proposal):
            return NotImplemented
        return (
            self.object_id == other.object_id
            and self.material_class == other.material_class
            and self.confidence == other.confidence
            and bool(np.allclose(self.point, other.point, rtol=0.0, atol=1e-12))
            and self.yaw_radians == other.yaw_radians
            and self.reference_time_nanos == other.reference_time_nanos
            and self.window_start_nanos == other.window_start_nanos
            and self.window_end_nanos == other.window_end_nanos
        )

    def encode(self) -> bytes:
        """Render as one datagram.

        Returns:
            The UTF-8 JSON the runtime reads.
        """
        x, y, z = (float(self.point[0]), float(self.point[1]), float(self.point[2]))
        return json.dumps(
            {
                "version": PROPOSAL_VERSION,
                "object_id": self.object_id,
                "material_class": self.material_class,
                "confidence": self.confidence,
                "x_meters": x,
                "y_meters": y,
                "z_meters": z,
                "yaw_radians": self.yaw_radians,
                "reference_time_nanos": self.reference_time_nanos,
                "window_start_nanos": self.window_start_nanos,
                "window_end_nanos": self.window_end_nanos,
            },
            sort_keys=True,
        ).encode("utf-8")


@dataclass(frozen=True)
class Outcome:
    """What the runtime did with one proposal.

    Attributes:
        verdict: One of `accepted`, `overridden` or `refused`.
        object_id: The object, when one was read.
        channel: The channel the policy resolved, when one was.
        reject_reason: Why it was routed to reject, when it was.
        check: The geometric check that failed, when one did.
        reason: What the decoder reported, when the proposal was refused.
    """

    verdict: str
    object_id: int | None = None
    channel: int | None = None
    reject_reason: str | None = None
    check: str | None = None
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        """Whether a decision was published for this proposal."""
        return self.verdict == "accepted"

    @property
    def overridden(self) -> bool:
        """Whether the safety layer refused the proposed point."""
        return self.verdict == "overridden"

    @classmethod
    def decode(cls, payload: bytes) -> Outcome:
        """Read one outcome datagram.

        Args:
            payload: The JSON the runtime sent.

        Returns:
            The outcome.

        Raises:
            ProposalError: If the payload is not an outcome.
        """
        try:
            read: dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProposalError(
                f"the runtime sent something unreadable: {error}"
            ) from (error)
        verdict = read.get("verdict")
        if verdict not in {"accepted", "overridden", "refused"}:
            raise ProposalError(f"the runtime reported an unknown verdict {verdict!r}")
        return cls(
            verdict=verdict,
            object_id=read.get("object_id"),
            channel=read.get("channel"),
            reject_reason=read.get("reject_reason"),
            check=read.get("check"),
            reason=read.get("reason"),
        )
