"""The seam, and the one implementation of it this spec ships.

Everything the two-spec split promises rests on the three names here.
`learned-tracker` replaces `SimulatorIdentity` with a trained model and changes
nothing above the protocol, and that promise is only checkable because `Cue` is
a closed type. What a learned model may look at is a dataclass a reviewer reads
in one sitting, rather than a rule somebody is trusted to keep.

An implementation is a pure function of its two arguments. It reads no clock,
opens no file, draws no random number and keeps nothing between calls, so a
replay of the same cues in the same order produces the same associations.
Every track it receives has already been propagated to the cue's instant by the
caller, so no implementation dead reckons and `belt_frame.propagate` has one
call site. `Tracker` is what holds state.

**`Cue` has no `previous` field, and the design document said it would.** The
tracks argument already carries the previous state of every open track, and a
seam whose whole value is being diffable should not carry the same thing twice.
The maintainer's framing, that the model sees the footprint, the pick point, an
optional code and the previous track state, is satisfied by `tracks`.

**`Cue.label` is the one field that makes this seam imperfect, and it is
deliberate.** A detection carries no identity, because the adapter throws away
the only one it has. Ground truth legitimately does
carry one, and the whole point of the shipped associator is that it uses it, so
that `learned-tracker` has something to be measured against. The alternative,
matching ground truth geometrically too, would mean this spec had already taken
identity away from the simulator and left the next one with no baseline.
`AC-TRACK-33` holds the learned model to building its input without this field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from clave.tracker.belt_frame import Footprint

GATE_MULTIPLE = 1.5
"""How far outside its own footprint a track will reach for an observation.

A multiple of the track's own major extent. That is what replaces
`association_radius_meters`, which was sized for a 0.16 m belt. The belt is
0.50 m, and a gate that scales with the object stays valid when the width
changes. The fixed radius is part of why only 7 of 134 demonstrations
produced a usable proposal.
"""


@dataclass(frozen=True)
class TrackSummary:
    """One open track, as an associator is allowed to see it.

    Attributes:
        track_id: Its stable identity.
        footprint: Where it is, already propagated to the cue's instant.
        height: What it is believed to stand, or None.
        digits: Every code associated to it so far.
        label: The simulator's object id, when one was folded in. Present so
            the shipped associator can match on it, and the field a learned
            model must build its input without.
        last_updated_nanos: When it last took evidence.
    """

    track_id: int
    footprint: Footprint
    height: float | None
    digits: tuple[str, ...]
    label: int | None
    last_updated_nanos: int


@dataclass(frozen=True)
class Cue:
    """Everything an associator may look at, and nothing else.

    Attributes:
        observed_at_nanos: When the observation was taken.
        footprint: Where it was seen, as observed and not propagated.
        digits: The decoded symbol, when the reading was a code.
        pick_point: Where a policy proposed reaching, when one did.
        height: A measured height, when one was taken.
        label: The simulator's object id, on a ground-truth cue only.
    """

    observed_at_nanos: int
    footprint: Footprint | None = None
    digits: str | None = None
    pick_point: NDArray[np.float64] | None = None
    height: float | None = None
    label: int | None = None

    def __post_init__(self) -> None:
        if self.pick_point is not None:
            object.__setattr__(
                self, "pick_point", np.asarray(self.pick_point, dtype=np.float64)
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cue):
            return NotImplemented
        if self.pick_point is None and other.pick_point is None:
            points_match = True
        elif self.pick_point is None or other.pick_point is None:
            points_match = False
        else:
            points_match = bool(
                np.allclose(self.pick_point, other.pick_point, rtol=0.0, atol=1e-12)
            )
        return (
            self.observed_at_nanos == other.observed_at_nanos
            and self.footprint == other.footprint
            and self.digits == other.digits
            and points_match
            and self.height == other.height
            and self.label == other.label
        )


@dataclass(frozen=True)
class Association:
    """Where one observation belongs.

    Attributes:
        track_id: The track it joins, or None to open a new one. Returning None
            rather than the nearest track at any distance is what keeps an
            observation of empty belt out of a real object's history.
    """

    track_id: int | None


class Associator(Protocol):
    """Decides which track an observation belongs to.

    The one seam between the record and the rule that fills it.
    """

    @property
    def name(self) -> str:
        """What decided the association, recorded in the run report."""

    def associate(self, cue: Cue, tracks: tuple[TrackSummary, ...]) -> Association:
        """Place one observation.

        Args:
            cue: Everything the implementation may look at.
            tracks: Every open track, already propagated to the cue's instant.

        Returns:
            The track it joins, or an association naming none.
        """


class SimulatorIdentity:
    """Joins by the simulator's label, and by geometry where there is none.

    This is not perception and its name says so. It exists to give the record a
    working association rule before any model is trained, and to give
    `learned-tracker` a baseline its identity recovery is reported against.

    It is asymmetric on purpose. A ground-truth cue carries the simulator's
    object id and matches on it exactly, which is the simulator supplying
    identity through the path real sensors use. A detection carries no identity
    at all, so it matches the nearest track inside a gate that scales with the
    track's own footprint. Ground truth for the label, geometry for the pixels,
    and the module name is the warning.
    """

    @property
    def name(self) -> str:
        """What decided the association."""
        return "simulator-identity"

    def associate(self, cue: Cue, tracks: tuple[TrackSummary, ...]) -> Association:
        """Place one observation against the open tracks.

        Args:
            cue: The observation.
            tracks: Every open track, propagated to the cue's instant.

        Returns:
            The track it joins, or an association naming none.
        """
        if cue.label is not None:
            for track in tracks:
                if track.label == cue.label:
                    return Association(track_id=track.track_id)
            # Detection may have opened the track before its simulator label
            # arrives. Fall back to geometry so the label joins that track.
            tracks = tuple(track for track in tracks if track.label is None)
        if cue.footprint is None:
            return Association(track_id=None)
        return Association(track_id=_nearest_within_gate(cue.footprint, tracks))


def _nearest_within_gate(
    footprint: Footprint, tracks: tuple[TrackSummary, ...]
) -> int | None:
    """Return the nearest track inside its own gate, or None.

    Args:
        footprint: Where the observation was seen.
        tracks: Every open track, propagated to the same instant.

    Returns:
        The track's identity, or None when nothing is close enough. The gate is
        a multiple of each track's own major extent, so a large package reaches
        further than a small one and neither is measured against a constant
        somebody chose for a different belt.
    """
    if not tracks:
        return None
    # One gate per track, and the distances to all of them at once: the
    # question is a nearest neighbour with a per-candidate ceiling, which is an
    # argmin over a masked array rather than a loop with two comparisons in it.
    centers = np.asarray([track.footprint.center for track in tracks], dtype=np.float64)
    gaps = np.linalg.norm(centers[:, :2] - np.asarray(footprint.center)[:2], axis=1)
    gates = GATE_MULTIPLE * np.asarray(
        [track.footprint.major_extent for track in tracks], dtype=np.float64
    )
    inside = np.flatnonzero(gaps <= gates)
    if inside.size == 0:
        return None
    return tracks[int(inside[int(np.argmin(gaps[inside]))])].track_id
