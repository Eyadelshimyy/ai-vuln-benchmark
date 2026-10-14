"""One belief per object, and the only record a consumer reads.

`Tracker` is the one mutable thing in this package. Everything it calls is a
pure function, so what changes between two observations is visible in one class
rather than spread across the modules it drives.

`Tracker.observe` passes every reading through the intake. That is what makes
the `GroundTruth` refusal a property of the system rather than a habit: there is
no second way in, and a test counts the doors.

`WasteObject` carries no channel. The contract lists one and calls it resolved
by the routing policy rather than by perception, which means perception could
only ever write a null there, and a field that is always absent is worse than an
absent field.

It does carry `simulated`, naming the fields that came from the simulator rather
than from a sensor. No per-instance classifier exists in this repository, so a
material folded from `GroundTruth` is the simulator's label. A record that does
not say so launders a supplied value into a perceived one, which is the failure
this spec exists to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from clave.tracker.association import Associator, Cue, TrackSummary
from clave.tracker.belt_frame import Footprint, elapsed_seconds, propagate
from clave.tracker.evidence import Code, Detection, Evidence, GroundTruth, Height, Role
from clave.tracker.fusion import FusionSettings, Posterior, Resolver, fold, mass_band
from clave.tracker.intake import Intake
from clave.tracker.motion import Estimate
from clave.tracker.motion import begin as begin_motion
from clave.tracker.motion import fold as fold_motion
from clave.world.config import Range

NANOS_PER_SECOND = 1_000_000_000
"""Nanoseconds in a second, for the monotonic instants an observation carries."""


@dataclass
class Track:
    """What the tracker believes about one object on the belt.

    Attributes:
        track_id: Stable for the object's life on the belt, and never reused.
        footprint: Where it was last seen, at `observed_at_nanos`.
        observed_at_nanos: The instant the footprint describes.
        posterior: What it is believed to be made of.
        height: Its estimated height above the belt, or None.
        codes: Every symbol associated to it, in the order they arrived.
        contributors: Which sources have been folded in, for auditing.
        evidence: Which roles have contributed.
        simulated: Which record fields derive from the simulator's label.
        label: The simulator's object id, when one was folded in.
        motion: Where the filter believes the object is and how fast it is
            going, or None before any detection has opened one. It is the
            estimate a consumer is propagated from; `footprint` keeps the
            extents and the yaw, which the filter does not model.
        first_seen_nanos: When it opened.
        last_updated_nanos: When it last took evidence.
    """

    track_id: int
    footprint: Footprint
    observed_at_nanos: int
    posterior: Posterior
    first_seen_nanos: int
    last_updated_nanos: int
    height: float | None = None
    codes: tuple[str, ...] = ()
    contributors: frozenset[str] = frozenset()
    evidence: frozenset[Role] = frozenset()
    simulated: frozenset[str] = frozenset()
    label: int | None = None
    motion: Estimate | None = None

    def at(self, at_nanos: int, belt_speed: float) -> Footprint:
        """Return where this track is at a later instant.

        Carried by the filter's own velocity estimate where one exists, and
        by the configured belt speed where none does. The difference is the
        lateral axis: the belt model says an object never drifts sideways,
        and an object rolling or settling does, by tens of millimetres over
        the horizon a pick is planned across.

        Args:
            at_nanos: The instant wanted.
            belt_speed: Belt speed in meters per second, used only before a
                detection has opened an estimate.

        Returns:
            The footprint, with its extents and yaw unchanged.
        """
        if self.motion is None:
            return propagate(
                self.footprint, belt_speed, self.observed_at_nanos, at_nanos
            )
        x, y = self.motion.at(at_nanos / NANOS_PER_SECOND)
        return replace(
            self.footprint,
            center_belt=np.asarray(
                (x, y, float(self.footprint.center[2])), dtype=np.float64
            ),
        )

    def summarize(self, at_nanos: int, belt_speed: float) -> TrackSummary:
        """Return this track as an associator is allowed to see it.

        Args:
            at_nanos: The instant to propagate to.
            belt_speed: Belt speed in meters per second.

        Returns:
            The summary.
        """
        return TrackSummary(
            track_id=self.track_id,
            footprint=self.at(at_nanos, belt_speed),
            height=self.height,
            digits=self.codes,
            label=self.label,
            last_updated_nanos=self.last_updated_nanos,
        )


@dataclass(frozen=True)
class WasteObject:
    """The only record a consumer reads.

    No field names a sensor: `evidence` names roles, so a consumer can report
    that a decision used a code without knowing which of three cameras read it.

    Attributes:
        track_id: The identity the tracker assigned.
        observed_at_nanos: When this description was settled.
        valid_until_nanos: When belt travel invalidates the pose, which is the
            instant the object reaches the measured window exit.
        footprint: Where it is, propagated to `observed_at_nanos`.
        height: Its estimated height above the belt, or None.
        material: Taxonomy identifier, or `reject`.
        material_confidence: How much of the belief that class holds.
        density: The bulk density band its class implies, or None when nothing
            in the world carries that class.
        mass: What it might weigh, as a band, or None.
        codes: Every symbol associated to it.
        evidence: Which roles contributed.
        simulated: Which of these fields came from the simulator rather than
            from a sensor.
    """

    track_id: int
    observed_at_nanos: int
    valid_until_nanos: int
    footprint: Footprint
    height: float | None
    material: str
    material_confidence: float
    density: Range | None
    mass: Range | None
    codes: tuple[str, ...]
    evidence: frozenset[Role]
    simulated: frozenset[str]

    @property
    def grasp_point(self) -> NDArray[np.float64]:
        """Where the end effector should meet the object.

        Computed rather than stored: a stored grasp point can disagree with the
        footprint it came from and a computed one cannot.
        """
        return self.footprint.center

    @property
    def grasp_axis(self) -> float:
        """The footprint's minor axis, as a rotation about the belt normal."""
        return self.footprint.yaw + 1.5707963267948966

    @property
    def grasp_width(self) -> float:
        """What a jaw would have to open to. A suction cup ignores it."""
        return self.footprint.minor_extent

    @property
    def surface_normal(self) -> NDArray[np.float64]:
        """Where a suction cup should point.

        Vertical, because objects travel in a single layer on a flat belt and
        nothing here measures a surface orientation. Stating it as vertical is
        honest; deriving a tilt from a footprint would not be.
        """
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)


@dataclass
class Tracker:
    """Holds one belief per object and folds evidence into it.

    Attributes:
        intake: The one door every reading crosses.
        associator: The rule deciding which track a reading joins.
        settings: The fusion floor, half life and density bands.
        belt_speed: Belt speed in meters per second.
        window_exit: Where the reachable window closes, in belt frame meters.
        unmeasured_extent: How wide to assume a track is before anything has
            measured it, in meters. A reading that carries no geometry, which
            `GroundTruth` does not, still opens a track somewhere, and the
            association gate scales with the track's own extent. Left at zero
            that gate is nothing and no later detection can ever join, so the
            two would run as separate tracks over one object. The caller passes
            the widest object the line handles, which
            `arm.max_grasp_width_meters` already states.
        resolve: Turns a decoded symbol into packaging components.
    """

    intake: Intake
    associator: Associator
    settings: FusionSettings
    belt_speed: float
    window_exit: float
    unmeasured_extent: float = 0.0
    resolve: Resolver | None = None
    _tracks: dict[int, Track] = field(default_factory=dict)
    _next_id: int = 1
    _retired: int = 0

    def observe(self, evidence: Evidence, at_nanos: int) -> None:
        """Fold one reading into whichever track it belongs to.

        Args:
            evidence: What an adapter produced.
            at_nanos: The instant to fold it at.

        Raises:
            IntakeError: If the boundary refuses the reading. Every reading
                crosses it, which is what makes the refusal a property of the
                system rather than a habit of its callers.
        """
        admitted = self.intake.accept(evidence)
        cue = _cue_of(admitted, self._extent())
        summaries = tuple(
            track.summarize(cue.observed_at_nanos, self.belt_speed)
            for track in self._tracks.values()
        )
        placed = self.associator.associate(cue, summaries)
        track = (
            self._tracks.get(placed.track_id) if placed.track_id is not None else None
        )
        if track is None:
            track = self._open(cue, at_nanos)
        self._fold(track, admitted, at_nanos)

    @property
    def retired(self) -> int:
        """How many tracks have been dropped for going unobserved."""
        return self._retired

    def settle(self, at_nanos: int) -> tuple[WasteObject, ...]:
        """Describe every track still worth describing, as of one instant.

        Retires first. A track nobody has observed for longer than the
        configured bound is dropped rather than carried forward, because its
        position is no longer known and a record says nothing about how old
        the estimate behind it is. Carried forward instead, such a track is
        dead reckoned indefinitely: measured before this existed, records
        past the arm's reach had a median error of ten metres and
        outnumbered the ones inside the sensing gate nineteen to one.

        Dropping it also clears the association gate. A stale track sits
        where nothing is, and a fresh detection landing near the real object
        fails to join it and opens a duplicate instead.

        Args:
            at_nanos: The instant to describe them at.

        Returns:
            One record per surviving track, in the order the tracks opened.
        """
        self._retire(at_nanos)
        return tuple(
            self._settle(track, at_nanos)
            for track in sorted(self._tracks.values(), key=lambda t: t.track_id)
        )

    def _retire(self, at_nanos: int) -> None:
        """Drop every track whose position estimate has gone stale.

        Args:
            at_nanos: Now.
        """
        bound = self.settings.retire_after
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if elapsed_seconds(track.observed_at_nanos, at_nanos) > bound
        ]
        for track_id in stale:
            del self._tracks[track_id]
        self._retired += len(stale)

    def _extent(self) -> float:
        """Return the width to assume before anything has measured one."""
        return max(self.unmeasured_extent, 1e-6)

    def _unmeasured(self) -> Footprint:
        """Return the footprint a track carries before anything measures it.

        The extent is what the association gate scales with, so a sentinel of
        nothing means nothing can ever join. The line's widest handled object is
        the honest assumption: it over-reaches rather than under-reaches, and a
        real detection replaces it the moment one arrives.
        """
        extent = self._extent()
        return Footprint(
            center=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
            major_extent=extent,
            minor_extent=extent,
            yaw=0.0,
        )

    def _open(self, cue: Cue, at_nanos: int) -> Track:
        """Start a new track from one cue.

        A `track_id` is minted here and never reused, so two objects can never
        share one and a retired identity never comes back.
        """
        track_id = self._next_id
        self._next_id += 1
        track = Track(
            track_id=track_id,
            footprint=cue.footprint or self._unmeasured(),
            observed_at_nanos=cue.observed_at_nanos,
            posterior=Posterior.uniform(),
            first_seen_nanos=at_nanos,
            last_updated_nanos=at_nanos,
        )
        self._tracks[track_id] = track
        return track

    def _fold(self, track: Track, evidence: Evidence, at_nanos: int) -> None:
        """Fold one admitted reading into one track."""
        resolver = self.resolve
        result = (
            fold(track.posterior, evidence, at_nanos, self.settings, resolver)
            if resolver is not None
            else fold(track.posterior, evidence, at_nanos, self.settings)
        )
        track.posterior = result.posterior
        track.contributors |= {evidence.source_id}
        track.evidence |= {evidence.role}
        track.last_updated_nanos = at_nanos

        payload = evidence.payload
        if isinstance(payload, Detection):
            seen = payload.footprint.center
            at_seconds = evidence.observed_at_nanos / NANOS_PER_SECOND
            track.motion = (
                begin_motion((seen[0], seen[1]), self.belt_speed, at_seconds)
                if track.motion is None
                else fold_motion(
                    track.motion,
                    (seen[0], seen[1]),
                    at_seconds,
                    self.settings.along,
                    self.settings.across,
                )
            )
            # The extents and the yaw come from the observation as they
            # always did: the filter models where an object is going, not
            # how wide it is. Its centre is overwritten by the estimate,
            # which is the whole difference from replacing outright.
            filtered = track.motion.position
            track.footprint = replace(
                payload.footprint,
                center_belt=np.asarray(
                    (filtered[0], filtered[1], float(seen[2])), dtype=np.float64
                ),
            )
            track.observed_at_nanos = evidence.observed_at_nanos
            if payload.height is not None:
                track.height = payload.height
        elif isinstance(payload, Height):
            track.height = payload.top_surface - track.footprint.center[2]
        elif isinstance(payload, Code):
            if payload.digits not in track.codes:
                track.codes = (*track.codes, payload.digits)
        elif isinstance(payload, GroundTruth):
            track.label = payload.object_id
            track.simulated |= {"material"}
            if np.allclose(
                track.footprint.center,
                self._unmeasured().center,
                rtol=0.0,
                atol=1e-12,
            ):
                track.footprint = replace(track.footprint, center_belt=payload.position)
                track.observed_at_nanos = evidence.observed_at_nanos

    def _settle(self, track: Track, at_nanos: int) -> WasteObject:
        """Turn one track into the record a consumer reads."""
        carried = track.at(at_nanos, self.belt_speed)
        material, confidence = track.posterior.most_likely
        density = self.settings.densities.get(material)
        volume = (
            carried.major_extent * carried.minor_extent * track.height
            if track.height is not None
            else 0.0
        )
        remaining = max(0.0, (self.window_exit - carried.center[0]))
        expires = (
            at_nanos + int(remaining / self.belt_speed * NANOS_PER_SECOND)
            if self.belt_speed > 0.0
            else at_nanos
        )
        return WasteObject(
            track_id=track.track_id,
            observed_at_nanos=at_nanos,
            valid_until_nanos=expires,
            footprint=carried,
            height=track.height,
            material=material,
            material_confidence=confidence,
            density=density,
            mass=mass_band(material, volume, self.settings),
            codes=track.codes,
            evidence=frozenset(track.evidence),
            simulated=frozenset(track.simulated),
        )


def _cue_of(evidence: Evidence, unmeasured_extent: float) -> Cue:
    """Return what an associator is allowed to see of one reading.

    Every field an implementation may read is filled here and nowhere else, so
    widening what a tracker exposes to the seam is one function to review.

    Args:
        evidence: An admitted reading.
        unmeasured_extent: How wide to call a reading that carries a position
            but no extent, which `GroundTruth` does. The association gate
            scales with it, so a sentinel here means nothing can ever join.

    Returns:
        The cue.
    """
    payload = evidence.payload
    if isinstance(payload, Detection):
        return Cue(
            observed_at_nanos=evidence.observed_at_nanos,
            footprint=payload.footprint,
            height=payload.height,
        )
    if isinstance(payload, Code):
        return Cue(observed_at_nanos=evidence.observed_at_nanos, digits=payload.digits)
    if isinstance(payload, Height):
        return Cue(
            observed_at_nanos=evidence.observed_at_nanos, height=payload.top_surface
        )
    if isinstance(payload, GroundTruth):
        return Cue(
            observed_at_nanos=evidence.observed_at_nanos,
            footprint=Footprint(
                center=payload.position,
                major_extent=unmeasured_extent,
                minor_extent=unmeasured_extent,
                yaw=0.0,
            ),
            label=payload.object_id,
        )
    return Cue(observed_at_nanos=evidence.observed_at_nanos)


def elapsed(earlier_nanos: int, later_nanos: int) -> float:
    """Return the seconds between two instants, for a caller that has both."""
    return elapsed_seconds(earlier_nanos, later_nanos)
