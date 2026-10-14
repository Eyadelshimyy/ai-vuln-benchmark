"""One visit, planned end to end before the arm moves.

The task machine used to hold a pose and wait for the flange to reach it.
That is enough to follow an object and not enough to pick one up: a jaw has
to arrive at a known instant, moving with the object, having come straight
down onto it, and none of those is available from a controller that finds
out when it arrives by arriving.

So a visit is planned as a sequence of timed arcs, each one's end being the
next one's start, and the arm is driven by the clock rather than by
proximity. [clave.control.trajectory] holds the arc algebra and
`docs/trajectory-formulation.md` the mathematics; this holds the sequence.

**The interception time is solved once; the aim is not.** A duration that
keeps changing is a duration nothing can be scheduled against, so `T` is
fixed at commit. Where the arm is aiming is refreshed every capture until
the descent begins, because the belt model carries the x axis exactly and
carries nothing else: an object rolling or settling drifts sideways out
from under a pose predicted three seconds ago. See [refine].

**The hold is a carry, not a stop.** While the jaw closes, the belt keeps
moving and the flange has to move with it. A quintic whose endpoints differ
by exactly the velocity times the duration, with that velocity at both ends
and no acceleration, reduces to a straight line at constant speed, so the
carry is the same object as every other arc rather than a special case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from clave.control.settings import Phase
from clave.control.trajectory import (
    Segment,
    State,
    approach,
    descend,
    descent_seconds,
    distance,
    same,
    soonest_feasible,
    where_carried,
    zeros,
)

JAW_OPEN = 0.0
"""The jaw command while the arm is travelling and descending."""

JAW_SHUT = 1.0
"""The jaw command from the moment the hold begins until the object is over
its chute."""

PEAK_OVER_MEAN = 15.0 / 8.0
"""What a rest-to-rest quintic's peak speed is, as a multiple of its mean.

Exact rather than a safety factor: with every boundary condition but the
endpoints at zero, only one basis term survives and its derivative is
`30 s^2 (1-s)^2`, which peaks at `15/8` in the middle.
`docs/trajectory-formulation.md` derives it.
"""

"""A floor on the delivery, so a chute already underneath is not a zero arc.

A segment of no duration divides by its own span. The floor is far below
any delivery the shipped geometry produces and exists so the arithmetic
cannot be handed a zero.
"""


def _transport(velocity: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return the belt-axis model transport the jaw matches ($\mathbf{v}_T$).

    The arcs are planned in the target frame. For the object target, transport
    is belt-axis only: the belt drives one axis, and contact-induced lateral
    or vertical body velocity is a disturbance, not the model the hold and
    retreat ride. Lateral drift still aims the target origin through
    [_drift_velocity] and [where_carried], for a finite horizon only.

    Args:
        velocity: The velocity a caller measured for the object.

    Returns:
        $(\max(0, v_x), 0, 0)$: no reverse travel along the belt, and no
        lateral or vertical matching.
    """
    return np.asarray((max(0.0, float(velocity[0])), 0.0, 0.0), dtype=np.float64)


def nearer_parallel_yaw(target: float, reference: float) -> float:
    """Return target or target+π, whichever is nearer reference (`AC-BEHAVE-03`).

    A parallel jaw is the same grip at θ and at θ+π. Choosing the nearer one
    keeps the wrist from slewing halfway around the tool while the visit is
    already short on joint travel.

    Args:
        target: The yaw the grasp claims, in radians.
        reference: The yaw the tool already holds, in radians.

    Returns:
        The chosen yaw, wrapped into (−π, π].
    """
    two_pi = 2.0 * math.pi

    def wrap(angle: float) -> float:
        return (angle + math.pi) % two_pi - math.pi

    flipped = wrap(target + math.pi)
    if abs(wrap(flipped - reference)) < abs(wrap(target - reference)):
        return flipped
    return wrap(target)


def _drift_velocity(velocity: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the velocity used only to aim the target origin for a while.

    Across the belt nothing drives the object, so this component is carried
    only through [where_carried] / `drift_horizon`. It must not become the
    matching velocity of HOLD or RETREAT (`AC-MOVE-71`).

    Args:
        velocity: The velocity a caller measured for the object.

    Returns:
        The same velocity with no component along the belt normal.
    """
    return np.asarray((float(velocity[0]), float(velocity[1]), 0.0), dtype=np.float64)


def _fit_segment(
    start: State,
    end: State,
    max_speed: float,
    max_acceleration: float,
    min_duration: float,
    *,
    segment_sample_count: int,
    bisection_passes: int,
) -> Segment:
    """Return a quintic segment respecting speed and acceleration ceilings.

    The duration is found by the monotone feasibility search in
    [clave.control.trajectory.soonest_feasible]: peak speed and acceleration
    both fall as the duration grows, so the smallest duration that fits is a
    bisection rather than a step over a range.

    Args:
        start: State at the beginning.
        end: State at the end.
        max_speed: Speed ceiling, in meters per second.
        max_acceleration: Acceleration ceiling, in meters per second squared.
        min_duration: Minimum duration to consider, in seconds.

    Returns:
        The fitted segment respecting both ceilings.
    """
    span = distance(start.position, end.position)
    ceiling = max(max_speed, 1e-6)
    lower = max(PEAK_OVER_MEAN * span / ceiling, min_duration)

    def feasible(seconds: float) -> bool:
        return Segment(start=start, end=end, duration=seconds).fits(
            max_speed, max_acceleration, segment_sample_count
        )

    return Segment(
        start=start,
        end=end,
        duration=soonest_feasible(feasible, lower, bisection_passes),
    )


@dataclass(frozen=True)
class VisitTick:
    """What the arm is asked for on one tick of a planned visit.

    Attributes:
        phase: Which phase of the visit this tick belongs to.
        position: Where the flange should be, in belt frame meters.
        velocity: How fast it should be moving there. The servo needs this
            as well as the position, because it leads the plant by a time
            rather than by a distance.
        yaw: What the tool should be turned to about the belt normal, or
            None to hold the rotation it already has.
        grip: What the jaw is commanded to, from zero open to one shut.
    """

    phase: Phase
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    yaw: float | None
    grip: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VisitTick):
            return NotImplemented
        return (
            self.phase is other.phase
            and same(self.position, other.position)
            and same(self.velocity, other.velocity)
            and self.yaw == other.yaw
            and self.grip == other.grip
        )


@dataclass(frozen=True)
class Leg:
    """One arc of a visit, with what the jaw does during it.

    Attributes:
        phase: Which phase of the visit this arc is.
        segment: The arc.
        grip: What the jaw is commanded to, from zero open to one shut.
    """

    phase: Phase
    segment: Segment
    grip: float


@dataclass(frozen=True)
class Plan:
    """A whole visit, timed from the instant it was made.

    Attributes:
        track_id: Which track it is about.
        legs: The arcs, in order.
        started_at: Simulated time the first arc begins, in seconds.
        pick_at: When the jaw reaches the object, in seconds. Reported
            because it is the figure the interception exists to produce.
        target_yaw: Orientation about the belt normal at the pick, or None.
        initial_yaw: Orientation about the belt normal at the approach start, or None.
        transport_velocity: Belt-axis model transport for the object target.
            World hold and retreat are this transport composed with rest and
            +Z in the object frame.
    """

    track_id: int
    legs: tuple[Leg, ...]
    started_at: float
    pick_at: float
    target_yaw: float | None = None
    initial_yaw: float | None = None
    transport_velocity: NDArray[np.float64] = field(default_factory=zeros)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plan):
            return NotImplemented
        return (
            self.track_id == other.track_id
            and self.legs == other.legs
            and self.started_at == other.started_at
            and self.pick_at == other.pick_at
            and self.target_yaw == other.target_yaw
            and self.initial_yaw == other.initial_yaw
            and same(self.transport_velocity, other.transport_velocity)
        )

    @property
    def duration(self) -> float:
        """How long the whole visit takes, in seconds."""
        return self._boundaries()[-1]

    def with_yaw(
        self,
        target_yaw: float | None = None,
        initial_yaw: float | None = None,
    ) -> Plan:
        """Return a copy of the plan with updated target and initial yaw."""
        return Plan(
            track_id=self.track_id,
            legs=self.legs,
            started_at=self.started_at,
            pick_at=self.pick_at,
            target_yaw=target_yaw if target_yaw is not None else self.target_yaw,
            initial_yaw=initial_yaw if initial_yaw is not None else self.initial_yaw,
            transport_velocity=self.transport_velocity,
        )

    def yaw_at(self, at_seconds: float) -> float | None:
        """Return the commanded tool yaw at a simulated instant, or None."""
        if self.target_yaw is None:
            return None
        if self.initial_yaw is None:
            return self.target_yaw
        track_duration = max(1e-6, self.pick_at - self.started_at)
        elapsed = at_seconds - self.started_at
        if elapsed >= track_duration:
            return self.target_yaw
        tau = float(np.clip(elapsed / track_duration, 0.0, 1.0))
        s = tau * tau * tau * (10.0 + tau * (-15.0 + 6.0 * tau))
        diff = (
            self.target_yaw - self.initial_yaw + math.pi / 2.0
        ) % math.pi - math.pi / 2.0
        two_pi = 2.0 * math.pi
        return (self.initial_yaw + s * diff + math.pi) % two_pi - math.pi

    def at(self, at_seconds: float) -> tuple[Phase, State, float] | None:
        """Return where the visit is, or None once it is over.

        A leg owns the half-open interval from its start to its end, and the
        last leg owns its end as well. That convention is what puts the pick
        instant in the hold rather than in the descent: the two arcs meet at
        one point, they agree about the state there, and they disagree about
        the phase and about whether the jaw is shut.

        Args:
            at_seconds: Simulated time.

        Returns:
            The phase, the state the flange should be in, and the jaw
            command. None when the plan has run out, which is how the
            caller learns the visit finished.
        """
        elapsed = at_seconds - self.started_at
        if elapsed < 0.0:
            first = self.legs[0]
            return first.phase, first.segment.at(0.0), first.grip
        # Against cumulative boundaries rather than by subtracting each leg
        # in turn. Subtracting accumulates rounding, and the instant that
        # rounding lands on is the end of the plan, which is exactly the
        # instant a caller asks about to learn the visit is over.
        edges = self._boundaries()
        for index, (opened, closed) in enumerate(zip(edges, edges[1:], strict=False)):
            last = index == len(self.legs) - 1
            if elapsed < closed or (last and elapsed <= closed):
                leg = self.legs[index]
                return leg.phase, leg.segment.at(elapsed - opened), leg.grip
        return None

    def _boundaries(self) -> tuple[float, ...]:
        """Return the elapsed time each leg starts at, and the total.

        Returns:
            One more number than there are legs, beginning at zero.
        """
        edges, running = [0.0], 0.0
        for leg in self.legs:
            running += leg.segment.duration
            edges.append(running)
        return tuple(edges)


def plan_pick(
    flange: State,
    track_id: int,
    object_position_belt: NDArray[np.float64] | None = None,
    belt_velocity_world: NDArray[np.float64] | None = None,
    approach_clearance_z: float | None = None,
    approach_speed: float = 0.0,
    dwell_seconds: float = 0.0,
    max_speed: float = 0.0,
    max_acceleration: float = 0.0,
    latest: float = 0.0,
    at_seconds: float = 0.0,
    margin: float = 1.0,
    target_position_world: NDArray[np.float64] | None = None,
    retreat_lift: float | None = None,
    *,
    drift_horizon: float,
    minimum_segment_seconds: float,
    closing_rise_seconds: float,
    segment_sample_count: int,
    bisection_passes: int,
    minimum_delivery_seconds: float,
    correction_steps: int,
    object_position: NDArray[np.float64] | None = None,
    belt_velocity: NDArray[np.float64] | None = None,
    z_offset: float | None = None,
    over: NDArray[np.float64] | None = None,
    belt_width: float = 0.50,
    belt_center_y: float = 0.0,
    belt_border_y: float | None = None,
    safe_height_world: float | None = None,
    cross_speed: float | None = None,
    jaw_rise: float = 0.0,
    clearance_flange_z: float | None = None,
) -> Plan | None:
    """Plan a whole visit, or report that there is no time for one.

    Args:
        flange: Where the flange is and how it is moving.
        track_id: Which track this visit is about.
        object_position_belt: Where the object is now, in belt frame meters.
        belt_velocity_world: How the belt is carrying it.
        approach_clearance_z: Clearance above the object to approach and retreat at.
        approach_speed: How fast to be descending on arrival.
        dwell_seconds: How long the jaw is given to close.
        max_speed: Speed ceiling, in meters per second.
        max_acceleration: Acceleration ceiling, in meters per second squared.
        latest: The longest interception worth considering, in seconds,
            normally what the object has left before it leaves the window.
        at_seconds: Simulated time the plan starts.
        margin: How much longer than the soonest feasible interception to
            take, as a multiple, leaving room for [refine] to correct the
            arc later. One leaves none.
        target_position_world: The chute mouth to release the object over,
            or None to end the visit at the retreat. None drops the object back
            on the belt, which is a line with nowhere to put anything.
        retreat_lift: How far to lift during retreat, in meters. If None,
            defaults to `approach_clearance_z`.
        object_position: Deprecated alias for object_position_belt.
        belt_velocity: Deprecated alias for belt_velocity_world.
        z_offset: Deprecated alias for approach_clearance_z.
        over: Deprecated alias for target_position_world.
        belt_width: Width of the belt in meters, used to compute border position.
        belt_center_y: Lateral center of the belt in meters.
        belt_border_y: Lateral coordinate of the belt border on the chute side,
            or None to compute from belt_center_y - belt_width / 2.0.
        safe_height_world: Flange height giving safe clearance above the belt side
            barrier, in meters, or None to derive from flange or target height.
        cross_speed: Speed across the belt border, in meters per second, or None
            to match approach_speed.
        jaw_rise: How far the flange climbs while the jaw shuts, in meters.
            Zero leaves the hold as a carry at the height the descent arrived at.
            Ignored when `clearance_flange_z` is set.
        clearance_flange_z: The flange height at which the shut jaw is already
            clear of the belt, in world meters. The hold then rises only by
            how far the descent ended below it. None leaves `jaw_rise` in
            force, which is how a test asks for a climb directly.

    Returns:
        The plan, or None when no interception inside `latest` respects both
        ceilings. None is the honest answer for an object the arm cannot
        reach in the belt it has left.
    """
    obj_pos = object_position if object_position is not None else object_position_belt
    if obj_pos is None:
        raise TypeError("plan_pick requires object_position_belt or object_position")
    measured = belt_velocity if belt_velocity is not None else belt_velocity_world
    if measured is None:
        raise TypeError("plan_pick requires belt_velocity_world or belt_velocity")
    # Aim uses drift (lateral for a horizon); matching uses belt-axis transport.
    drift = _drift_velocity(measured)
    transport = _transport(measured)
    clearance = z_offset if z_offset is not None else approach_clearance_z
    if clearance is None:
        raise TypeError("plan_pick requires approach_clearance_z or z_offset")
    target_pos = over if over is not None else target_position_world
    max_accel = max_acceleration
    border_y = (
        belt_border_y
        if belt_border_y is not None
        else (belt_center_y - belt_width / 2.0)
    )
    cross_v = cross_speed if cross_speed is not None else approach_speed

    entry_arc: Segment | None = None
    reaching: Segment | None = None
    obj_pos_at_border = obj_pos

    if target_pos is not None and flange.position[1] <= border_y:
        if safe_height_world is not None:
            safe_height = max(target_pos[2], safe_height_world)
        else:
            safe_height = max(target_pos[2], flange.position[2])
        border_approach = State(
            position=np.asarray(
                (target_pos[0], border_y, safe_height),
                dtype=np.float64,
            ),
            velocity=np.asarray((0.0, cross_v, 0.0), dtype=np.float64),
            acceleration=zeros(),
        )
        entry_arc = _fit_segment(
            flange,
            border_approach,
            max_speed,
            max_accel,
            min_duration=minimum_segment_seconds,
            segment_sample_count=segment_sample_count,
            bisection_passes=bisection_passes,
        )
        if entry_arc.duration >= latest:
            return None
        remaining_latest = latest - entry_arc.duration
        # Along the belt only. The border waypoint stands upstream of the pick
        # and the arc that follows it carries the drift across from there, so
        # extrapolating the lateral component here as well would count it twice.
        obj_pos_at_border = (
            obj_pos
            + transport
            * np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            * entry_arc.duration
        )
        reaching = approach(
            border_approach,
            obj_pos_at_border,
            drift,
            clearance,
            approach_speed,
            max_speed,
            max_accel,
            remaining_latest,
            margin,
            drift_horizon=drift_horizon,
            segment_sample_count=segment_sample_count,
            bisection_passes=bisection_passes,
        )
    else:
        reaching = approach(
            flange,
            obj_pos,
            drift,
            clearance,
            approach_speed,
            max_speed,
            max_accel,
            latest,
            margin,
            drift_horizon=drift_horizon,
            segment_sample_count=segment_sample_count,
            bisection_passes=bisection_passes,
        )

    if reaching is None:
        return None

    return _assemble(
        reaching,
        track_id,
        obj_pos_at_border,
        drift,
        transport,
        clearance,
        approach_speed,
        dwell_seconds,
        at_seconds,
        target_pos,
        max_speed,
        retreat_lift,
        entry_segment=entry_arc,
        belt_border_y=border_y,
        safe_height_world=safe_height_world,
        cross_speed=cross_v,
        max_acceleration=max_accel,
        drift_horizon=drift_horizon,
        minimum_segment_seconds=minimum_segment_seconds,
        closing_rise_seconds=closing_rise_seconds,
        jaw_rise=jaw_rise,
        clearance_flange_z=clearance_flange_z,
        segment_sample_count=segment_sample_count,
        bisection_passes=bisection_passes,
        minimum_delivery_seconds=minimum_delivery_seconds,
        correction_steps=correction_steps,
    )


def refine(
    plan: Plan,
    object_position_belt: NDArray[np.float64] | None = None,
    belt_velocity_world: NDArray[np.float64] | None = None,
    approach_clearance_z: float | None = None,
    approach_speed: float = 0.0,
    dwell_seconds: float = 0.0,
    max_speed: float = 0.0,
    max_acceleration: float = 0.0,
    at_seconds: float = 0.0,
    target_position_world: NDArray[np.float64] | None = None,
    retreat_lift: float | None = None,
    *,
    drift_horizon: float,
    minimum_segment_seconds: float,
    closing_rise_seconds: float,
    segment_sample_count: int,
    bisection_passes: int,
    minimum_delivery_seconds: float,
    correction_steps: int,
    object_position: NDArray[np.float64] | None = None,
    belt_velocity: NDArray[np.float64] | None = None,
    z_offset: float | None = None,
    over: NDArray[np.float64] | None = None,
    belt_width: float = 0.50,
    belt_center_y: float = 0.0,
    belt_border_y: float | None = None,
    safe_height_world: float | None = None,
    cross_speed: float | None = None,
    jaw_rise: float = 0.0,
    clearance_flange_z: float | None = None,
) -> Plan | None:
    """Correct a active plan against a fresher estimate of the object.

    Refinement keeps the arrival time already chosen and solves only for a
    new approach arc that lands on the refreshed position at that instant.
    The descent, the carry and the retreat are then hung off the new end
    exactly as they were off the old one.

    Args:
        plan: The active plan.
        object_position_belt: Where the object is now believed to be.
        belt_velocity_world: How the belt is moving.
        approach_clearance_z: Clearance above the object, in meters.
        approach_speed: How fast the flange is coming down on arrival.
        dwell_seconds: How long the jaw is given to close.
        max_speed: Speed ceiling, in meters per second.
        max_acceleration: Acceleration ceiling, in meters per second squared.
        at_seconds: Current simulated time.
        target_position_world: The chute mouth to release over, carried
            through unchanged.
        retreat_lift: How far to lift during retreat, in meters. If None,
            defaults to `approach_clearance_z`.
        object_position: Deprecated alias for object_position_belt.
        belt_velocity: Deprecated alias for belt_velocity_world.
        z_offset: Deprecated alias for approach_clearance_z.
        over: Deprecated alias for target_position_world.
        belt_width: Width of the belt in meters.
        belt_center_y: Lateral center of the belt in meters.
        belt_border_y: Lateral position of the belt border on the chute side.
        safe_height_world: Safe clearance height above the belt side barrier.
        cross_speed: Speed across the belt border, in meters per second.
        jaw_rise: How far the flange climbs while the jaw shuts, in meters.
            Ignored when `clearance_flange_z` is set.
        clearance_flange_z: The flange height at which the shut jaw is already
            clear of the belt. See [plan_pick].

    Returns:
        The re-aimed plan, or None when there is nothing left to re-aim or
        no positive fraction of the correction stays inside both ceilings.
        A fraction of zero is not a correction, and it is the refusal a
        caller already treats as a re-aim that did not land.
    """
    obj_pos = object_position if object_position is not None else object_position_belt
    if obj_pos is None:
        raise TypeError("refine requires object_position_belt or object_position")
    measured = belt_velocity if belt_velocity is not None else belt_velocity_world
    if measured is None:
        raise TypeError("refine requires belt_velocity_world or belt_velocity")
    drift = _drift_velocity(measured)
    transport = _transport(measured)
    clearance = z_offset if z_offset is not None else approach_clearance_z
    if clearance is None:
        raise TypeError("refine requires approach_clearance_z or z_offset")
    target_pos = over if over is not None else target_position_world
    max_accel = max_acceleration
    border_y = (
        belt_border_y
        if belt_border_y is not None
        else (belt_center_y - belt_width / 2.0)
    )
    cross_v = cross_speed if cross_speed is not None else approach_speed

    track_legs = [leg for leg in plan.legs if leg.phase is Phase.TRACK]
    dt = descent_seconds(clearance, approach_speed)
    elapsed = at_seconds - plan.started_at

    if len(track_legs) > 1:
        entry_leg = track_legs[0].segment
        track_leg = track_legs[1].segment
        if elapsed < entry_leg.duration:
            rem_entry = entry_leg.duration - elapsed
            new_entry_arc = Segment(
                start=entry_leg.at(elapsed),
                end=entry_leg.end,
                duration=rem_entry,
            )
            if not new_entry_arc.fits(max_speed, max_accel, segment_sample_count):
                return None
            # Along the belt only, for the same reason as the entry waypoint.
            obj_pos_at_border = (
                obj_pos
                + transport * np.asarray([1.0, 0.0, 0.0], dtype=np.float64) * rem_entry
            )
            new_track_arc = _feasible_arc(
                start=entry_leg.end,
                kept=track_leg.end,
                wanted=State(
                    position=where_carried(
                        obj_pos_at_border,
                        drift,
                        track_leg.duration + dt,
                        clearance,
                        drift_horizon,
                    ),
                    velocity=np.asarray(
                        (
                            transport[0],
                            transport[1],
                            transport[2] - approach_speed,
                        ),
                        dtype=np.float64,
                    ),
                    acceleration=zeros(),
                ),
                duration=track_leg.duration,
                max_speed=max_speed,
                max_acceleration=max_accel,
                segment_sample_count=segment_sample_count,
                correction_steps=correction_steps,
            )
            if new_track_arc is None:
                return None
            aimed = _object_under(
                new_track_arc.end.position,
                drift,
                new_track_arc.duration + dt,
                clearance,
                drift_horizon,
            )
            return _assemble(
                new_track_arc,
                plan.track_id,
                aimed,
                drift,
                transport,
                clearance,
                approach_speed,
                dwell_seconds,
                at_seconds,
                target_pos,
                max_speed,
                retreat_lift,
                entry_segment=new_entry_arc,
                belt_border_y=border_y,
                safe_height_world=safe_height_world,
                cross_speed=cross_v,
                max_acceleration=max_accel,
                drift_horizon=drift_horizon,
                minimum_segment_seconds=minimum_segment_seconds,
                closing_rise_seconds=closing_rise_seconds,
                jaw_rise=jaw_rise,
                clearance_flange_z=clearance_flange_z,
                segment_sample_count=segment_sample_count,
                bisection_passes=bisection_passes,
                minimum_delivery_seconds=minimum_delivery_seconds,
                correction_steps=correction_steps,
            )
        if elapsed < entry_leg.duration + track_leg.duration:
            rem_track = (entry_leg.duration + track_leg.duration) - elapsed
            track_elapsed = elapsed - entry_leg.duration
            current_state = track_leg.at(track_elapsed)
            new_track_arc = _feasible_arc(
                start=current_state,
                kept=track_leg.end,
                wanted=State(
                    position=where_carried(
                        obj_pos,
                        drift,
                        rem_track + dt,
                        clearance,
                        drift_horizon,
                    ),
                    velocity=np.asarray(
                        (
                            transport[0],
                            transport[1],
                            transport[2] - approach_speed,
                        ),
                        dtype=np.float64,
                    ),
                    acceleration=zeros(),
                ),
                duration=rem_track,
                max_speed=max_speed,
                max_acceleration=max_accel,
                segment_sample_count=segment_sample_count,
                correction_steps=correction_steps,
            )
            if new_track_arc is None:
                return None
            aimed = _object_under(
                new_track_arc.end.position,
                drift,
                new_track_arc.duration + dt,
                clearance,
                drift_horizon,
            )
            return _assemble(
                new_track_arc,
                plan.track_id,
                aimed,
                drift,
                transport,
                clearance,
                approach_speed,
                dwell_seconds,
                at_seconds,
                target_pos,
                max_speed,
                retreat_lift,
                belt_border_y=border_y,
                safe_height_world=safe_height_world,
                cross_speed=cross_v,
                max_acceleration=max_accel,
                drift_horizon=drift_horizon,
                minimum_segment_seconds=minimum_segment_seconds,
                closing_rise_seconds=closing_rise_seconds,
                jaw_rise=jaw_rise,
                clearance_flange_z=clearance_flange_z,
                segment_sample_count=segment_sample_count,
                bisection_passes=bisection_passes,
                minimum_delivery_seconds=minimum_delivery_seconds,
                correction_steps=correction_steps,
            )
        return None

    reaching = track_legs[0].segment
    remaining = reaching.duration - elapsed
    if remaining <= 0.0:
        return None
    arc = _feasible_arc(
        start=reaching.at(elapsed),
        kept=reaching.end,
        wanted=State(
            position=where_carried(
                obj_pos,
                drift,
                remaining + dt,
                clearance,
                drift_horizon,
            ),
            velocity=np.asarray(
                (
                    transport[0],
                    transport[1],
                    transport[2] - approach_speed,
                ),
                dtype=np.float64,
            ),
            acceleration=zeros(),
        ),
        duration=remaining,
        max_speed=max_speed,
        max_acceleration=max_accel,
        segment_sample_count=segment_sample_count,
        correction_steps=correction_steps,
    )
    if arc is None:
        return None
    aimed = _object_under(
        arc.end.position, drift, arc.duration + dt, clearance, drift_horizon
    )
    return _assemble(
        arc,
        plan.track_id,
        aimed,
        drift,
        transport,
        clearance,
        approach_speed,
        dwell_seconds,
        at_seconds,
        target_pos,
        max_speed,
        retreat_lift,
        belt_border_y=border_y,
        safe_height_world=safe_height_world,
        cross_speed=cross_v,
        max_acceleration=max_accel,
        drift_horizon=drift_horizon,
        minimum_segment_seconds=minimum_segment_seconds,
        closing_rise_seconds=closing_rise_seconds,
        jaw_rise=jaw_rise,
        clearance_flange_z=clearance_flange_z,
        segment_sample_count=segment_sample_count,
        bisection_passes=bisection_passes,
        minimum_delivery_seconds=minimum_delivery_seconds,
        correction_steps=correction_steps,
    )


def retarget_descent(
    plan: Plan,
    object_position: NDArray[np.float64],
    belt_velocity: NDArray[np.float64],
    approach_clearance_z: float,
    approach_speed: float,
    dwell_seconds: float,
    max_speed: float,
    max_acceleration: float,
    at_seconds: float,
    retreat_lift: float | None = None,
    over: NDArray[np.float64] | None = None,
    belt_border_y: float | None = None,
    safe_height_world: float | None = None,
    cross_speed: float | None = None,
    jaw_rise: float = 0.0,
    clearance_flange_z: float | None = None,
    *,
    drift_horizon: float,
    minimum_segment_seconds: float,
    closing_rise_seconds: float,
    segment_sample_count: int,
    bisection_passes: int,
    minimum_delivery_seconds: float,
    correction_steps: int,
) -> Plan | None:
    """Point the descent already in progress at where the object is now.

    The approach was solved against an earlier estimate. Once the flange is
    on the way down, that estimate is at most one steer old, and the object
    has whatever velocity it has now. Rebuilding the whole visit would send
    the arm back up. The arrival stays the instant already chosen. What moves
    is the end of the descent, and the carry and the retreat are hung off
    that end the same way a fresh plan would.

    Args:
        plan: The active visit, already on its descent.
        object_position: Where the object is now.
        belt_velocity: How it is moving.
        approach_clearance_z: The clearance the descent was planned through.
        approach_speed: How fast the descent was coming down.
        dwell_seconds: How long the jaw stays shut.
        max_speed: Speed ceiling, in meters per second.
        max_acceleration: Acceleration ceiling, in meters per second squared.
        at_seconds: Simulated time.
        drift_horizon: How long a lateral velocity is carried.
        retreat_lift: How far the retreat climbs, when it differs from the
            clearance.
        over: The chute mouth the delivery was heading for, or None.
        belt_border_y: The belt edge the delivery crosses.
        safe_height_world: The height the delivery clears the barrier at.
        cross_speed: How fast the delivery crosses that edge.
        jaw_rise: How far the flange climbs while the jaw shuts, in meters.
            Ignored when `clearance_flange_z` is set.
        clearance_flange_z: The flange height at which the shut jaw is already
            clear of the belt. See [plan_pick].

    Returns:
        The visit from this instant on, or None when this instant is not
        inside the descent or no positive fraction of the correction stays
        under the speed ceiling. None leaves the caller on the plan it
        already has. The descent's duration is already fixed, and that
        duration is what sets its acceleration, so the acceleration ceiling
        that gates an approach does not gate this splice: the descent
        already in hand is past it.
    """
    transport = _transport(belt_velocity)
    elapsed = at_seconds - plan.started_at
    edges = plan._boundaries()
    descent_index = next(
        (index for index, leg in enumerate(plan.legs) if leg.phase is Phase.DESCEND),
        None,
    )
    if descent_index is None:
        return None
    opened, closed = edges[descent_index], edges[descent_index + 1]
    if elapsed < opened or elapsed >= closed:
        return None
    into = elapsed - opened
    remaining = closed - elapsed
    if remaining <= 1e-3:
        return None
    leg = plan.legs[descent_index]
    here = leg.segment.at(into)
    # The duration is the time left, not a duration searched for, so the
    # acceleration is whatever that time produces. The descent this replaces
    # was accepted on the same terms: its duration comes from the clearance
    # and the approach speed, and its peak acceleration is already past the
    # ceiling that gates an approach. What a correction can break is the
    # speed ceiling, and the part of the correction that stays under it is
    # the part that flies. None of it fitting leaves the plan already in hand.
    #
    # Aim with belt-axis transport only (`AC-BEHAVE-02`). Projecting lateral
    # drift over a few hundred milliseconds of remaining descent invented the
    # cross-belt jab measured on seed 0 (peak |vy| ≈ 0.94 m/s at grasp height).
    # The fresher object position still moves the end (AC-MOVE-63); only the
    # invented future lateral is dropped.
    dropping = _feasible_arc(
        start=here,
        kept=leg.segment.end,
        wanted=State(
            position=where_carried(
                object_position,
                transport,
                remaining,
                0.0,
                drift_horizon,
            ),
            velocity=transport,
            acceleration=zeros(),
        ),
        duration=remaining,
        max_speed=max_speed,
        max_acceleration=None,
        segment_sample_count=segment_sample_count,
        correction_steps=correction_steps,
    )
    if dropping is None:
        return None
    ceiling = max_acceleration
    rise = _closing_rise(dropping.end.position[2], jaw_rise, clearance_flange_z)
    held = _hold(dropping.end, dwell_seconds, transport, rise, closing_rise_seconds)
    rising = _retreat(
        held[-1].segment.end, approach_clearance_z, approach_speed, transport
    )
    legs = [
        Leg(Phase.DESCEND, dropping, leg.grip),
        *held,
        Leg(Phase.RETREAT, rising, JAW_SHUT),
    ]
    if over is not None and safe_height_world is not None and belt_border_y is not None:
        cross_v = approach_speed if cross_speed is None else cross_speed
        lift = approach_clearance_z if retreat_lift is None else retreat_lift
        safe_height = max(over[2], safe_height_world)
        climbing = _climb(
            rising.end,
            max(safe_height, held[-1].segment.end.position[2] + lift),
            transport,
            max_speed,
            ceiling,
            segment_sample_count=segment_sample_count,
            bisection_passes=bisection_passes,
            minimum_delivery_seconds=minimum_delivery_seconds,
        )
        legs.append(Leg(Phase.RETREAT, climbing, JAW_SHUT))
        border = State(
            position=np.asarray(
                (over[0], belt_border_y, safe_height),
                dtype=np.float64,
            ),
            velocity=np.asarray((0.0, -cross_v, 0.0), dtype=np.float64),
            acceleration=zeros(),
        )
        chute = State(
            position=over,
            velocity=zeros(),
            acceleration=zeros(),
        )
        legs.append(
            Leg(
                Phase.DELIVER,
                _fit_segment(
                    climbing.end,
                    border,
                    max_speed,
                    ceiling,
                    min_duration=minimum_segment_seconds,
                    segment_sample_count=segment_sample_count,
                    bisection_passes=bisection_passes,
                ),
                JAW_SHUT,
            )
        )
        legs.append(
            Leg(
                Phase.DELIVER,
                _fit_segment(
                    border,
                    chute,
                    max_speed,
                    ceiling,
                    min_duration=minimum_delivery_seconds,
                    segment_sample_count=segment_sample_count,
                    bisection_passes=bisection_passes,
                ),
                JAW_SHUT,
            )
        )
    return Plan(
        track_id=plan.track_id,
        legs=tuple(legs),
        started_at=at_seconds,
        pick_at=at_seconds + remaining,
        target_yaw=plan.target_yaw,
        initial_yaw=plan.initial_yaw,
        transport_velocity=transport,
    )


def _deliver(
    start: State,
    target_position_world: NDArray[np.float64],
    max_speed: float,
    *,
    minimum_delivery_seconds: float,
) -> Segment:
    """Return the arc that carries the object to its chute and lets go.

    Args:
        start: Where the retreat ended, at rest above the belt.
        target_position_world: The chute mouth to release over.
        max_speed: Speed ceiling, in meters per second.

    Returns:
        The arc, ending at rest over the mouth.
    """
    span = distance(start.position, target_position_world)
    seconds = max(PEAK_OVER_MEAN * span / max_speed, minimum_delivery_seconds)
    return Segment(
        start=start,
        end=State(
            position=target_position_world,
            velocity=zeros(),
            acceleration=zeros(),
        ),
        duration=seconds,
    )


def _assemble(
    reaching: Segment,
    track_id: int,
    object_position_belt: NDArray[np.float64],
    aim_velocity: NDArray[np.float64],
    transport_velocity: NDArray[np.float64],
    approach_clearance_z: float,
    approach_speed: float,
    dwell_seconds: float,
    at_seconds: float,
    target_position_world: NDArray[np.float64] | None = None,
    max_speed: float = 1.0,
    retreat_lift: float | None = None,
    *,
    drift_horizon: float,
    minimum_segment_seconds: float,
    closing_rise_seconds: float,
    segment_sample_count: int,
    bisection_passes: int,
    minimum_delivery_seconds: float,
    correction_steps: int,
    entry_segment: Segment | None = None,
    belt_border_y: float = -0.25,
    safe_height_world: float | None = None,
    cross_speed: float | None = None,
    max_acceleration: float,
    jaw_rise: float = 0.0,
    clearance_flange_z: float | None = None,
) -> Plan:
    """Hang the descent, the carry, the retreat and delivery off an approach arc.

    Args:
        reaching: The approach tracking arc.
        track_id: Which track the visit is about.
        object_position_belt: Where the object is at `at_seconds`.
        aim_velocity: Drift-aware velocity for aiming the target origin.
        transport_velocity: Belt-axis model transport the jaw matches.
        approach_clearance_z: Clearance above the object, in meters.
        approach_speed: How fast the flange is coming down on arrival.
        dwell_seconds: How long the jaw is given to close.
        at_seconds: Simulated time the approach begins.
        target_position_world: Where to release the object over.
        max_speed: Speed ceiling in meters per second.
        retreat_lift: How far to lift during retreat, in meters.
        entry_segment: Preceding arc to the intermediary border waypoint.
        belt_border_y: Lateral coordinate of the belt border on the chute side.
        safe_height_world: Safe clearance height above the belt side barrier.
        cross_speed: Speed across the belt border, in meters per second.
        max_acceleration: Acceleration ceiling in meters per second squared.
        jaw_rise: How far the flange climbs while the jaw shuts, in meters.

    Returns:
        The whole visit.
    """
    cross_v = cross_speed if cross_speed is not None else approach_speed
    dropping = descend(
        reaching,
        object_position_belt,
        aim_velocity,
        approach_clearance_z,
        approach_speed,
        drift_horizon=drift_horizon,
    )
    if not same(dropping.end.velocity, transport_velocity):
        dropping = Segment(
            start=dropping.start,
            end=State(
                position=dropping.end.position,
                velocity=transport_velocity,
                acceleration=dropping.end.acceleration,
            ),
            duration=dropping.duration,
        )
    rise = _closing_rise(dropping.end.position[2], jaw_rise, clearance_flange_z)
    held = _hold(
        dropping.end, dwell_seconds, transport_velocity, rise, closing_rise_seconds
    )
    rising = _retreat(
        held[-1].segment.end, approach_clearance_z, approach_speed, transport_velocity
    )
    legs = []
    if entry_segment is not None:
        legs.append(Leg(Phase.TRACK, entry_segment, JAW_OPEN))
    legs.extend(
        [
            Leg(Phase.TRACK, reaching, JAW_OPEN),
            Leg(Phase.DESCEND, dropping, JAW_OPEN),
            *held,
            Leg(Phase.RETREAT, rising, JAW_SHUT),
        ]
    )
    if target_position_world is not None:
        if safe_height_world is not None:
            safe_height = max(target_position_world[2], safe_height_world)
        else:
            safe_height = max(target_position_world[2], reaching.start.position[2])
        reach_up = approach_clearance_z if retreat_lift is None else retreat_lift
        # Delivery is interception with v_T = 0 (AC-MOVE-70).
        climbing = _climb(
            rising.end,
            max(safe_height, held[-1].segment.end.position[2] + reach_up),
            transport_velocity,
            max_speed,
            max_acceleration,
            segment_sample_count=segment_sample_count,
            bisection_passes=bisection_passes,
            minimum_delivery_seconds=minimum_delivery_seconds,
        )
        legs.append(Leg(Phase.RETREAT, climbing, JAW_SHUT))
        border_retreat = State(
            position=np.asarray(
                (target_position_world[0], belt_border_y, safe_height), dtype=np.float64
            ),
            velocity=np.asarray((0.0, -cross_v, 0.0), dtype=np.float64),
            acceleration=zeros(),
        )
        chute_target = State(
            position=target_position_world,
            velocity=zeros(),
            acceleration=zeros(),
        )
        seg_to_border = _fit_segment(
            climbing.end,
            border_retreat,
            max_speed,
            max_acceleration,
            min_duration=minimum_segment_seconds,
            segment_sample_count=segment_sample_count,
            bisection_passes=bisection_passes,
        )
        seg_to_chute = _fit_segment(
            border_retreat,
            chute_target,
            max_speed,
            max_acceleration,
            min_duration=minimum_delivery_seconds,
            segment_sample_count=segment_sample_count,
            bisection_passes=bisection_passes,
        )
        legs.append(Leg(Phase.DELIVER, seg_to_border, JAW_SHUT))
        legs.append(Leg(Phase.DELIVER, seg_to_chute, JAW_SHUT))
    entry_duration = entry_segment.duration if entry_segment is not None else 0.0
    return Plan(
        track_id=track_id,
        legs=tuple(legs),
        started_at=at_seconds,
        pick_at=at_seconds + entry_duration + reaching.duration + dropping.duration,
        transport_velocity=transport_velocity,
    )


def _object_under(
    end: NDArray[np.float64],
    velocity: NDArray[np.float64],
    seconds: float,
    clearance: float,
    drift_horizon: float,
) -> NDArray[np.float64]:
    """Return the object a carried aim at `end` was predicted from.

    This is `where_carried` run backwards over `seconds` at `clearance`. A
    partial correction blends two aims, and the descent has to come down onto
    that blend. Handing it the object the whole correction wanted sends the
    descent sideways across the gap the approach was not allowed to close.

    Args:
        end: Where the approach arc finishes, clearance above the object.
        velocity: How the object is moving.
        seconds: How far ahead the aim was carried, which is the approach
            plus the descent. The approach finishes where the object will be
            when the jaw arrives, not when the descent begins.
        clearance: How far above the object the arc finishes, in meters.
        drift_horizon: How long a lateral velocity is carried.

    Returns:
        The object position the arc is aiming at, at the arc's own start.
    """
    across = min(seconds, drift_horizon)
    return np.asarray(
        (
            end[0] - velocity[0] * seconds,
            end[1] - velocity[1] * across,
            end[2] - clearance,
        ),
        dtype=np.float64,
    )


def _mix(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Return the point `alpha` of the way from `left` to `right`."""
    return (1.0 - alpha) * left + alpha * right


def _blend(left: State, right: State, alpha: float) -> State:
    """Return the state `alpha` of the way from `left` to `right`."""
    return State(
        position=_mix(left.position, right.position, alpha),
        velocity=_mix(left.velocity, right.velocity, alpha),
        acceleration=_mix(left.acceleration, right.acceleration, alpha),
    )


def _feasible_arc(
    start: State,
    kept: State,
    wanted: State,
    duration: float,
    max_speed: float,
    max_acceleration: float | None,
    *,
    segment_sample_count: int,
    correction_steps: int,
) -> Segment | None:
    """Return the arc toward `wanted` that stays inside the ceiling.

    `kept` is the end already in hand. The largest fraction of the way from
    `kept` to `wanted` whose quintic respects the ceiling is the one flown.
    A fraction of zero is not a correction, and the caller keeps the plan it
    already has rather than replacing it with a re-fit of the same end.

    Args:
        start: Where the arc begins.
        kept: The end the plan already aims at.
        wanted: The end the freshest estimate asks for.
        duration: How long the arc has, in seconds. Already chosen.
        max_speed: Speed ceiling, in meters per second.
        max_acceleration: Acceleration ceiling, in meters per second squared,
            or None when only the speed ceiling applies. A descent's duration
            is fixed by the clearance and the approach speed, and that
            duration is what sets its acceleration, so the acceleration
            ceiling is not what refuses it.

    Returns:
        The arc, or None when no positive fraction fits.
    """
    best: Segment | None = None
    for step in range(correction_steps + 1):
        alpha = step / correction_steps
        if alpha == 0.0:
            continue
        arc = Segment(start=start, end=_blend(kept, wanted, alpha), duration=duration)
        fits = (
            arc.peak_speed(segment_sample_count) <= max_speed
            if max_acceleration is None
            else arc.fits(max_speed, max_acceleration, segment_sample_count)
        )
        if fits:
            best = arc
    return best


def _rise(
    start: State,
    seconds: float,
    belt_velocity: NDArray[np.float64],
    lift: float,
) -> Segment:
    """Return the arc that raises the flange while the jaw is closing.

    Args:
        start: Where the descent ended, already moving with the belt.
        seconds: How long the rise takes.
        belt_velocity: How the belt is moving.
        lift: How far the flange climbs, in meters.

    Returns:
        The arc. Vertical velocity is zero at both ends, so it chains with
        a descent that arrived moving only with the belt and with the carry
        that follows.
    """
    return Segment(
        start=start,
        end=State(
            position=(
                start.position
                + belt_velocity * seconds
                + np.asarray([0.0, 0.0, lift], dtype=np.float64)
            ),
            velocity=belt_velocity,
            acceleration=zeros(),
        ),
        duration=seconds,
    )


def _closing_rise(
    flange_z: float, jaw_rise: float, clearance_flange_z: float | None
) -> float:
    """Return how far the hold climbs from a descent that ended at `flange_z`.

    Args:
        flange_z: Where the descent ended, in world meters.
        jaw_rise: The climb a caller asked for directly.
        clearance_flange_z: The flange height at which the shut jaw is already
            clear, or None to use `jaw_rise` unchanged.

    Returns:
        The climb, in meters. Zero when the shut jaw is already clear, so a
        grasp the pads can drop onto is not lifted off.
    """
    if clearance_flange_z is None:
        return jaw_rise
    short = clearance_flange_z - flange_z
    return 0.0 if short <= 1e-6 else short


def _hold(
    start: State,
    seconds: float,
    belt_velocity: NDArray[np.float64],
    lift: float,
    closing_rise_seconds: float,
) -> tuple[Leg, ...]:
    """Return the hold: a rise while the jaw shuts, then a carry.

    Args:
        start: Where the descent ended.
        seconds: How long the jaw is given to close.
        belt_velocity: How the belt is moving.
        lift: How far the flange climbs during the close, in meters. Zero
            leaves the hold as the carry it was.

    Returns:
        One or two hold legs. The jaw is commanded shut on both.
    """
    if lift <= 1e-6 or seconds <= 1e-3:
        carry = _carry(start, max(seconds, 1e-3), belt_velocity)
        return (Leg(Phase.HOLD, carry, JAW_SHUT),)
    rise_for = min(closing_rise_seconds, seconds)
    risen = _rise(start, rise_for, belt_velocity, lift)
    legs = [Leg(Phase.HOLD, risen, JAW_SHUT)]
    rest = seconds - rise_for
    if rest > 1e-3:
        legs.append(Leg(Phase.HOLD, _carry(risen.end, rest, belt_velocity), JAW_SHUT))
    return tuple(legs)


def _carry(start: State, seconds: float, belt_velocity: NDArray[np.float64]) -> Segment:
    """Return the arc that rides along with the belt while the jaw closes.

    Args:
        start: Where the descent ended, already moving with the belt.
        seconds: How long to carry for.
        belt_velocity: How the belt is moving.

    Returns:
        The arc. Its endpoints differ by exactly the velocity times the
        duration, with that velocity at both ends and no acceleration, so
        the quintic reduces to a straight line at constant speed.
    """
    return Segment(
        start=start,
        end=State(
            position=start.position + belt_velocity * seconds,
            velocity=belt_velocity,
            acceleration=zeros(),
        ),
        duration=seconds,
    )


def _retreat(
    start: State,
    z_offset: float,
    approach_speed: float,
    belt_velocity: NDArray[np.float64],
) -> Segment:
    """Return the descent read backwards: the arc that leaves the object.

    The descent arrives at the object moving with it, so the retreat leaves the
    object moving with it and arrives at the clearance still rising at the
    speed the descent came down at. Sharing the clearance means sharing the
    duration, and it means the two arcs are one arc read in opposite
    directions.

    Ending at rest is not a neutral alternative. The object's frame carries the
    belt and the flange does not, so a retreat that ends at rest slides
    backwards through the whole lift relative to the object it is holding: for
    as long as the lift lasts.

    Args:
        start: Where the hold ended, moving with the belt.
        z_offset: The clearance the descent came down through, in meters.
        approach_speed: The speed the descent came down at, which the retreat
            leaves at.
        belt_velocity: Model transport (belt-axis only).

    Returns:
        The arc: vertical in the object target frame (+Z over the descent
        duration), composed with transport into world.
    """
    seconds = descent_seconds(z_offset, approach_speed)
    return Segment(
        start=start,
        end=State(
            position=(
                start.position
                + belt_velocity * seconds
                + np.asarray([0.0, 0.0, z_offset], dtype=np.float64)
            ),
            velocity=belt_velocity
            + np.asarray([0.0, 0.0, approach_speed], dtype=np.float64),
            acceleration=zeros(),
        ),
        duration=seconds,
    )


def _climb(
    start: State,
    target_z: float,
    belt_velocity: NDArray[np.float64],
    max_speed: float,
    max_acceleration: float,
    *,
    segment_sample_count: int,
    bisection_passes: int,
    minimum_delivery_seconds: float,
) -> Segment:
    """Return the arc that lifts the object to the height the delivery crosses at.

    An arc of its own rather than part of the retreat, because it exists to
    clear the belt's side barrier and not to leave the object: folding the two
    together is what made the retreat's shape depend on where the next chute
    happens to stand.

    It ends moving with the belt and level, so the crossing that follows starts
    from a state the belt keeps rather than from whatever vertical speed the
    lift happened to finish on.

    Args:
        start: Where the retreat ended, moving with the belt and rising.
        target_z: The height to climb to, in world frame meters.
        belt_velocity: How the belt is moving.
        max_speed: Speed ceiling, in meters per second.
        max_acceleration: Acceleration ceiling, in meters per second squared.

    Returns:
        The arc. Its duration is found by the monotone feasibility search,
        because the distance it has to cover grows with the duration: the belt
        keeps carrying the object while the flange climbs. Longer is always
        easier, so the smallest duration that fits both ceilings is a
        bisection rather than a step over a range.
    """
    ceiling = max(max_speed, 1e-6)
    level = np.asarray(
        [start.position[0], start.position[1], target_z],
        dtype=np.float64,
    )
    span = float(np.linalg.norm(level - start.position))
    lower = max(PEAK_OVER_MEAN * span / ceiling, minimum_delivery_seconds)

    def arc(seconds: float) -> Segment:
        return Segment(
            start=start,
            end=State(
                position=np.concatenate(
                    (
                        start.position[:2] + belt_velocity[:2] * seconds,
                        np.asarray([target_z], dtype=np.float64),
                    )
                ),
                velocity=belt_velocity,
                acceleration=zeros(),
            ),
            duration=seconds,
        )

    def feasible(seconds: float) -> bool:
        return arc(seconds).fits(max_speed, max_acceleration, segment_sample_count)

    return arc(soonest_feasible(feasible, lower, bisection_passes))
