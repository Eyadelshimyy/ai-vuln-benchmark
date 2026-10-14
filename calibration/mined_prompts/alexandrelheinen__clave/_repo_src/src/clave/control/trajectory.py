"""Trajectories with known duration, so an interception time can be solved for.

Motion until now stepped a reference toward a goal under a speed and an
acceleration bound, which is enough to arrive somewhere and not enough to
pick anything up. Arriving is not the problem: a jaw has to reach an object
at a known instant, moving with it, having come straight down onto it. That
needs a trajectory whose duration is a parameter rather than an outcome.

**Quintic Hermite segments, one per axis.** A quintic has six coefficients
and a segment has six boundary conditions per axis: position, velocity and
acceleration at each end. So the polynomial is determined exactly by what
the phase asks for, in closed form, with no fitting and no solver. That is
also why a quintic and not a B-spline: the requirement is control of
acceleration at the waypoints, and a Hermite form states the acceleration
there as a coefficient rather than approaching it through control points.

**The interception time is a fixed point.** Where the arm must be depends on
how long it takes to get there, because the belt carries the object while
the arm travels. Written out, the approach point is

    p_approach(T) = p_object(t0) + v_object * (T + dt) + (0, 0, z_offset)

and the segment reaching it has to respect the speed and acceleration
ceilings over its whole length. Peak speed and peak acceleration both fall
as `T` grows, so the smallest feasible `T` is found by bisection on a
monotone predicate rather than by an optimiser. It is suboptimal and it is
knowably suboptimal: what the pick needs is a duration it can count on, not
the shortest one that exists.

**The descent duration is not free.** In the frame moving with the belt the
descent is purely vertical, from height `z_offset` at speed `V_approach` to
rest on the object. Writing the quintic in `u = 1 - s` and reading the
leading term, which is cubic because position, velocity and acceleration
all vanish at the end, gives the exact condition for never passing below
the object:

    dt <= 5 * z_offset / (2 * V_approach)

This uses `2 * z_offset / V_approach`, four fifths of that limit, which
leaves margin rather than sitting on a boundary. Acceleration is not the
binding constraint here, so there is nothing to buy by stretching toward it,
and a shorter descent keeps the prediction horizon short, which is the
reason the clearance is small in the first place.

**Both ends move with the belt.** The terminal velocity at the pick is the
object's own, not zero, and the approach velocity is the object's plus the
descent. Two things follow and both are large. A jaw arriving at rest has
the object sliding through it at belt speed, which is the one thing a grasp
cannot tolerate. And the horizontal catch-up a stationary approach forces
dominates the descent: matching the object instead dropped peak acceleration
Peak acceleration falls with a longer descent, and the slip at the jaw
goes to zero when the flange moves with the belt.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

_ZERO = np.zeros(3, dtype=np.float64)
"""Three zeros, used only as a template for `zeros()` copies."""


def zeros() -> NDArray[np.float64]:
    """Return a fresh three-component zero vector.

    Returns:
        A new array, never a shared mutable default.
    """
    return _ZERO.copy()


def same(
    one: NDArray[np.float64],
    other: NDArray[np.float64],
    *,
    atol: float = 1e-12,
) -> bool:
    """Report whether two spatial vectors agree within a tolerance.

    Args:
        one: A vector.
        other: Another.
        atol: Absolute tolerance, in the same units as the components.

    Returns:
        Whether every component agrees within `atol`. Bare `==` on arrays is
        element-wise and ambiguous as a boolean, so callers use this instead.
    """
    return bool(np.allclose(one, other, rtol=0.0, atol=atol))


def norm(vector: NDArray[np.float64]) -> float:
    """Return the length of a vector.

    Args:
        vector: The vector.

    Returns:
        Its Euclidean norm, as a Python float.
    """
    return float(np.linalg.norm(vector))


def distance(one: NDArray[np.float64], other: NDArray[np.float64]) -> float:
    """Return the distance between two positions.

    Args:
        one: A position.
        other: Another.

    Returns:
        The Euclidean distance between them, as a Python float.
    """
    return float(np.linalg.norm(one - other))


def clip(value: float, ceiling: float) -> float:
    """Return a value held between zero and a ceiling.

    Args:
        value: The value.
        ceiling: The most it may be.

    Returns:
        `value` with a floor at zero and that ceiling, which is what every
        speed, torque and width in this package is held to.
    """
    return float(np.clip(value, 0.0, ceiling))


def bisect_feasible(
    feasible: Callable[[float], bool],
    low: float,
    high: float,
    passes: int,
) -> float:
    """Return the smallest value in a bracket whose predicate holds.

    A predicate asking whether a duration respects a set of ceilings is
    monotone -- peak speed and peak acceleration of a quintic both fall as its
    duration grows, so every longer duration fits at least as well -- which is
    why the answer is a bisection and not a general root find. It is also why
    scipy is not used here: `scipy.optimize.bisect` and `brentq` solve
    `f(x) == 0` for a continuous `f`, and what is being searched here is a
    boolean, for which bisection on monotonicity is exact and simpler than
    anything a solver would do with it.

    Args:
        feasible: Whether a duration satisfies the ceilings. Monotone: false
            below the answer and true at or above it.
        low: A duration known to be infeasible.
        high: A duration known to be feasible.
        passes: How many halvings to take.

    Returns:
        The smallest feasible duration the bracket resolves to, which is `high`
        itself when the bracket is already tight.
    """
    for _ in range(passes):
        middle = 0.5 * (low + high)
        if middle <= low:
            break
        if feasible(middle):
            high = middle
        else:
            low = middle
    return high


def soonest_feasible(
    feasible: Callable[[float], bool],
    low: float,
    passes: int,
) -> float:
    """Return the smallest duration at or above `low` whose predicate holds.

    `bisect_feasible` solves a bracket once a caller has both ends. A duration
    search that starts from a lower bound has only the failing end, so this
    grows the passing end by doubling until the predicate holds, then bisects
    between the two. Doubling rather than a fixed step because the feasible
    duration has no natural ceiling here the way a belt window bounds an
    interception: it could be any size, and doubling reaches it in a
    logarithmic number of steps.

    Args:
        feasible: Whether a duration satisfies the ceilings. Monotone: false
            below the answer and true at or above it.
        low: A lower bound at or below the answer. Must be positive, because
            the bracket grows from it by multiplication.

    Returns:
        The smallest feasible duration at or above `low`, which is `low`
        itself when `low` already fits.

    Raises:
        ValueError: If `low` is not positive.
    """
    if low <= 0.0:
        raise ValueError(f"a duration lower bound of {low} is not positive")
    if feasible(low):
        return low
    high = low
    while not feasible(high):
        high *= 2.0
    return bisect_feasible(feasible, low, high, passes)


@dataclass(frozen=True)
class State:
    """Where the flange is, how fast, and how hard it is changing.

    Attributes:
        position: In belt frame meters.
        velocity: In meters per second.
        acceleration: In meters per second squared.
    """

    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return (
            same(self.position, other.position)
            and same(self.velocity, other.velocity)
            and same(self.acceleration, other.acceleration)
        )

    @classmethod
    def at_rest(cls, position: NDArray[np.float64]) -> State:
        """Return a state standing still at a position.

        Args:
            position: Where it stands.

        Returns:
            The state, with no velocity and no acceleration.
        """
        return cls(position=position, velocity=zeros(), acceleration=zeros())


@dataclass(frozen=True)
class Segment:
    """One quintic Hermite arc, with the duration it takes.

    Attributes:
        start: The boundary condition at the beginning.
        end: The boundary condition at the end.
        duration: How long it lasts, in seconds.
    """

    start: State
    end: State
    duration: float

    def at(self, elapsed: float) -> State:
        """Return the state this many seconds in.

        Args:
            elapsed: Seconds since the segment began. Clamped to the
                segment, so asking past the end returns the end rather than
                extrapolating a quintic, which diverges fast.

        Returns:
            The state.
        """
        # One instant, evaluated in closed form. `sample` is the same polynomial
        # over a whole grid and is what a peak search wants; sending the physics
        # tick through it builds a basis matrix for a single row, and that tick
        # is two milliseconds. The two agree to a floating-point rounding, which
        # `test_one_instant_matches_the_sampled_polynomial` guards.
        span = max(self.duration, 1e-12)
        s = min(max(elapsed / span, 0.0), 1.0)
        weights = _basis_at(s)
        d_weights = _first_at(s)
        dd_weights = _second_at(s)
        position = np.empty(3, dtype=np.float64)
        velocity = np.empty(3, dtype=np.float64)
        acceleration = np.empty(3, dtype=np.float64)
        for axis in range(3):
            terms = _axis_terms(self, axis, span)
            position[axis] = _dot(weights, terms)
            velocity[axis] = _dot(d_weights, terms) / span
            acceleration[axis] = _dot(dd_weights, terms) / (span * span)
        return State(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )

    def sample(
        self, elapsed: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the state at each of many instants, in one pass.

        The five boundary conditions of a quintic Hermite arc are six terms per
        axis, and the basis weights depend only on normalized time. Writing the
        terms as a `(6, 3)` array -- position, velocity and acceleration at each
        end, each carrying its power of the span -- turns the whole evaluation
        into one matrix product per quantity, over as many instants as the
        caller likes. The alternative, which this replaced, was a loop over
        three axes with six terms summed element-wise inside it, run once per
        sampled instant.

        Args:
            elapsed: Seconds since the segment began, per instant. Clamped to
                the segment, so asking past the end returns the end rather
                than extrapolating a quintic, which diverges fast.

        Returns:
            The position, velocity and acceleration at each instant, each of
            shape `(len(elapsed), 3)`.
        """
        span = max(self.duration, 1e-12)
        s = np.clip(np.asarray(elapsed, dtype=np.float64) / span, 0.0, 1.0)
        return _evaluate(self, span, s)

    def peaks(self, sample_count: int) -> tuple[float, float]:
        """Return the largest speed and acceleration anywhere on the segment.

        Returns:
            The peak speed in meters per second and the peak acceleration in
            meters per second squared, both as the largest Euclidean norm over
            a sampling of the segment.
        """
        # Normalized open grid, cached per sample count (`AC-PERF-04`). The
        # same Hermite basis as `sample`, without a fresh linspace or a
        # clip-and-divide on every feasibility check.
        span = max(self.duration, 1e-12)
        _, velocity, acceleration = _evaluate(self, span, _unit_grid(sample_count))
        return (
            float(np.linalg.norm(velocity, axis=1).max()),
            float(np.linalg.norm(acceleration, axis=1).max()),
        )

    def peak_speed(self, sample_count: int) -> float:
        """Return the largest speed anywhere on the segment, in meters per second."""
        return self.peaks(sample_count)[0]

    def peak_acceleration(self, sample_count: int) -> float:
        """Return the largest acceleration anywhere, in meters per second squared."""
        return self.peaks(sample_count)[1]

    def fits(
        self, max_speed: float, max_acceleration: float, sample_count: int
    ) -> bool:
        """Report whether the segment stays inside both ceilings.

        Args:
            max_speed: Speed ceiling, in meters per second.
            max_acceleration: Acceleration ceiling, in meters per second
                squared.

        Returns:
            Whether both hold everywhere on the segment.
        """
        peak_speed, peak_acceleration = self.peaks(sample_count)
        return peak_speed <= max_speed and peak_acceleration <= max_acceleration


def descent_seconds(z_offset: float, approach_speed: float) -> float:
    """Return how long the descent onto an object takes.

    Args:
        z_offset: How far above the object the approach stands, in meters.
        approach_speed: How fast the flange is coming down when it gets
            there, in meters per second.

    Returns:
        The duration, `2 * z_offset / approach_speed`. The exact condition
        for never passing below the object is
        `dt <= 5 * z_offset / (2 * approach_speed)`, read off the leading
        cubic term of the quintic written about its end, and this sits at
        four fifths of it rather than on it.

    Raises:
        ValueError: If either argument is not positive, because neither a
            descent of no height nor one at no speed has a duration.
    """
    if z_offset <= 0.0 or approach_speed <= 0.0:
        raise ValueError(
            f"a descent of {z_offset} m at {approach_speed} m/s has no duration"
        )
    return 2.0 * z_offset / approach_speed


def descent_limit_seconds(z_offset: float, approach_speed: float) -> float:
    """Return the longest descent that never passes below the object.

    Args:
        z_offset: How far above the object the approach stands, in meters.
        approach_speed: How fast the flange is coming down, in meters per
            second.

    Returns:
        `5 * z_offset / (2 * approach_speed)`, which is where the leading
        cubic term of the quintic written about its end changes sign.
    """
    return 2.5 * z_offset / approach_speed


def approach(
    flange: State,
    object_position_belt: NDArray[np.float64] | None = None,
    object_velocity_world: NDArray[np.float64] | None = None,
    approach_clearance_z: float | None = None,
    approach_speed: float = 0.0,
    max_speed: float = 0.0,
    max_acceleration: float = 0.0,
    latest: float = 0.0,
    margin: float = 1.0,
    *,
    drift_horizon: float,
    segment_sample_count: int,
    bisection_passes: int,
    object_position: NDArray[np.float64] | None = None,
    object_velocity: NDArray[np.float64] | None = None,
    z_offset: float | None = None,
) -> Segment | None:
    """Return the soonest feasible arc onto the point above a moving object.

    The arc ends above where the object will be when the descent that
    follows finishes, moving with the object and already coming down at the
    approach speed, so the two arcs chain without the flange stopping
    between them.

    Args:
        flange: Where the flange is now and how it is moving.
        object_position_belt: Where the object is now, in belt frame meters.
        object_velocity_world: How the belt is carrying it, in meters per second.
        approach_clearance_z: Clearance above the object to approach at, in meters.
        approach_speed: How fast to be descending on arrival, in meters per
            second.
        max_speed: Speed ceiling, in meters per second.
        max_acceleration: Acceleration ceiling, in meters per second squared.
        latest: The longest interception worth considering, in seconds,
            which is normally what the object has left before it leaves the
            window.
        margin: How much longer than the soonest feasible interception to
            take, as a multiple. One takes the soonest, which sits exactly
            on whichever ceiling binds and therefore leaves no room to
            re-aim the arc later: every correction breaks the bound it was
            already touching. Anything above one buys that room at the cost
            of a later pick.
        object_position: Deprecated alias for object_position_belt.
        object_velocity: Deprecated alias for object_velocity_world.
        z_offset: Deprecated alias for approach_clearance_z.

    Returns:
        The arc, or None when no interception inside `latest` respects both
        ceilings. None is the honest answer for an object the arm cannot
        reach in the belt it has left, and the caller drops it rather than
        chasing it.
    """
    obj_pos = object_position if object_position is not None else object_position_belt
    if obj_pos is None:
        raise TypeError("approach requires object_position_belt or object_position")
    obj_vel = object_velocity if object_velocity is not None else object_velocity_world
    if obj_vel is None:
        raise TypeError("approach requires object_velocity_world or object_velocity")
    clearance = z_offset if z_offset is not None else approach_clearance_z
    if clearance is None:
        raise TypeError("approach requires approach_clearance_z or z_offset")

    dt = descent_seconds(clearance, approach_speed)
    # Aim positions carry lateral drift for a horizon; matching velocity is
    # belt-axis only (AC-MOVE-71). The world image of object-frame rest at the
    # pick is transport, not contact-disturbed cvel.
    match = np.asarray([max(0.0, float(obj_vel[0])), 0.0, 0.0], dtype=np.float64)

    def arc(seconds: float) -> Segment:
        end_velocity = match.copy()
        end_velocity[2] -= approach_speed
        return Segment(
            start=flange,
            end=State(
                position=where_carried(
                    obj_pos,
                    obj_vel,
                    seconds + dt,
                    clearance,
                    drift_horizon,
                ),
                velocity=end_velocity,
                acceleration=zeros(),
            ),
            duration=seconds,
        )

    # Peak speed and peak acceleration both fall as the duration grows, so the
    # predicate is monotone and the soonest feasible interception is the
    # smallest feasible duration in the bracket.
    if not arc(latest).fits(max_speed, max_acceleration, segment_sample_count):
        return None
    seconds = bisect_feasible(
        lambda middle: arc(middle).fits(
            max_speed, max_acceleration, segment_sample_count
        ),
        0.0,
        latest,
        bisection_passes,
    )
    return arc(min(seconds * margin, latest))


def descend(
    approach_arc: Segment,
    object_position_belt: NDArray[np.float64] | None = None,
    object_velocity_world: NDArray[np.float64] | None = None,
    approach_clearance_z: float | None = None,
    approach_speed: float = 0.0,
    *,
    drift_horizon: float,
    object_position: NDArray[np.float64] | None = None,
    object_velocity: NDArray[np.float64] | None = None,
    z_offset: float | None = None,
) -> Segment:
    """Return the arc from the approach point down onto the object.

    Args:
        approach_arc: The arc that ended above the object, whose end is this
            one's start, so the flange never stops between them.
        object_position_belt: Where the object was when the approach was planned.
        object_velocity_world: How the belt is carrying it.
        approach_clearance_z: The clearance the approach stood at, in meters.
        approach_speed: How fast the flange is coming down, in meters per
            second.
        drift_horizon: How long the object's drift across the belt is carried
            for, in seconds. See [where_carried].
        object_position: Deprecated alias for object_position_belt.
        object_velocity: Deprecated alias for object_velocity_world.
        z_offset: Deprecated alias for approach_clearance_z.

    Returns:
        The arc. It ends moving with the object rather than at rest, because
        a jaw arriving stopped has the object sliding through it at belt
        speed.
    """
    obj_pos = object_position if object_position is not None else object_position_belt
    if obj_pos is None:
        raise TypeError("descend requires object_position_belt or object_position")
    obj_vel = object_velocity if object_velocity is not None else object_velocity_world
    if obj_vel is None:
        raise TypeError("descend requires object_velocity_world or object_velocity")
    clearance = z_offset if z_offset is not None else approach_clearance_z
    if clearance is None:
        raise TypeError("descend requires approach_clearance_z or z_offset")

    dt = descent_seconds(clearance, approach_speed)
    arrival = approach_arc.duration + dt
    match = np.asarray([max(0.0, float(obj_vel[0])), 0.0, 0.0], dtype=np.float64)
    return Segment(
        start=approach_arc.end,
        end=State(
            position=where_carried(obj_pos, obj_vel, arrival, 0.0, drift_horizon),
            velocity=match,
            acceleration=zeros(),
        ),
        duration=dt,
    )


def where(
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    seconds: float,
    lift: float,
) -> NDArray[np.float64]:
    """Return where a point carried at a constant velocity will be.

    Public because the sequence in [clave.control.pick] predicts against the
    same model the arcs here are solved against, and two copies of a
    prediction is how the arc and the pose it aims at come to disagree.

    Args:
        position: Where it is now.
        velocity: How it is moving.
        seconds: How far ahead.
        lift: How far above the result to sit, in meters.

    Returns:
        The predicted position.
    """
    return np.asarray(
        position + velocity * seconds + np.asarray([0.0, 0.0, lift]),
        dtype=np.float64,
    )


def where_carried(
    position: NDArray[np.float64],
    velocity: NDArray[np.float64],
    seconds: float,
    lift: float,
    drift_horizon: float,
) -> NDArray[np.float64]:
    """Return where an object will be, carrying its drift across the belt for a while.

    The two horizontal axes are not the same process. Along the belt the object
    is driven and its velocity holds for as long as the belt does. Across the
    belt nothing drives it: the drift is a transient, so it is carried only for
    `drift_horizon`. Carrying a lateral velocity for the whole visit invents
    motion that is no longer there.

    Args:
        position: Where the object is now.
        velocity: How it is moving.
        seconds: How far ahead.
        lift: How far above the result to sit, in meters.
        drift_horizon: How long to keep carrying the component across the belt,
            in seconds. The component along the belt is carried for the whole
            `seconds`.

    Returns:
        The predicted position.
    """
    # The rule is a duration per axis: the belt's own axis is carried for the
    # whole interval, the axis across it only for as long as the drift is
    # remembered, and the vertical axis is not carried at all.
    carried = np.asarray([seconds, min(seconds, drift_horizon), 0.0])
    return np.asarray(
        position + velocity * carried + np.asarray([0.0, 0.0, lift]),
        dtype=np.float64,
    )


def _dot(weights: tuple[float, ...], terms: tuple[float, ...]) -> float:
    """Return the dot product of a basis row and one axis of coefficients.

    Args:
        weights: The six basis weights at one instant.
        terms: The six coefficients of one axis.

    Returns:
        The polynomial, or its derivative, on that axis.
    """
    return sum(weight * term for weight, term in zip(weights, terms, strict=True))


def _axis_terms(segment: Segment, axis: int, span: float) -> tuple[float, ...]:
    """Return one axis of a segment's Hermite coefficients.

    Args:
        segment: The arc.
        axis: Which component, 0, 1 or 2.
        span: The duration the velocity and acceleration terms are scaled by.

    Returns:
        The six coefficients, in the same order as `_basis_at`.
    """
    span2 = span * span
    return (
        segment.start.position[axis],
        segment.start.velocity[axis] * span,
        segment.start.acceleration[axis] * span2,
        segment.end.acceleration[axis] * span2,
        segment.end.velocity[axis] * span,
        segment.end.position[axis],
    )


def _basis_at(s: float) -> tuple[float, ...]:
    """Return the quintic Hermite basis at one normalized time.

    The same polynomial as `_basis`, written for one instant. A physics tick
    asks for one row, and building the `(n, 6)` stack `_basis` returns costs
    more than the arithmetic.

    Args:
        s: Normalized time, from zero to one.

    Returns:
        The six weights, ordered to match the coefficient vector.
    """
    s2, s3, s4, s5 = s * s, s**3, s**4, s**5
    return (
        1 - 10 * s3 + 15 * s4 - 6 * s5,
        s - 6 * s3 + 8 * s4 - 3 * s5,
        0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5,
        0.5 * s3 - s4 + 0.5 * s5,
        -4 * s3 + 7 * s4 - 3 * s5,
        10 * s3 - 15 * s4 + 6 * s5,
    )


def _first_at(s: float) -> tuple[float, ...]:
    """Return the basis differentiated once, at one normalized time.

    Args:
        s: Normalized time, from zero to one.

    Returns:
        The six weights.
    """
    s2, s3, s4 = s * s, s**3, s**4
    return (
        -30 * s2 + 60 * s3 - 30 * s4,
        1 - 18 * s2 + 32 * s3 - 15 * s4,
        s - 4.5 * s2 + 6 * s3 - 2.5 * s4,
        1.5 * s2 - 4 * s3 + 2.5 * s4,
        -12 * s2 + 28 * s3 - 15 * s4,
        30 * s2 - 60 * s3 + 30 * s4,
    )


def _second_at(s: float) -> tuple[float, ...]:
    """Return the basis differentiated twice, at one normalized time.

    Args:
        s: Normalized time, from zero to one.

    Returns:
        The six weights.
    """
    s2, s3 = s * s, s**3
    return (
        -60 * s + 180 * s2 - 120 * s3,
        -36 * s + 96 * s2 - 60 * s3,
        1 - 9 * s + 18 * s2 - 10 * s3,
        3 * s - 12 * s2 + 10 * s3,
        -24 * s + 84 * s2 - 60 * s3,
        60 * s - 180 * s2 + 120 * s3,
    )


_UNIT_GRIDS: dict[int, NDArray[np.float64]] = {}
"""Cached normalized sample grids keyed by `sample_count` (`AC-PERF-04`)."""


def _unit_grid(sample_count: int) -> NDArray[np.float64]:
    """Return the open unit grid used by peak searches, cached per count.

    Args:
        sample_count: Number of intervals along the segment (samples is one
            more, matching the previous `linspace(0, duration, count + 1)`).

    Returns:
        Normalized times from zero to one, shape `(sample_count + 1,)`.
    """
    cached = _UNIT_GRIDS.get(sample_count)
    if cached is None:
        cached = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)
        _UNIT_GRIDS[sample_count] = cached
    return cached


def _coefficient_matrix(segment: Segment, span: float) -> NDArray[np.float64]:
    """Return the `(6, 3)` Hermite coefficient matrix for one segment.

    Args:
        segment: The arc.
        span: Duration used to scale velocity and acceleration rows.

    Returns:
        Rows ordered as start position, start velocity, start acceleration,
        end acceleration, end velocity, end position.
    """
    span2 = span * span
    terms = np.empty((6, 3), dtype=np.float64)
    terms[0] = segment.start.position
    terms[1] = segment.start.velocity * span
    terms[2] = segment.start.acceleration * span2
    terms[3] = segment.end.acceleration * span2
    terms[4] = segment.end.velocity * span
    terms[5] = segment.end.position
    return terms


def _evaluate(
    segment: Segment, span: float, s: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate position, velocity and acceleration on a normalized grid.

    Args:
        segment: The arc.
        span: Its duration, already floored away from zero.
        s: Normalized times in `[0, 1]`.

    Returns:
        Position, velocity and acceleration, each shape `(len(s), 3)`.
    """
    terms = _coefficient_matrix(segment, span)
    return (
        _basis(s) @ terms,
        (_first(s) @ terms) / span,
        (_second(s) @ terms) / (span * span),
    )


def _fill_basis(
    out: NDArray[np.float64], columns: list[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Write six basis columns into a `(n, 6)` buffer and return it.

    Args:
        out: Destination, shape `(n, 6)`.
        columns: The six weight arrays, one per Hermite coefficient.

    Returns:
        `out`.
    """
    for index, column in enumerate(columns):
        out[:, index] = column
    return out


def _basis(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the quintic Hermite basis at each normalized time.

    Args:
        s: Normalized time, from zero to one, as an array.

    Returns:
        The six weights at each time, shape `(len(s), 6)`, ordered to match
        the coefficient vector: start position, start velocity, start
        acceleration, end acceleration, end velocity, end position.
    """
    s2, s3, s4, s5 = s * s, s**3, s**4, s**5
    out = np.empty((s.shape[0], 6), dtype=np.float64)
    return _fill_basis(
        out,
        [
            1 - 10 * s3 + 15 * s4 - 6 * s5,
            s - 6 * s3 + 8 * s4 - 3 * s5,
            0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5,
            0.5 * s3 - s4 + 0.5 * s5,
            -4 * s3 + 7 * s4 - 3 * s5,
            10 * s3 - 15 * s4 + 6 * s5,
        ],
    )


def _first(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the basis differentiated once with respect to normalized time.

    Args:
        s: Normalized time, as an array.

    Returns:
        The six weights at each time, shape `(len(s), 6)`.
    """
    s2, s3, s4 = s * s, s**3, s**4
    out = np.empty((s.shape[0], 6), dtype=np.float64)
    return _fill_basis(
        out,
        [
            -30 * s2 + 60 * s3 - 30 * s4,
            1 - 18 * s2 + 32 * s3 - 15 * s4,
            s - 4.5 * s2 + 6 * s3 - 2.5 * s4,
            1.5 * s2 - 4 * s3 + 2.5 * s4,
            -12 * s2 + 28 * s3 - 15 * s4,
            30 * s2 - 60 * s3 + 30 * s4,
        ],
    )


def _second(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the basis differentiated twice with respect to normalized time.

    Args:
        s: Normalized time, as an array.

    Returns:
        The six weights at each time, shape `(len(s), 6)`.
    """
    s2, s3 = s * s, s**3
    out = np.empty((s.shape[0], 6), dtype=np.float64)
    return _fill_basis(
        out,
        [
            -60 * s + 180 * s2 - 120 * s3,
            -36 * s + 96 * s2 - 60 * s3,
            1 - 9 * s + 18 * s2 - 10 * s3,
            3 * s - 12 * s2 + 10 * s3,
            -24 * s + 84 * s2 - 60 * s3,
            60 * s - 180 * s2 + 120 * s3,
        ],
    )
