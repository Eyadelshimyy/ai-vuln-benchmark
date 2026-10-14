"""Trajectories with a known duration, and the interception they let us solve."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.control.trajectory import (
    Segment,
    State,
    approach,
    descend,
    descent_limit_seconds,
    descent_seconds,
    soonest_feasible,
)

BELT = np.asarray((0.314, 0.0, 0.0), dtype=np.float64)
MAX_SPEED = 1.00
MAX_ACCELERATION = 2.50
Z_OFFSET = 0.05
DRIFT_HORIZON = 0.30
SEGMENT_SAMPLES = 64
BISECTION_PASSES = 40
APPROACH_SPEED = 0.25
PARK = np.asarray((0.45, -1.00, 1.20), dtype=np.float64)


def moving(position: NDArray[np.float64]) -> State:
    """A state standing still somewhere."""
    return State.at_rest(np.asarray(position, dtype=np.float64))


def test_a_segment_meets_every_boundary_condition_it_was_given() -> None:
    """A segment meets every boundary condition it was given.

    Six conditions per axis and six coefficients, so the polynomial is
    determined rather than fitted. This is the property the whole
    formulation rests on: if the acceleration at a waypoint were only
    approached, chaining two arcs would jerk.
    """
    start = State(
        position=np.asarray((0.1, 0.2, 1.0), dtype=np.float64),
        velocity=np.asarray((0.3, -0.1, 0.0), dtype=np.float64),
        acceleration=np.asarray((0.5, 0.0, -0.2), dtype=np.float64),
    )
    end = State(
        position=np.asarray((0.6, 0.0, 0.9), dtype=np.float64),
        velocity=np.asarray((0.0, 0.2, -0.25), dtype=np.float64),
        acceleration=np.zeros(3, dtype=np.float64),
    )
    arc = Segment(start=start, end=end, duration=0.8)

    begins, finishes = arc.at(0.0), arc.at(0.8)
    for axis in range(3):
        assert begins.position[axis] == pytest.approx(start.position[axis])
        assert begins.velocity[axis] == pytest.approx(start.velocity[axis])
        assert begins.acceleration[axis] == pytest.approx(start.acceleration[axis])
        assert finishes.position[axis] == pytest.approx(end.position[axis])
        assert finishes.velocity[axis] == pytest.approx(end.velocity[axis])
        assert finishes.acceleration[axis] == pytest.approx(end.acceleration[axis])


def test_peaks_match_sampled_polynomial_on_cached_grid() -> None:
    """AC-PERF-04: peaks on the cached unit grid match sampling the duration."""
    arc = Segment(
        start=State(
            position=np.asarray((0.1, 0.2, 1.0), dtype=np.float64),
            velocity=np.asarray((0.3, -0.1, 0.0), dtype=np.float64),
            acceleration=np.asarray((0.5, 0.0, -0.2), dtype=np.float64),
        ),
        end=State(
            position=np.asarray((0.6, 0.0, 0.9), dtype=np.float64),
            velocity=np.asarray((0.0, 0.2, -0.25), dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
        ),
        duration=0.8,
    )
    _, velocity, acceleration = arc.sample(
        np.linspace(0.0, arc.duration, SEGMENT_SAMPLES + 1, dtype=np.float64)
    )
    expected = (
        float(np.linalg.norm(velocity, axis=1).max()),
        float(np.linalg.norm(acceleration, axis=1).max()),
    )
    assert arc.peaks(SEGMENT_SAMPLES) == pytest.approx(expected, rel=0, abs=1e-12)


def test_one_instant_matches_the_sampled_polynomial() -> None:
    """One instant matches the sampled polynomial.

    The physics tick evaluates the closed form directly, and a peak search
    evaluates the same polynomial on a grid. The two are one polynomial, so
    a point read either way is the same point.
    """
    arc = Segment(
        start=State(
            position=np.asarray((0.1, 0.2, 1.0), dtype=np.float64),
            velocity=np.asarray((0.3, -0.1, 0.0), dtype=np.float64),
            acceleration=np.asarray((0.5, 0.0, -0.2), dtype=np.float64),
        ),
        end=State(
            position=np.asarray((0.6, 0.0, 0.9), dtype=np.float64),
            velocity=np.asarray((0.0, 0.2, -0.25), dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
        ),
        duration=0.8,
    )
    for fraction in (0.0, 0.17, 0.5, 0.83, 1.0):
        elapsed = arc.duration * fraction
        direct = arc.at(elapsed)
        position, velocity, acceleration = arc.sample(np.asarray([elapsed]))
        assert direct.position == pytest.approx(tuple(position[0]), abs=1e-12)
        assert direct.velocity == pytest.approx(tuple(velocity[0]), abs=1e-12)
        assert direct.acceleration == pytest.approx(tuple(acceleration[0]), abs=1e-12)


def test_asking_past_the_end_returns_the_end() -> None:
    """Asking past the end returns the end.

    A quintic extrapolated diverges fast, and a controller that reads one
    tick late should hold rather than be thrown across the cell.
    """
    arc = Segment(
        start=State.at_rest(np.asarray((0.0, 0.0, 1.0), dtype=np.float64)),
        end=State.at_rest(np.asarray((0.5, 0.0, 1.0), dtype=np.float64)),
        duration=0.5,
    )
    assert arc.at(9.0).position == pytest.approx(arc.at(0.5).position)


def test_the_dip_starts_exactly_where_the_algebra_says() -> None:
    """The dip starts exactly where the algebra says.

    Writing the quintic about its end in `u = 1 - s`, the leading term is
    cubic because position, velocity and acceleration all vanish there, and
    its coefficient is `-2 * (2 * V * dt - 5 * Z)`. So the descent passes
    below the object exactly when `dt > 5 * Z / (2 * V)`. This brackets that
    boundary from both sides rather than checking one safe value, which is
    what tells the two apart.
    """
    clearance, speed = 0.05, 0.25
    boundary = descent_limit_seconds(clearance, speed)
    assert boundary == pytest.approx(0.50)

    top = np.asarray((0.0, 0.0, 1.00), dtype=np.float64)
    for span, dips in ((boundary * 0.98, False), (boundary * 1.10, True)):
        arc = Segment(
            start=State(
                position=top,
                velocity=np.asarray((BELT[0], 0.0, -speed), dtype=np.float64),
                acceleration=np.zeros(3, dtype=np.float64),
            ),
            end=State(
                position=np.asarray(
                    (BELT[0] * span, 0.0, top[2] - clearance),
                    dtype=np.float64,
                ),
                velocity=BELT,
                acceleration=np.zeros(3, dtype=np.float64),
            ),
            duration=span,
        )
        lowest = min(float(arc.at(span * i / 400).position[2]) for i in range(401))
        under = float(top[2] - clearance) - lowest
        assert (under > 1e-6) is dips, f"{span:.3f} s dipped {under * 1000:.3f} mm"


def test_the_descent_used_sits_inside_that_boundary_with_margin() -> None:
    """The descent used sits inside that boundary with margin.

    Four fifths of the limit rather than on it. Acceleration is not the
    binding constraint, so stretching toward the boundary buys nothing, and
    a shorter descent keeps the prediction horizon short.
    """
    assert descent_seconds(0.05, 0.25) == pytest.approx(0.40)
    assert descent_seconds(0.05, 0.25) == pytest.approx(
        0.8 * descent_limit_seconds(0.05, 0.25)
    )


def test_the_chosen_descent_never_dips() -> None:
    """The chosen descent never dips."""
    clearance, speed = 0.05, 0.25
    exact = descent_seconds(clearance, speed)
    top = np.asarray((0.0, 0.0, 1.00), dtype=np.float64)
    for stretch, dips in ((1.0, False),):
        span = exact * stretch
        arc = Segment(
            start=State(
                position=top,
                velocity=np.asarray((BELT[0], 0.0, -speed), dtype=np.float64),
                acceleration=np.zeros(3, dtype=np.float64),
            ),
            end=State(
                position=np.asarray(
                    (BELT[0] * span, 0.0, top[2] - clearance),
                    dtype=np.float64,
                ),
                velocity=BELT,
                acceleration=np.zeros(3, dtype=np.float64),
            ),
            duration=span,
        )
        lowest = min(float(arc.at(span * i / 200).position[2]) for i in range(201))
        under = float(top[2] - clearance) - lowest
        assert (under > 1e-5) is dips, f"stretch {stretch} dipped {under * 1000:.2f} mm"


def test_a_descent_that_matches_the_object_costs_far_less_acceleration() -> None:
    """A descent that matches the object costs far less acceleration.

    Ending at rest means the flange stops while the object keeps moving, so
    the arc has to cover the belt travel horizontally inside the descent.
    That horizontal catch-up, and not the vertical braking, is what
    dominates the acceleration.
    """
    span = descent_seconds(Z_OFFSET, APPROACH_SPEED)
    top = State(
        position=np.asarray((0.0, 0.0, 1.00), dtype=np.float64),
        velocity=np.asarray((BELT[0], 0.0, -APPROACH_SPEED), dtype=np.float64),
        acceleration=np.zeros(3, dtype=np.float64),
    )
    landing = np.asarray((BELT[0] * span, 0.0, 0.95), dtype=np.float64)

    stopping = Segment(top, State.at_rest(landing), span).peak_acceleration(
        SEGMENT_SAMPLES
    )
    matching = Segment(
        top,
        State(
            position=landing,
            velocity=BELT,
            acceleration=np.zeros(3, dtype=np.float64),
        ),
        span,
    ).peak_acceleration(SEGMENT_SAMPLES)
    assert matching < 0.35 * stopping


def test_the_approach_ends_above_where_the_object_will_be() -> None:
    """AC-MOVE-10: the approach ends above where the object will be.

    Checked against the definition rather than a constant: the arc's end
    sits one clearance above the object's position at the instant the
    descent that follows will finish.
    """
    now = np.asarray((-0.40, 0.10, 0.95), dtype=np.float64)
    arc = approach(
        moving(PARK),
        now,
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        MAX_SPEED,
        MAX_ACCELERATION,
        latest=6.0,
        drift_horizon=DRIFT_HORIZON,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
    )
    assert arc is not None
    arrival = arc.duration + descent_seconds(Z_OFFSET, APPROACH_SPEED)
    assert arc.end.position[0] == pytest.approx(now[0] + BELT[0] * arrival)
    assert arc.end.position[2] == pytest.approx(now[2] + Z_OFFSET)


def test_the_approach_arrives_already_descending_and_moving_with_the_belt() -> None:
    """The approach arrives already descending and moving with the belt.

    So the two arcs chain rather than the flange stopping between them,
    which is the whole reason the terminal velocity is a boundary condition
    and not an outcome.
    """
    arc = approach(
        moving(PARK),
        np.asarray((-0.40, 0.10, 0.95), dtype=np.float64),
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        MAX_SPEED,
        MAX_ACCELERATION,
        latest=6.0,
        drift_horizon=DRIFT_HORIZON,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
    )
    assert arc is not None
    assert arc.end.velocity[0] == pytest.approx(BELT[0])
    assert arc.end.velocity[2] == pytest.approx(-APPROACH_SPEED)


def test_the_approach_respects_both_ceilings_over_its_whole_length() -> None:
    """AC-MOVE-09: the approach respects both ceilings over its whole length."""
    arc = approach(
        moving(PARK),
        np.asarray((-0.40, 0.10, 0.95), dtype=np.float64),
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        MAX_SPEED,
        MAX_ACCELERATION,
        latest=6.0,
        drift_horizon=DRIFT_HORIZON,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
    )
    assert arc is not None
    assert arc.peak_speed(SEGMENT_SAMPLES) <= MAX_SPEED + 1e-9
    assert arc.peak_acceleration(SEGMENT_SAMPLES) <= MAX_ACCELERATION + 1e-9


def test_the_interception_found_is_the_soonest_feasible_one() -> None:
    """The interception found is the soonest feasible one.

    A duration the pick can count on rather than the shortest that exists,
    and the bisection is only correct because the predicate is monotone.
    This pins that: a shade sooner does not fit.
    """
    arc = approach(
        moving(PARK),
        np.asarray((-0.40, 0.10, 0.95), dtype=np.float64),
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        MAX_SPEED,
        MAX_ACCELERATION,
        latest=6.0,
        drift_horizon=DRIFT_HORIZON,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
    )
    assert arc is not None
    hurried = Segment(arc.start, arc.end, arc.duration * 0.85)
    assert not hurried.fits(MAX_SPEED, MAX_ACCELERATION, SEGMENT_SAMPLES)


def test_a_search_from_only_a_lower_bound_finds_the_soonest_fit() -> None:
    """`soonest_feasible` supplies the passing end a caller does not have.

    `bisect_feasible` needs both ends of a bracket, and a search that starts
    from a lower bound has only the failing end. This doubles upward to find
    a passing end, then bisects: a shade sooner than the answer still fails.
    """

    def feasible(seconds: float) -> bool:
        return seconds >= 0.6

    found = soonest_feasible(feasible, 0.4, BISECTION_PASSES)
    assert found == pytest.approx(0.6)
    assert not feasible(found * 0.85)


def test_a_lower_bound_that_already_fits_is_kept() -> None:
    """The search does not move a duration that already fits."""
    assert soonest_feasible(
        lambda seconds: seconds >= 0.2, 0.5, BISECTION_PASSES
    ) == pytest.approx(0.5)


def test_a_feasibility_search_needs_a_positive_seed() -> None:
    """The bracket grows by doubling, so a zero or negative seed cannot grow."""
    with pytest.raises(ValueError):
        soonest_feasible(lambda seconds: True, 0.0, BISECTION_PASSES)


def test_an_object_that_cannot_be_reached_in_time_is_refused() -> None:
    """An object that cannot be reached in time is refused.

    None rather than a plan the arm cannot execute, so the caller drops the
    object instead of chasing it off the end of the belt.
    """
    assert (
        approach(
            moving(PARK),
            np.asarray((-0.40, 0.10, 0.95), dtype=np.float64),
            BELT,
            Z_OFFSET,
            APPROACH_SPEED,
            MAX_SPEED,
            MAX_ACCELERATION,
            latest=0.2,
            drift_horizon=DRIFT_HORIZON,
            segment_sample_count=SEGMENT_SAMPLES,
            bisection_passes=BISECTION_PASSES,
        )
        is None
    )


def test_the_descent_starts_exactly_where_the_approach_ended() -> None:
    """The descent starts exactly where the approach ended.

    Position, velocity and acceleration all carried across, which is what
    makes the pair one motion rather than two with a stop between them.
    """
    now = np.asarray((-0.40, 0.10, 0.95), dtype=np.float64)
    first = approach(
        moving(PARK),
        now,
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        MAX_SPEED,
        MAX_ACCELERATION,
        latest=6.0,
        drift_horizon=DRIFT_HORIZON,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
    )
    assert first is not None
    second = descend(
        first,
        now,
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        drift_horizon=DRIFT_HORIZON,
    )
    assert second.start == first.end


def test_the_descent_lands_on_the_object_moving_with_it() -> None:
    """AC-MOVE-10: the descent lands on the object, moving with it."""
    now = np.asarray((-0.40, 0.10, 0.95), dtype=np.float64)
    first = approach(
        moving(PARK),
        now,
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        MAX_SPEED,
        MAX_ACCELERATION,
        latest=6.0,
        drift_horizon=DRIFT_HORIZON,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
    )
    assert first is not None
    second = descend(
        first,
        now,
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        drift_horizon=DRIFT_HORIZON,
    )
    arrival = first.duration + second.duration
    assert second.end.position[0] == pytest.approx(now[0] + BELT[0] * arrival)
    assert second.end.position[2] == pytest.approx(now[2])
    assert second.end.velocity == pytest.approx(BELT)


def test_the_pair_never_stops_between_the_two_arcs() -> None:
    """The pair never stops between the two arcs.

    Swept across the join rather than checked at it, because a profile that
    happens to be moving at the waypoint and stalls either side of it is
    still an arm that goes, stops and goes.
    """
    now = np.asarray((-0.40, 0.10, 0.95), dtype=np.float64)
    first = approach(
        moving(PARK),
        now,
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        MAX_SPEED,
        MAX_ACCELERATION,
        latest=6.0,
        drift_horizon=DRIFT_HORIZON,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
    )
    assert first is not None
    second = descend(
        first,
        now,
        BELT,
        Z_OFFSET,
        APPROACH_SPEED,
        drift_horizon=DRIFT_HORIZON,
    )
    for arc, fraction in ((first, 0.9), (first, 1.0), (second, 0.0), (second, 0.1)):
        speed = math.dist((0, 0, 0), arc.at(arc.duration * fraction).velocity)
        assert speed > 0.1, f"the flange all but stopped at the join: {speed:.3f} m/s"


def test_a_belt_the_arm_cannot_outrun_refuses_every_interception() -> None:
    """A belt the arm cannot outrun refuses every interception.

    Feasibility is not that the arm beats the belt. A quintic from rest
    peaks at about 1.875 times its average, so the ceiling has to clear
    that multiple of the belt speed before any interception exists. This
    fails if that margin is ever quietly assumed away.
    """
    quick = np.asarray((0.9, 0.0, 0.0), dtype=np.float64)
    assert (
        approach(
            moving(PARK),
            np.asarray((-0.40, 0.10, 0.95), dtype=np.float64),
            quick,
            Z_OFFSET,
            APPROACH_SPEED,
            MAX_SPEED,
            MAX_ACCELERATION,
            latest=8.0,
            drift_horizon=DRIFT_HORIZON,
            segment_sample_count=SEGMENT_SAMPLES,
            bisection_passes=BISECTION_PASSES,
        )
        is None
    )


def test_a_descent_of_no_height_has_no_duration() -> None:
    """A descent of no height has no duration."""
    for clearance, speed in ((0.0, 0.25), (0.05, 0.0)):
        with pytest.raises(ValueError, match="no duration"):
            descent_seconds(clearance, speed)
