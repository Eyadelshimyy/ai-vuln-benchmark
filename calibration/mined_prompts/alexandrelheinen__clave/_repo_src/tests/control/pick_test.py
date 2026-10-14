"""One visit, planned end to end before the arm moves.

The claims worth pinning are about the seams between the arcs rather than
about any one arc, because the arcs themselves are tested in
`trajectory_test.py`. A sequence whose legs do not chain produces a step in
velocity at every waypoint, and a step in velocity is a lurch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.control.pick import (
    JAW_OPEN,
    JAW_SHUT,
    Plan,
    nearer_parallel_yaw,
    plan_pick,
    refine,
    retarget_descent,
)
from clave.control.settings import Phase
from clave.control.trajectory import State

BELT = np.asarray((0.314, 0.0, 0.0), dtype=np.float64)
CLEARANCE = 0.050
APPROACH_SPEED = 0.25
DWELL = 0.30
PARK = np.asarray((0.45, -1.00, 1.20), dtype=np.float64)
OBJECT = np.asarray((0.30, 0.0, 1.035), dtype=np.float64)
DRIFT_HORIZON = 0.30
MINIMUM_SEGMENT = 0.50
CLOSING_RISE = 0.20
SEGMENT_SAMPLES = 64
BISECTION_PASSES = 40
MINIMUM_DELIVERY = 0.05
CORRECTION_STEPS = 32


def a_plan(**overrides: object) -> Plan | None:
    """A plan from the park pose onto an object the belt is carrying."""
    fields: dict[str, object] = {
        "flange": State.at_rest(PARK),
        "track_id": 1,
        "object_position": OBJECT,
        "belt_velocity": BELT,
        "z_offset": CLEARANCE,
        "approach_speed": APPROACH_SPEED,
        "dwell_seconds": DWELL,
        "max_speed": 1.00,
        "max_acceleration": 2.50,
        "latest": 4.00,
        "at_seconds": 0.0,
        "margin": 1.0,
        "drift_horizon": DRIFT_HORIZON,
        "minimum_segment_seconds": MINIMUM_SEGMENT,
        "closing_rise_seconds": CLOSING_RISE,
        "segment_sample_count": SEGMENT_SAMPLES,
        "bisection_passes": BISECTION_PASSES,
        "minimum_delivery_seconds": MINIMUM_DELIVERY,
        "correction_steps": CORRECTION_STEPS,
    }
    fields.update(overrides)
    return plan_pick(**fields)  # type: ignore[arg-type]


def test_a_plan_runs_track_descend_hold_retreat() -> None:
    """AC-MOVE-38: a plan runs track, descend, hold and retreat."""
    plan = a_plan()
    assert plan is not None
    assert [leg.phase for leg in plan.legs] == [
        Phase.TRACK,
        Phase.DESCEND,
        Phase.HOLD,
        Phase.RETREAT,
    ]


def test_every_leg_begins_where_the_last_one_ended() -> None:
    """AC-MOVE-39: every leg begins where the last one ended.

    Position, velocity and acceleration all three, because a sequence that
    only matches position produces a step in velocity at every waypoint.
    """
    plan = a_plan()
    assert plan is not None
    for before, after in zip(plan.legs, plan.legs[1:], strict=False):
        assert after.segment.start.position == pytest.approx(
            before.segment.end.position
        )
        assert after.segment.start.velocity == pytest.approx(
            before.segment.end.velocity
        )
        assert after.segment.start.acceleration == pytest.approx(
            before.segment.end.acceleration
        )


def test_the_hold_is_a_carry_at_exactly_belt_speed() -> None:
    """AC-MOVE-39: the hold is a carry at exactly belt speed.

    A quintic whose endpoints differ by the velocity times the duration, with
    that velocity at both ends and no acceleration, collapses to a straight
    line at constant speed. The jaw therefore closes with no relative motion
    at all, which is the precondition a grasp needs.
    """
    plan = a_plan()
    assert plan is not None
    hold = plan.legs[2].segment
    for step in range(11):
        state = hold.at(hold.duration * step / 10.0)
        assert state.velocity == pytest.approx(BELT, abs=1e-12)
        assert state.acceleration == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_the_jaw_stays_open_until_the_hold() -> None:
    """AC-MOVE-38: the jaw stays open until the hold.

    A jaw closing on the way down sweeps the object off the belt, which is
    the failure this whole sequence exists to remove.
    """
    plan = a_plan()
    assert plan is not None
    grips = {leg.phase: leg.grip for leg in plan.legs}
    assert grips[Phase.TRACK] == JAW_OPEN
    assert grips[Phase.DESCEND] == JAW_OPEN
    assert grips[Phase.HOLD] == JAW_SHUT
    assert grips[Phase.RETREAT] == JAW_SHUT


def test_the_pick_instant_is_when_the_hold_begins() -> None:
    """AC-MOVE-40: the pick instant is when the hold begins."""
    plan = a_plan()
    assert plan is not None
    assert plan.at(plan.pick_at) is not None
    phase, _, grip = plan.at(plan.pick_at)  # type: ignore[misc]
    assert phase is Phase.HOLD
    assert grip == JAW_SHUT


def test_the_retreat_is_the_descent_reversed() -> None:
    """AC-MOVE-48: the retreat is the descent reversed.

    Same clearance, and therefore the same duration, entering at the object's
    own velocity and leaving at that velocity plus the approach speed along the
    belt normal. A retreat that ends at rest is not a neutral choice: the
    object's frame carries the belt and the flange does not, so the whole lift
    slides backwards relative to the object it is holding.
    """
    plan = a_plan()
    assert plan is not None
    descend = next(leg for leg in plan.legs if leg.phase is Phase.DESCEND).segment
    retreat = next(leg for leg in plan.legs if leg.phase is Phase.RETREAT).segment
    assert retreat.duration == pytest.approx(descend.duration)
    assert retreat.end.position[2] - descend.end.position[2] == pytest.approx(CLEARANCE)
    assert retreat.start.velocity == pytest.approx(descend.end.velocity)
    assert retreat.end.velocity == pytest.approx(
        np.asarray((BELT[0], BELT[1], BELT[2] + APPROACH_SPEED), dtype=np.float64)
    )
    # Vertical in the frame the object lives in: the belt carries the travel
    # and the belt's normal carries the lift, and nothing else moves.
    assert retreat.end.position[1] == pytest.approx(retreat.start.position[1])
    assert retreat.end.position[0] - retreat.start.position[0] == pytest.approx(
        BELT[0] * retreat.duration
    )


def test_a_vertical_velocity_does_not_move_the_grasp_plane() -> None:
    """AC-MOVE-46: a vertical velocity does not move the grasp plane.

    An object settling on the belt carries a downward velocity that gravity
    gives it and the belt does not. Predicting that component forward over an
    interception asks for a grasp below the object, and on this line below the
    belt itself, which is what the replay of a recorded run found.
    """
    still = a_plan()
    settling = a_plan(
        belt_velocity=np.asarray((BELT[0], BELT[1], -0.40), dtype=np.float64),
    )
    assert still is not None and settling is not None
    quiet = next(leg for leg in still.legs if leg.phase is Phase.DESCEND).segment
    sinking = next(leg for leg in settling.legs if leg.phase is Phase.DESCEND).segment
    assert sinking.end.position[2] == pytest.approx(quiet.end.position[2])
    assert sinking.end.position[2] == pytest.approx(OBJECT[2])


def test_a_lateral_velocity_carries_the_object_only_while_it_lasts() -> None:
    """AC-MOVE-46: drift across the belt is predicted for as long as it lasts.

    Only the height is not a transport axis. But the two horizontal axes are
    not one process either: along the belt the object is driven and its velocity
    holds for as long as the belt does, and across the belt nothing drives it
    and the drift decays within a fraction of a second. Measured on the shipped
    line the autocorrelation of the cross-belt velocity is +0.04 after 0.2 s and
    −0.02 after 0.8 s, and the p90 lateral speed is 0.261 m/s; carried over the
    four seconds a visit commits ahead that is 900 mm, and the run that did it
    aimed two visits in nine at a pose 208 and 346 mm from any object.
    """
    from pathlib import Path

    from clave.control.settings import ControlSettings
    from clave.world.config import load

    horizon = ControlSettings.load(
        load(
            Path(__file__).resolve().parents[2] / "configs" / "runtime" / "control.yml"
        )
    ).task.drift_horizon

    drifting = a_plan(belt_velocity=np.asarray((BELT[0], 0.05, 0.0), dtype=np.float64))
    assert drifting is not None
    across = next(leg for leg in drifting.legs if leg.phase is Phase.DESCEND).segment
    carried = across.end.position[1] - OBJECT[1]
    assert carried > 0.0, "the drift across the belt is not predicted at all"
    assert carried <= 0.05 * horizon + 1e-9, (
        "the drift is carried past the horizon it lasts for"
    )


def test_a_descent_is_retargeted_only_while_it_is_underway() -> None:
    """AC-MOVE-63: retargeting a descent moves its end and nothing earlier.

    Called before the descent there is still an approach to bend, and that is
    refine's job. Called during it, the end moves onto the position handed in
    and the arrival stays the instant the visit already chose.
    """
    plan = a_plan()
    assert plan is not None

    def steer(position: NDArray[np.float64], at: float) -> Plan | None:
        return retarget_descent(
            plan=plan,
            object_position=position,
            belt_velocity=BELT,
            approach_clearance_z=CLEARANCE,
            approach_speed=APPROACH_SPEED,
            dwell_seconds=DWELL,
            max_speed=1.00,
            max_acceleration=2.50,
            at_seconds=at,
            drift_horizon=DRIFT_HORIZON,
            minimum_segment_seconds=MINIMUM_SEGMENT,
            closing_rise_seconds=CLOSING_RISE,
            segment_sample_count=SEGMENT_SAMPLES,
            bisection_passes=BISECTION_PASSES,
            minimum_delivery_seconds=MINIMUM_DELIVERY,
            correction_steps=CORRECTION_STEPS,
        )

    assert steer(OBJECT, plan.started_at + 0.01) is None
    edges = plan._boundaries()
    index = next(i for i, leg in enumerate(plan.legs) if leg.phase is Phase.DESCEND)
    at = plan.started_at + edges[index] + 0.05
    # Where the belt has carried the object by this instant, plus 20 mm across
    # it. Handing the position from the start of the visit asks the descent to
    # run back up the belt, which is the correction the speed ceiling refuses.
    carried = np.asarray(
        (
            OBJECT[0] + BELT[0] * at,
            OBJECT[1] + 0.020,
            OBJECT[2],
        ),
        dtype=np.float64,
    )
    steered = steer(carried, at)
    assert steered is not None
    assert steered.pick_at == pytest.approx(plan.pick_at, abs=1e-9)
    end = next(
        leg.segment.end.position for leg in steered.legs if leg.phase is Phase.DESCEND
    )
    assert end[1] == pytest.approx(carried[1], abs=1e-9)
    assert end[0] == pytest.approx(OBJECT[0] + BELT[0] * plan.pick_at, abs=1e-9)
    assert end[2] == pytest.approx(OBJECT[2], abs=1e-9)
    for before, after in zip(steered.legs, steered.legs[1:], strict=False):
        assert after.segment.start.position == pytest.approx(
            before.segment.end.position
        )


def test_a_plan_runs_out_rather_than_extrapolating() -> None:
    """AC-MOVE-42: a plan runs out rather than extrapolating.

    A quintic asked for a time past its end diverges fast, so the plan
    reports that it is over instead of answering.
    """
    plan = a_plan()
    assert plan is not None
    assert plan.at(plan.started_at + plan.duration + 1e-9) is None
    before = plan.at(plan.started_at - 1.0)
    assert before is not None
    assert before[1].position == pytest.approx(PARK)


def test_no_interception_inside_the_window_is_no_plan() -> None:
    """AC-MOVE-41: no interception inside the window is no plan.

    Refusing is the honest answer for an object the arm cannot reach in the
    belt it has left, and it is what lets the caller move on to the next one.
    """
    assert a_plan(latest=0.20) is None


def test_the_duration_is_the_sum_of_its_legs() -> None:
    """AC-MOVE-42: the duration is the sum of its legs."""
    plan = a_plan()
    assert plan is not None
    assert plan.duration == pytest.approx(
        sum(leg.segment.duration for leg in plan.legs)
    )
    assert plan.at(plan.started_at + plan.duration) is not None


def test_re_aiming_keeps_the_arrival_time_and_moves_the_target() -> None:
    """AC-MOVE-40: re-aiming keeps the arrival time and moves the target.

    The whole point of separating the two: a duration that keeps changing is
    a duration nothing can be scheduled against, and an aim that never
    changes is an aim three seconds out of date.
    """
    plan = a_plan(margin=1.15)
    assert plan is not None
    drifted = np.asarray(
        (OBJECT[0] + BELT[0] * 0.5, OBJECT[1] + 0.040, OBJECT[2]),
        dtype=np.float64,
    )
    again = refine(
        plan=plan,
        object_position=drifted,
        belt_velocity=BELT,
        z_offset=CLEARANCE,
        approach_speed=APPROACH_SPEED,
        dwell_seconds=DWELL,
        max_speed=1.00,
        max_acceleration=2.50,
        at_seconds=0.5,
        drift_horizon=DRIFT_HORIZON,
        minimum_segment_seconds=MINIMUM_SEGMENT,
        closing_rise_seconds=CLOSING_RISE,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
        minimum_delivery_seconds=MINIMUM_DELIVERY,
        correction_steps=CORRECTION_STEPS,
    )
    assert again is not None
    assert again.pick_at == pytest.approx(plan.pick_at, abs=1e-9)
    moved = math.dist(
        plan.legs[1].segment.end.position, again.legs[1].segment.end.position
    )
    assert moved == pytest.approx(0.040, abs=1e-6)


def test_re_aiming_begins_where_the_arm_has_got_to() -> None:
    """AC-MOVE-40: re-aiming begins where the arm has got to.

    Position, velocity and acceleration, because a correction that only
    matched position would put a step in velocity into the middle of an
    approach.
    """
    plan = a_plan(margin=1.15)
    assert plan is not None
    here = plan.legs[0].segment.at(0.5)
    again = refine(
        plan=plan,
        object_position=OBJECT,
        belt_velocity=BELT,
        z_offset=CLEARANCE,
        approach_speed=APPROACH_SPEED,
        dwell_seconds=DWELL,
        max_speed=1.00,
        max_acceleration=2.50,
        at_seconds=0.5,
        drift_horizon=DRIFT_HORIZON,
        minimum_segment_seconds=MINIMUM_SEGMENT,
        closing_rise_seconds=CLOSING_RISE,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
        minimum_delivery_seconds=MINIMUM_DELIVERY,
        correction_steps=CORRECTION_STEPS,
    )
    assert again is not None
    start = again.legs[0].segment.start
    assert start.position == pytest.approx(here.position)
    assert start.velocity == pytest.approx(here.velocity)
    assert start.acceleration == pytest.approx(here.acceleration)


def test_nothing_is_re_aimed_once_the_descent_has_begun() -> None:
    """AC-MOVE-40: nothing is re-aimed once the descent has begun.

    Swapping the target under a jaw already coming down turns an approach
    into a swipe.
    """
    plan = a_plan(margin=1.15)
    assert plan is not None
    descending = plan.started_at + plan.legs[0].segment.duration + 0.01
    assert (
        refine(
            plan=plan,
            object_position=OBJECT,
            belt_velocity=BELT,
            z_offset=CLEARANCE,
            approach_speed=APPROACH_SPEED,
            dwell_seconds=DWELL,
            max_speed=1.00,
            max_acceleration=2.50,
            at_seconds=descending,
            drift_horizon=DRIFT_HORIZON,
            minimum_segment_seconds=MINIMUM_SEGMENT,
            closing_rise_seconds=CLOSING_RISE,
            segment_sample_count=SEGMENT_SAMPLES,
            bisection_passes=BISECTION_PASSES,
            minimum_delivery_seconds=MINIMUM_DELIVERY,
            correction_steps=CORRECTION_STEPS,
        )
        is None
    )


def test_a_margin_widens_the_correction_an_arc_will_accept() -> None:
    """AC-MOVE-42: a margin widens the correction an arc will accept.

    Which is why it exists. The bisection returns the soonest arc that fits,
    so it sits on whichever ceiling binds, and a correction large enough to
    matter breaks the bound it was already touching. Asserting that some one
    correction is refused would be asserting more than that: whether a given
    nudge fits depends on where it points, and the property worth pinning is
    that the room grows.
    """

    def accepted(margin: float, sideways: float) -> bool:
        plan = a_plan(margin=margin)
        assert plan is not None
        return (
            refine(
                plan=plan,
                object_position=np.asarray(
                    (OBJECT[0], OBJECT[1] + sideways, OBJECT[2]),
                    dtype=np.float64,
                ),
                belt_velocity=BELT,
                z_offset=CLEARANCE,
                approach_speed=APPROACH_SPEED,
                dwell_seconds=DWELL,
                max_speed=1.00,
                max_acceleration=2.50,
                at_seconds=0.5,
                drift_horizon=DRIFT_HORIZON,
                minimum_segment_seconds=MINIMUM_SEGMENT,
                closing_rise_seconds=CLOSING_RISE,
                segment_sample_count=SEGMENT_SAMPLES,
                bisection_passes=BISECTION_PASSES,
                minimum_delivery_seconds=MINIMUM_DELIVERY,
                correction_steps=CORRECTION_STEPS,
            )
            is not None
        )

    def widest(margin: float) -> float:
        low, high = 0.0, 2.0
        for _ in range(30):
            middle = 0.5 * (low + high)
            if accepted(margin, middle):
                low = middle
            else:
                high = middle
        return low

    assert widest(1.15) > widest(1.00)


def test_a_margin_buys_its_room_by_intercepting_later() -> None:
    """AC-MOVE-42: a margin buys its room by intercepting later."""
    soonest, roomy = a_plan(margin=1.0), a_plan(margin=1.15)
    assert soonest is not None and roomy is not None
    assert roomy.pick_at > soonest.pick_at
    assert roomy.pick_at == pytest.approx(
        (soonest.pick_at - 0.40) * 1.15 + 0.40, rel=1e-6
    )


def test_a_visit_ends_over_its_chute_rather_than_over_the_belt() -> None:
    """AC-DROP-05: a visit ends over its chute rather than over the belt.

    Without a delivery leg the plan runs out at the retreat and the jaw
    opens fifty millimetres above the belt, which drops the object back
    where it came from. That is what the line did before this existed.
    """
    mouth = (0.58, -0.42, 0.90)
    plan = a_plan(over=mouth)
    assert plan is not None
    assert [leg.phase for leg in plan.legs][-1] is Phase.DELIVER
    last = plan.legs[-1].segment
    assert last.end.position == pytest.approx(mouth)
    assert last.end.velocity == pytest.approx((0.0, 0.0, 0.0))


def test_the_jaw_holds_all_the_way_to_the_chute() -> None:
    """AC-DROP-05: the jaw holds all the way to the chute.

    Opening at the end of the retreat is the failure this leg removes, so
    the grip has to survive every leg after the hold.
    """
    plan = a_plan(over=(0.58, -0.42, 0.90))
    assert plan is not None
    after_hold = False
    for leg in plan.legs:
        if leg.phase is Phase.HOLD:
            after_hold = True
        if after_hold:
            assert leg.grip == JAW_SHUT, f"{leg.phase} let go early"


def test_the_delivery_stays_inside_the_speed_ceiling() -> None:
    """AC-DROP-05: the delivery stays inside the speed ceiling.

    Nothing constrains this arc's duration from outside, so it is set from
    the distance and the ceiling: a rest-to-rest quintic peaks at 15/8 of
    its mean speed, and taking that as the duration touches the ceiling
    once rather than crossing it.
    """
    plan = a_plan(over=(1.14, -0.42, 0.90), max_speed=1.00)
    assert plan is not None
    assert plan.legs[-1].segment.peak_speed(SEGMENT_SAMPLES) <= 1.00 + 1e-6


def test_no_chute_means_no_delivery_rather_than_a_broken_one() -> None:
    """AC-DROP-05: no chute means no delivery rather than a broken one.

    A channel with no opening is a configuration error the world refuses at
    load. Here it means the caller asked for a visit that ends at the
    retreat, which is what the motion-only profiles want.
    """
    plan = a_plan()
    assert plan is not None
    assert [leg.phase for leg in plan.legs][-1] is Phase.RETREAT


def test_approach_enters_via_border_with_horizontal_perpendicular_speed() -> None:
    """AC-DROP-06: approach routes through border with transverse horizontal speed."""
    mouth = (0.58, -0.42, 0.90)
    plan = a_plan(over=mouth)
    assert plan is not None
    entry_leg = plan.legs[0]
    assert entry_leg.phase is Phase.TRACK
    assert entry_leg.segment.end.position[0] == pytest.approx(0.58)
    assert entry_leg.segment.end.position[1] == pytest.approx(-0.25)
    assert entry_leg.segment.end.position[2] >= 1.20
    assert entry_leg.segment.end.velocity[0] == pytest.approx(0.0)
    assert entry_leg.segment.end.velocity[1] > 0.0
    assert entry_leg.segment.end.velocity[2] == pytest.approx(0.0)


def test_the_barrier_is_cleared_by_an_arc_of_its_own() -> None:
    """AC-MOVE-49 and AC-DROP-06: the barrier is cleared by an arc of its own.

    The retreat leaves the object; the arc after it lifts to the height the
    delivery crosses the belt's side barrier at. Folding the two together is
    what made the retreat's shape depend on where the next chute stands.
    """
    mouth = (0.58, -0.42, 0.90)
    plan = a_plan(over=mouth)
    assert plan is not None
    lifts = [leg.segment for leg in plan.legs if leg.phase is Phase.RETREAT]
    assert len(lifts) == 2
    retreat, climb = lifts
    assert retreat.end.position[2] - retreat.start.position[2] == pytest.approx(
        CLEARANCE
    )
    assert climb.end.position[2] >= 1.20
    for step in range(11):
        state = climb.at(climb.duration * step / 10.0)
        assert state.position[1] == pytest.approx(climb.start.position[1])
        assert state.velocity[1] == pytest.approx(0.0)


def test_the_retreat_does_not_depend_on_where_the_delivery_goes() -> None:
    """AC-MOVE-49: the retreat does not depend on where the delivery goes.

    Two visits to the same object with a barrier at two heights fly the same
    retreat, or the shape of the pick is decided by the belt's furniture.
    """
    mouth = (0.58, -0.42, 0.90)
    low = a_plan(over=mouth, safe_height_world=1.20)
    high = a_plan(over=mouth, safe_height_world=1.45)
    assert low is not None and high is not None
    first = next(leg for leg in low.legs if leg.phase is Phase.RETREAT).segment
    second = next(leg for leg in high.legs if leg.phase is Phase.RETREAT).segment
    assert first.duration == pytest.approx(second.duration)
    assert first.end.velocity == pytest.approx(second.end.velocity)
    # Its own shape, which is what the barrier must not decide. Where the arm
    # came from is allowed to differ: a higher entry arc is a different arc.
    for axis in range(3):
        assert first.end.position[axis] - first.start.position[axis] == pytest.approx(
            second.end.position[axis] - second.start.position[axis]
        )


def test_delivery_exits_via_border_with_horizontal_perpendicular_speed() -> None:
    """AC-DROP-06: delivery passes through border with transverse horizontal speed."""
    mouth = (0.58, -0.42, 0.90)
    plan = a_plan(over=mouth)
    assert plan is not None
    deliver_legs = [leg for leg in plan.legs if leg.phase is Phase.DELIVER]
    assert len(deliver_legs) == 2
    border_leg = deliver_legs[0]
    assert border_leg.segment.end.position[0] == pytest.approx(0.58)
    assert border_leg.segment.end.position[1] == pytest.approx(-0.25)
    assert border_leg.segment.end.position[2] >= 1.20
    assert border_leg.segment.end.velocity[0] == pytest.approx(0.0)
    assert border_leg.segment.end.velocity[1] < 0.0
    assert border_leg.segment.end.velocity[2] == pytest.approx(0.0)


def test_parameters_govern_border_clearance_and_cross_speed() -> None:
    """AC-DROP-06: belt geometry, safe clearance and cross speed govern arcs."""
    mouth = (0.60, -0.50, 0.85)
    plan = a_plan(
        over=mouth,
        belt_width=0.80,
        belt_center_y=0.10,
        safe_height_world=1.35,
        cross_speed=0.40,
    )
    assert plan is not None
    # Border y: 0.10 - 0.80 / 2.0 = -0.30
    entry_leg = plan.legs[0]
    assert entry_leg.phase is Phase.TRACK
    assert entry_leg.segment.end.position[0] == pytest.approx(0.60)
    assert entry_leg.segment.end.position[1] == pytest.approx(-0.30)
    assert entry_leg.segment.end.position[2] >= 1.35
    assert entry_leg.segment.end.velocity[1] == pytest.approx(0.40)

    deliver_legs = [leg for leg in plan.legs if leg.phase is Phase.DELIVER]
    assert len(deliver_legs) == 2
    border_retreat_leg = deliver_legs[0]
    assert border_retreat_leg.segment.end.position[0] == pytest.approx(0.60)
    assert border_retreat_leg.segment.end.position[1] == pytest.approx(-0.30)
    assert border_retreat_leg.segment.end.position[2] >= 1.35
    assert border_retreat_leg.segment.end.velocity[1] == pytest.approx(-0.40)


def test_plan_smooths_and_updates_yaw_through_quintic_interpolation() -> None:
    """Plan interpolates yaw smoothly across approach with parallel-jaw symmetry."""
    plan = a_plan()
    assert plan is not None
    oriented_plan = plan.with_yaw(target_yaw=1.20, initial_yaw=0.20)
    assert oriented_plan.target_yaw == pytest.approx(1.20)
    assert oriented_plan.initial_yaw == pytest.approx(0.20)

    # At start, yaw matches initial_yaw
    start_yaw = oriented_plan.yaw_at(oriented_plan.started_at)
    assert start_yaw is not None
    assert start_yaw == pytest.approx(0.20)

    # At pick_at or beyond, yaw matches target_yaw
    pick_yaw = oriented_plan.yaw_at(oriented_plan.pick_at)
    assert pick_yaw is not None
    assert pick_yaw == pytest.approx(1.20)
    assert oriented_plan.yaw_at(oriented_plan.pick_at + 1.0) == pytest.approx(1.20)

    # Midpoint smoothly transitions
    mid_t = 0.5 * (oriented_plan.started_at + oriented_plan.pick_at)
    mid_yaw = oriented_plan.yaw_at(mid_t)
    assert mid_yaw is not None
    assert 0.20 < mid_yaw < 1.20

    # Parallel jaw 180-degree symmetry: target at yaw + pi resolves with zero spin
    sym_plan = plan.with_yaw(target_yaw=0.20 + math.pi, initial_yaw=0.20)
    sym_mid = sym_plan.yaw_at(mid_t)
    assert sym_mid is not None
    assert sym_mid == pytest.approx(0.20)


def test_a_descent_correction_past_the_ceiling_is_taken_part_way() -> None:
    """AC-MOVE-65: a descent correction past the ceiling is taken part way.

    Handing the object's position from the start of the visit asks the descent
    to run back up the belt. The whole correction breaks the speed ceiling.
    The part that stays under it still moves the end, and the arc that flies
    respects the ceiling.
    """
    plan = a_plan()
    assert plan is not None
    edges = plan._boundaries()
    index = next(i for i, leg in enumerate(plan.legs) if leg.phase is Phase.DESCEND)
    at = plan.started_at + edges[index] + 0.05
    stale = next(
        leg.segment.end.position for leg in plan.legs if leg.phase is Phase.DESCEND
    )
    steered = retarget_descent(
        plan=plan,
        object_position=OBJECT,
        belt_velocity=BELT,
        approach_clearance_z=CLEARANCE,
        approach_speed=APPROACH_SPEED,
        dwell_seconds=DWELL,
        max_speed=1.00,
        max_acceleration=2.50,
        at_seconds=at,
        drift_horizon=DRIFT_HORIZON,
        minimum_segment_seconds=MINIMUM_SEGMENT,
        closing_rise_seconds=CLOSING_RISE,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
        minimum_delivery_seconds=MINIMUM_DELIVERY,
        correction_steps=CORRECTION_STEPS,
    )
    assert steered is not None
    descent = next(leg.segment for leg in steered.legs if leg.phase is Phase.DESCEND)
    assert descent.peak_speed(SEGMENT_SAMPLES) <= 1.00 + 1e-6
    assert math.dist(descent.end.position, OBJECT) < math.dist(stale, OBJECT)


def test_an_approach_correction_past_the_ceiling_is_taken_part_way() -> None:
    """AC-MOVE-66: an approach correction past the ceiling is taken part way.

    Late in the approach, little time remains. The whole correction does not
    fit and the fraction that does is what flies, including the descent hung
    off that fraction. A fraction of zero would be the plan already in hand,
    which refine reports by returning None.
    """
    plan = a_plan(margin=1.15)
    assert plan is not None
    track = next(leg.segment for leg in plan.legs if leg.phase is Phase.TRACK)
    at = track.duration - 0.30
    wild = np.asarray(
        (OBJECT[0] + BELT[0] * at, OBJECT[1] + 1.10, OBJECT[2]),
        dtype=np.float64,
    )
    again = refine(
        plan=plan,
        object_position=wild,
        belt_velocity=BELT,
        z_offset=CLEARANCE,
        approach_speed=APPROACH_SPEED,
        dwell_seconds=DWELL,
        max_speed=1.00,
        max_acceleration=2.50,
        at_seconds=at,
        drift_horizon=DRIFT_HORIZON,
        minimum_segment_seconds=MINIMUM_SEGMENT,
        closing_rise_seconds=CLOSING_RISE,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
        minimum_delivery_seconds=MINIMUM_DELIVERY,
        correction_steps=CORRECTION_STEPS,
    )
    assert again is not None
    old = next(
        leg.segment.end.position for leg in plan.legs if leg.phase is Phase.DESCEND
    )
    new = next(
        leg.segment.end.position for leg in again.legs if leg.phase is Phase.DESCEND
    )
    assert math.dist(new, wild) < math.dist(old, wild)
    assert abs(new[1] - wild[1]) > 0.20
    for leg in again.legs:
        if leg.phase is Phase.TRACK:
            assert leg.segment.fits(1.00, 2.50, SEGMENT_SAMPLES)
        assert leg.segment.peak_speed(SEGMENT_SAMPLES) <= 1.00 + 1e-6


def test_the_hold_rises_while_the_jaw_closes() -> None:
    """AC-MOVE-67: the hold rises by how much further the shut jaw hangs.

    The descent arrives at the open jaw's clearance. The first part of the
    hold climbs the extra hang, and the rest carries at that height. The
    retreat still lifts its own clearance above where the hold finished.
    """
    drop = 0.0132
    plan = a_plan(jaw_rise=drop)
    assert plan is not None
    holds = [leg for leg in plan.legs if leg.phase is Phase.HOLD]
    assert [leg.grip for leg in holds] == [JAW_SHUT, JAW_SHUT]
    rise, carry = (leg.segment for leg in holds)
    assert rise.duration == pytest.approx(0.20)
    assert rise.end.position[2] - rise.start.position[2] == pytest.approx(drop)
    assert carry.end.position[2] == pytest.approx(rise.end.position[2])
    assert carry.end.velocity == pytest.approx(BELT)
    retreat = next(leg.segment for leg in plan.legs if leg.phase is Phase.RETREAT)
    assert retreat.end.position[2] - carry.end.position[2] == pytest.approx(CLEARANCE)


def test_matching_transport_drops_lateral_velocity() -> None:
    """AC-MOVE-71: matching transport is belt-axis only.

    Lateral body velocity aims the target origin for a horizon; it is not the
    velocity HOLD and RETREAT ride.
    """
    from clave.control.pick import _drift_velocity, _transport

    measured = np.asarray((0.314, 0.50, -0.10), dtype=np.float64)
    assert _transport(measured) == pytest.approx((0.314, 0.0, 0.0))
    assert _drift_velocity(measured) == pytest.approx((0.314, 0.50, 0.0))
    assert _transport(
        np.asarray((-0.20, 0.10, 0.0), dtype=np.float64)
    ) == pytest.approx((0.0, 0.0, 0.0))


def test_hold_and_retreat_are_vertical_in_the_object_frame() -> None:
    """AC-MOVE-69 / AC-MOVE-72: hold≈rest and retreat=+Z in the object frame.

    World motion during those legs is only the belt-axis transport composed
    with the local arc. Subtracting transport * t leaves pure vertical retreat
    and no horizontal hold speed.
    """
    plan = a_plan(belt_velocity=np.asarray((BELT[0], 0.40, 0.0), dtype=np.float64))
    assert plan is not None
    transport = plan.transport_velocity
    assert transport == pytest.approx(np.asarray((BELT[0], 0.0, 0.0), dtype=np.float64))

    hold = next(leg for leg in plan.legs if leg.phase is Phase.HOLD).segment
    for step in range(11):
        state = hold.at(hold.duration * step / 10.0)
        assert state.velocity[1] == pytest.approx(0.0, abs=1e-9)
        assert state.velocity[0] == pytest.approx(transport[0], abs=1e-9)

    retreat = next(leg for leg in plan.legs if leg.phase is Phase.RETREAT).segment
    # Object frame: subtract transport travel from start to end.
    local = (
        retreat.end.position[0]
        - retreat.start.position[0]
        - transport[0] * retreat.duration,
        retreat.end.position[1]
        - retreat.start.position[1]
        - transport[1] * retreat.duration,
        retreat.end.position[2] - retreat.start.position[2],
    )
    assert local[0] == pytest.approx(0.0, abs=1e-9)
    assert local[1] == pytest.approx(0.0, abs=1e-9)
    assert local[2] == pytest.approx(CLEARANCE, abs=1e-9)


def test_a_spiked_lateral_velocity_does_not_rewrite_hold_or_retreat() -> None:
    """AC-MOVE-72: a late-descent retarget must not put HOLD on spiked vy.

    Contact can give the parcel a large lateral cvel. Aim may move; matching
    transport stays belt-axis.
    """
    plan = a_plan()
    assert plan is not None
    edges = plan._boundaries()
    index = next(i for i, leg in enumerate(plan.legs) if leg.phase is Phase.DESCEND)
    at = plan.started_at + edges[index] + 0.05
    spiked = np.asarray((BELT[0], 0.50, 0.0), dtype=np.float64)
    steered = retarget_descent(
        plan=plan,
        object_position=np.asarray(
            (OBJECT[0] + 0.02, OBJECT[1] + 0.03, OBJECT[2]),
            dtype=np.float64,
        ),
        belt_velocity=spiked,
        approach_clearance_z=CLEARANCE,
        approach_speed=APPROACH_SPEED,
        dwell_seconds=DWELL,
        max_speed=1.00,
        max_acceleration=2.50,
        at_seconds=at,
        drift_horizon=DRIFT_HORIZON,
        minimum_segment_seconds=MINIMUM_SEGMENT,
        closing_rise_seconds=CLOSING_RISE,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
        minimum_delivery_seconds=MINIMUM_DELIVERY,
        correction_steps=CORRECTION_STEPS,
    )
    assert steered is not None
    assert steered.transport_velocity == pytest.approx(
        np.asarray((BELT[0], 0.0, 0.0), dtype=np.float64),
    )
    hold = next(leg for leg in steered.legs if leg.phase is Phase.HOLD).segment
    assert hold.start.velocity[1] == pytest.approx(0.0, abs=1e-9)
    assert abs(hold.start.velocity[1]) < 0.05
    retreat = next(leg for leg in steered.legs if leg.phase is Phase.RETREAT).segment
    assert retreat.start.velocity[1] == pytest.approx(0.0, abs=1e-9)


def test_delivery_is_an_intercept_to_a_stationary_chute() -> None:
    """AC-MOVE-70: delivery is interception with v_T = 0 at the chute.

    The visit ends at rest over the mouth. That is the stationary special case
    of the same interception formulation the pick uses.
    """
    mouth = (0.58, -0.42, 1.20)
    plan = a_plan(
        over=mouth,
        belt_border_y=-0.25,
        safe_height_world=1.20,
        cross_speed=APPROACH_SPEED,
    )
    assert plan is not None
    delivers = [leg for leg in plan.legs if leg.phase is Phase.DELIVER]
    assert delivers, "a chute visit must include delivery legs"
    last = delivers[-1].segment
    assert last.end.position == pytest.approx(mouth)
    assert last.end.velocity == pytest.approx((0.0, 0.0, 0.0))


def test_commanded_yaw_picks_the_nearer_parallel_grip() -> None:
    """AC-BEHAVE-03: θ or θ+π, whichever is nearer the tool's current yaw."""
    assert nearer_parallel_yaw(0.0, 0.1) == pytest.approx(0.0)
    # π and −π are the same angle after wrap.
    assert abs(nearer_parallel_yaw(0.0, math.pi)) == pytest.approx(math.pi, abs=1e-9)
    # Target near −π/2 while the tool sits near +π/2: flipping lands closer.
    chosen = nearer_parallel_yaw(-math.pi / 2.0, math.pi / 2.0)
    assert chosen == pytest.approx(math.pi / 2.0, abs=1e-9)


def test_descent_retarget_does_not_project_lateral_drift() -> None:
    """AC-BEHAVE-02: descent splice aims with transport, not lateral drift.

    A fresher object *position* still moves the end (AC-MOVE-63). Inventing
    future cross-belt travel over the remaining time is what produced the
    side jab at grasp height.
    """
    plan = a_plan()
    assert plan is not None
    edges = plan._boundaries()
    index = next(i for i, leg in enumerate(plan.legs) if leg.phase is Phase.DESCEND)
    at = plan.started_at + edges[index] + 0.05
    remaining = edges[index + 1] - (at - plan.started_at)
    # Object is already at y=0.02; body velocity claims 0.5 m/s of lateral
    # drift. Projecting that over `remaining` would move the end by ~0.1 m.
    carried = np.asarray(
        (OBJECT[0] + BELT[0] * at, 0.020, OBJECT[2]),
        dtype=np.float64,
    )
    drifting = np.asarray((BELT[0], 0.50, 0.0), dtype=np.float64)
    steered = retarget_descent(
        plan=plan,
        object_position=carried,
        belt_velocity=drifting,
        approach_clearance_z=CLEARANCE,
        approach_speed=APPROACH_SPEED,
        dwell_seconds=DWELL,
        max_speed=1.00,
        max_acceleration=2.50,
        at_seconds=at,
        drift_horizon=DRIFT_HORIZON,
        minimum_segment_seconds=MINIMUM_SEGMENT,
        closing_rise_seconds=CLOSING_RISE,
        segment_sample_count=SEGMENT_SAMPLES,
        bisection_passes=BISECTION_PASSES,
        minimum_delivery_seconds=MINIMUM_DELIVERY,
        correction_steps=CORRECTION_STEPS,
    )
    assert steered is not None
    end = next(
        leg.segment.end.position for leg in steered.legs if leg.phase is Phase.DESCEND
    )
    assert end[1] == pytest.approx(0.020, abs=1e-6)
    assert abs(end[1] - (0.020 + 0.50 * remaining)) > 0.05
