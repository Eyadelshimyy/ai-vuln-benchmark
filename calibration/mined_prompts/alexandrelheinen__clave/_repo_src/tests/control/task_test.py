"""The phases of one visit, under both profiles.

The two are driven differently and are tested differently because of it. A
motion-only visit is stepped toward a goal and ends when the flange gets
there, so its tests move the flange onto the goal by hand. A full visit is
planned as timed arcs and ends when the clock says so, so its tests advance
the clock and read what the plan asks for.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.control.selection import Candidate, Queue
from clave.control.settings import (
    CalibrationSettings,
    ControlSettings,
    MotionSettings,
    Phase,
    Profile,
    TaskSettings,
)
from clave.control.story import StoryLog
from clave.control.task import TaskError, TaskMachine
from clave.control.trajectory import same
from clave.world.effector import Effector

ROOT = Path(__file__).resolve().parents[2]
BELT_SURFACE = 0.90
PARK = np.asarray((0.45, -1.00, 1.20), dtype=np.float64)
BELT_SPEED = 0.314
LIMITS = MotionSettings(
    max_speed=1.00,
    max_acceleration=2.50,
    intercept_passes=3,
    minimum_segment_seconds=0.50,
    segment_sample_count=64,
    bisection_passes=40,
    minimum_delivery_seconds=0.05,
    correction_steps=32,
)
# Test-only algorithm parameters (production loads these from control.yml).
DRIFT_HORIZON = 0.30
CLOSING_RISE = 0.20
AIM_TOLERANCE = 0.030
LEG_SAMPLES = 5
EFFECTOR = Effector(
    finger_length=0.1558,
    pad_thickness=0.008,
    pad_depth=0.022,
    pad_height=0.0375,
    grasp_height=0.045,
    lowest_below_flange=0.1735,
    open_lowest_below_flange=0.1603,
    jaw_clearance=0.025,
    opening=0.085,
)
"""The shipped jaw, so a full visit has the geometry it plans against."""

PICK_Z = BELT_SURFACE + EFFECTOR.flange_floor
"""The lowest flange pose the jaw may take, which is where a marker sits.

Written as arithmetic rather than as a number because it moved when the
clearance grew to cover the load the jaw picks up, and a fixture pinned to the
old millimetre silently stopped planning anything at all.
"""

PINCH_Z = PICK_Z - EFFECTOR.finger_length
"""Where the pads close for that pose, which is what a marker's anchor holds."""


def task_settings(**overrides: object) -> TaskSettings:
    """Task settings for the motion-only profile."""
    fields: dict[str, object] = {
        "profile": Profile.MOTION_ONLY,
        "approach_height": 0.220,
        "arrival_tolerance": 0.010,
        "dwell_seconds": 0.30,
        "grasp_clearance": 0.050,
        "approach_speed": 0.25,
        "interception_limit": 4.00,
        "interception_margin": 1.15,
        "park_position": PARK,
        "park_marker_color": (0.85, 0.10, 0.10),
        "aim_tolerance": AIM_TOLERANCE,
        "drift_horizon": DRIFT_HORIZON,
        "closing_rise_seconds": CLOSING_RISE,
        "leg_samples": LEG_SAMPLES,
    }
    fields.update(overrides)
    return TaskSettings(**fields)  # type: ignore[arg-type]


def machine(**overrides: object) -> TaskMachine:
    """A machine with no calibration offset unless a test adds one.

    The belt and the ceilings go in whatever the profile, because a
    motion-only machine ignores them and a test that switches profile should
    not also have to remember to switch its fixture.
    """
    story = overrides.pop("story", None)
    if story is not None and not isinstance(story, StoryLog):
        raise TypeError("story must be a StoryLog")
    raw_chutes = overrides.pop("chutes", None)
    chutes = (
        cast("dict[str, NDArray[np.float64]]", raw_chutes)
        if raw_chutes is not None
        else None
    )
    return TaskMachine(
        task_settings(**overrides),
        CalibrationSettings(
            flange_offset=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        ),
        belt_surface=BELT_SURFACE,
        belt_speed=BELT_SPEED,
        motion=LIMITS,
        effector=EFFECTOR,
        story=story,
        chutes=chutes,
    )


def queue_of(*candidates: Candidate) -> Queue:
    """A queue holding these candidates in this order."""
    return Queue(order=candidates, recomputed=True)


def candidate(track_id: int = 1, x: float = 0.30, y: float = 0.0) -> Candidate:
    """One candidate the arm could serve."""
    return Candidate(
        track_id=track_id,
        anchor=np.asarray((x, y, PINCH_Z), dtype=np.float64),
        flange=np.asarray((x, y, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
    )


def test_an_empty_queue_sends_the_arm_to_park() -> None:
    """AC-MOVE-03: an empty queue sends the arm to park."""
    goal = machine().step(
        queue_of(),
        flange=np.asarray((0.0, 0.0, 1.30), dtype=np.float64),
        at_seconds=0.0,
    )
    assert goal.phase is Phase.PARK
    assert goal.position == pytest.approx(PARK)
    assert goal.track_id is None


def test_arriving_at_park_settles_into_standby() -> None:
    """An empty queue, once the arm is there, settles into standby."""
    arm = machine()
    arm.step(
        queue_of(),
        flange=np.asarray((0.0, 0.0, 1.30), dtype=np.float64),
        at_seconds=0.0,
    )
    goal = arm.step(queue_of(), flange=PARK, at_seconds=0.1)
    assert goal.phase is Phase.STANDBY
    assert goal.position == pytest.approx(PARK)


def test_a_queue_with_a_head_puts_the_arm_in_tracking() -> None:
    """AC-MOVE-04: a queue with a head puts the arm in tracking."""
    goal = machine().step(queue_of(candidate(7)), flange=PARK, at_seconds=0.0)
    assert goal.phase is Phase.TRACK
    assert goal.track_id == 7


def test_the_goal_rides_at_the_configured_approach_height() -> None:
    """The goal rides at the configured approach height above the belt.

    Reference only, so the flange never descends: the height it tracks at is the
    height it holds.
    """
    goal = machine().step(queue_of(candidate()), flange=PARK, at_seconds=0.0)
    assert goal.position[2] == pytest.approx(BELT_SURFACE + 0.220)


def test_the_goal_carries_the_calibration_offset() -> None:
    """AC-MOVE-06: the goal carries the calibration offset."""
    offset = TaskMachine(
        task_settings(),
        CalibrationSettings(
            flange_offset=np.asarray((0.01, -0.02, 0.03), dtype=np.float64),
        ),
        belt_surface=BELT_SURFACE,
    )
    goal = offset.step(queue_of(candidate(x=0.30, y=0.10)), PARK, 0.0)
    assert goal.position[0] == pytest.approx(0.31)
    assert goal.position[1] == pytest.approx(0.08)
    assert goal.position[2] == pytest.approx(BELT_SURFACE + 0.220 + 0.03)


def test_the_goal_turns_the_tool_to_the_closing_axis() -> None:
    """AC-MOVE-07: the goal turns the tool to the closing axis."""
    goal = machine().step(queue_of(candidate()), PARK, 0.0)
    assert goal.yaw == pytest.approx(math.pi / 2.0)


def test_an_unoriented_candidate_asks_for_no_rotation() -> None:
    """AC-MOVE-08: an unoriented candidate asks for no rotation."""
    round_one = Candidate(
        track_id=1,
        anchor=np.asarray((0.3, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.3, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=None,
        distance_before_leaving=1.0,
    )
    assert machine().step(queue_of(round_one), PARK, 0.0).yaw is None


def test_a_visit_is_served_after_the_dwell_and_the_arm_moves_on() -> None:
    """AC-MOVE-04: a visit is served after the dwell, and the arm moves on."""
    arm = machine()
    first, second = candidate(1, x=0.30), candidate(2, x=0.60)
    here = np.asarray((0.30, 0.0, BELT_SURFACE + 0.220), dtype=np.float64)

    assert arm.step(queue_of(first, second), PARK, 0.0).track_id == 1
    # Arrived, but the dwell has not elapsed.
    assert arm.step(queue_of(first, second), here, 0.1).track_id == 1
    assert arm.step(queue_of(first, second), here, 0.3).track_id == 1
    # The dwell elapsed, so this track is served and the next one is taken.
    assert arm.step(queue_of(first, second), here, 0.45).track_id == 2
    assert arm.served == (1,)


def test_a_track_still_out_of_reach_does_not_start_its_dwell() -> None:
    """A track still out of tolerance does not start its dwell.

    Otherwise a visit completes because time passed rather than because the
    arm arrived, which is the failure that makes a distance report meaningless.
    """
    arm = machine()
    only = candidate(1, x=0.30)
    far = np.asarray((0.0, 0.0, 1.30), dtype=np.float64)
    for at_seconds in (0.0, 0.5, 1.0, 5.0):
        assert arm.step(queue_of(only), far, at_seconds).track_id == 1
    assert arm.served == ()


def test_a_refusal_faults_the_track_and_the_arm_holds_still() -> None:
    """AC-MOVE-05: a refusal faults the track and the arm holds still."""
    arm = machine()
    here = np.asarray((0.1, 0.2, 1.10), dtype=np.float64)
    goal = arm.step(queue_of(candidate(3)), here, 0.0, refusal="outside the annulus")
    assert goal.phase is Phase.FAULT
    assert goal.position == pytest.approx(here)
    assert arm.faults == ((3, "outside the annulus"),)


def test_a_refused_track_is_not_offered_again() -> None:
    """AC-MOVE-05: a refused track is not offered again.

    A pose the solver or the envelope refused will be refused next tick too,
    so retrying it forever is an arm that stops working on the first bad pose.
    """
    arm = machine()
    bad, good = candidate(3, x=0.30), candidate(4, x=0.60)
    arm.step(queue_of(bad, good), PARK, 0.0, refusal="outside the annulus")
    goal = arm.step(queue_of(bad, good), PARK, 0.1)
    assert goal.phase is Phase.TRACK
    assert goal.track_id == 4


def test_a_track_that_leaves_the_queue_is_abandoned() -> None:
    """A track that leaves the queue is abandoned.

    The object fell off the end of the belt or its track retired. Either way
    the arm has nothing to arrive at, and holding the old goal would park the
    flange over empty belt.
    """
    arm = machine()
    arm.step(queue_of(candidate(1)), PARK, 0.0)
    goal = arm.step(queue_of(candidate(2, x=0.80)), PARK, 0.1)
    assert goal.track_id == 2
    assert arm.served == ()


def _fly(arm: TaskMachine, only: Candidate, span: float = 8.0) -> list[Phase]:
    """Run a planned visit to its end and return the phases it passed through.

    Args:
        arm: The machine, already committed or about to be.
        only: The candidate to serve.
        span: How many seconds to run for at worst.

    Returns:
        The phases seen, in order and without repeats.
    """
    arm.step(queue_of(only), PARK, 0.0)
    seen: list[Phase] = []
    at_seconds = 0.0
    while at_seconds < span:
        at_seconds += 0.01
        flown = arm.tick(at_seconds, PARK)
        if flown is None:
            break
        seen.append(flown.phase)
    return list(dict.fromkeys(seen))


def test_a_full_visit_tracks_descends_grasps_and_retreats() -> None:
    """AC-MOVE-18: a full visit tracks, descends, grasps and retreats."""
    arm = machine(profile=Profile.FULL_VISIT)
    assert _fly(arm, candidate(1, x=0.30)) == [
        Phase.TRACK,
        Phase.DESCEND,
        Phase.HOLD,
        Phase.RETREAT,
    ]
    assert arm.served == (1,)
    assert arm.active is False


def test_a_planned_visit_shuts_the_jaw_only_once_it_is_on_the_object() -> None:
    """AC-MOVE-38: a planned visit shuts the jaw only once it is on the object.

    A jaw that closes on the way down sweeps the object off the belt, which
    is exactly the failure the whole formulation exists to remove.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    only = candidate(1, x=0.30)
    arm.step(queue_of(only), PARK, 0.0)
    at_seconds, shut_from = 0.0, None
    while at_seconds < 8.0:
        at_seconds += 0.01
        flown = arm.tick(at_seconds, PARK)
        if flown is None:
            break
        if flown.grip > 0.0 and shut_from is None:
            shut_from = flown.phase
    assert shut_from is Phase.HOLD


def test_a_planned_visit_meets_the_object_moving_with_the_belt() -> None:
    """AC-MOVE-39: a planned visit meets the object moving with the belt.

    A jaw arriving at rest has the object sliding through it at belt speed,
    which is the one thing a grasp cannot tolerate.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    only = candidate(1, x=0.30)
    arm.step(queue_of(only), PARK, 0.0)
    at_seconds = 0.0
    while at_seconds < 8.0:
        at_seconds += 0.01
        flown = arm.tick(at_seconds, PARK)
        assert flown is not None
        if flown.phase is Phase.HOLD:
            assert flown.velocity == pytest.approx(
                np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
                abs=1e-9,
            )
            return
    pytest.fail("the visit never reached the object")


def test_a_planned_visit_descends_onto_where_the_object_will_be() -> None:
    """AC-MOVE-40: a planned visit descends onto where the object will be.

    Not onto where it was. The belt carries the object the whole time the
    arm is travelling, so the pick pose is the marker's pose carried forward
    by the interception and the descent together.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    only = candidate(1, x=0.30)
    arm.step(queue_of(only), PARK, 0.0)
    at_seconds, pick = 0.0, None
    while at_seconds < 8.0:
        at_seconds += 0.01
        flown = arm.tick(at_seconds, PARK)
        assert flown is not None
        if flown.phase is Phase.HOLD:
            pick, elapsed = flown.position, at_seconds
            break
    assert pick is not None
    assert pick[2] == pytest.approx(only.flange[2], abs=1e-3)
    assert pick[0] == pytest.approx(only.flange[0] + BELT_SPEED * elapsed, abs=5e-3)


def test_an_object_with_no_interception_is_missed_rather_than_chased() -> None:
    """AC-MOVE-41: an object with no interception is missed rather than chased.

    A candidate about to leave the belt cannot be reached in the time it has
    left, and the arm gives it up: chasing it costs the objects behind it.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    leaving = Candidate(
        track_id=7,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.0,
        distance_before_leaving=0.01,
    )
    goal = arm.step(queue_of(leaving), PARK, 0.0)
    assert arm.missed == (7,)
    assert arm.active is False
    assert goal.position == pytest.approx(PARK)
    # And it is not offered again on the next capture.
    assert arm.step(queue_of(leaving), PARK, 0.5).track_id is None


def test_a_missed_head_yields_to_the_next_candidate() -> None:
    """AC-BEHAVE-01: a head with no interception yields to the next candidate."""
    arm = machine(profile=Profile.FULL_VISIT)
    hopeless = Candidate(
        track_id=7,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.0,
        distance_before_leaving=0.01,
    )
    reachable = candidate(3, x=0.30)
    goal = arm.step(queue_of(hopeless, reachable), PARK, 0.0)
    assert arm.missed == (7,)
    assert arm.active is True
    assert arm.plan is not None
    assert arm.plan.track_id == 3
    assert goal.track_id == 3


def test_a_plan_owns_the_arm_until_it_runs_out() -> None:
    """AC-MOVE-42: a plan owns the arm until it runs out.

    Re-deciding mid-visit turns an interception into a chase: each fresh arc
    is solved against a fresh estimate and the arm never arrives.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    first = candidate(1, x=0.30)
    arm.step(queue_of(first), PARK, 0.0)
    arm.tick(0.01, PARK)
    nearer = candidate(2, x=0.40)
    goal = arm.step(queue_of(nearer, first), PARK, 0.5)
    assert goal.track_id == 1
    assert arm.active is True


def test_a_refusal_tears_up_the_plan_it_was_built_on() -> None:
    """AC-MOVE-43: a refusal tears up the plan it was built on.

    Every arc after the refused pose was built on it, so continuing to fly
    them would command the rest of a sequence whose premise the solver has
    already rejected.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    only = candidate(1, x=0.30)
    arm.step(queue_of(only), PARK, 0.0)
    assert arm.active is True
    arm.step(queue_of(only), PARK, 0.5, refusal="outside the annulus")
    assert arm.active is False
    assert arm.tick(0.6, PARK) is None
    assert arm.faults == ((1, "outside the annulus"),)


def test_the_full_visit_profile_refuses_to_run_without_ceilings() -> None:
    """AC-MOVE-44: the full-visit profile refuses to run without ceilings."""
    with pytest.raises(TaskError, match="ceilings"):
        TaskMachine(
            task_settings(profile=Profile.FULL_VISIT),
            CalibrationSettings(
                flange_offset=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
            ),
            belt_surface=BELT_SURFACE,
            belt_speed=BELT_SPEED,
        )


def test_the_full_visit_profile_refuses_to_run_on_a_stopped_belt() -> None:
    """AC-MOVE-44: the full-visit profile refuses to run on a stopped belt."""
    with pytest.raises(TaskError, match="carries nothing to intercept"):
        TaskMachine(
            task_settings(profile=Profile.FULL_VISIT),
            CalibrationSettings(
                flange_offset=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
            ),
            belt_surface=BELT_SURFACE,
            motion=LIMITS,
        )


def test_the_motion_only_profile_never_descends() -> None:
    """AC-MOVE-18: the motion-only profile never descends."""
    arm = machine()
    only = candidate(1, x=0.30)
    goal = arm.step(queue_of(only), PARK, 0.0)
    for tick in range(20):
        goal = arm.step(queue_of(only), goal.position, 0.2 * (tick + 1))
        assert goal.phase in {Phase.TRACK, Phase.PARK, Phase.STANDBY}


def test_a_completed_visit_records_how_near_the_arm_got() -> None:
    """AC-MOVE-12: a completed visit records how near the arm got."""
    arm = machine()
    only = candidate(1, x=0.30)
    here = np.asarray((0.30, 0.0, BELT_SURFACE + 0.220), dtype=np.float64)
    arm.step(queue_of(only), PARK, 0.0)
    arm.step(queue_of(only), here, 0.1)
    arm.step(queue_of(only), here, 0.5)
    assert arm.served == (1,)
    assert len(arm.arrivals) == 1
    assert arm.arrivals[0] == pytest.approx(0.0, abs=1e-9)


def test_a_settling_candidate_is_served_at_the_marker_pose() -> None:
    """AC-MOVE-47: a settling candidate is served at the marker pose.

    The candidate carries the object's own velocity, and an object settling on
    the belt has a downward component in it. The plan predicts the belt's axes
    and not that one, so the pose it descends to is the pose the marker asked
    for rather than wherever gravity was taking the object.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    settling = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, -0.40), dtype=np.float64),
    )
    arm.step(queue_of(settling), PARK, 0.0)
    assert arm.plan is not None
    descend = next(leg for leg in arm.plan.legs if leg.phase is Phase.DESCEND)
    assert descend.segment.end.position[2] == pytest.approx(settling.flange[2])


def test_the_shipped_configuration_builds_a_machine() -> None:
    """The shipped configuration builds a machine."""
    from clave.world.config import load

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    world = ROOT / "configs" / "world" / "sorting_line.yml"
    arm = TaskMachine(
        settings.task,
        settings.calibration,
        belt_surface=BELT_SURFACE,
        belt_speed=BELT_SPEED,
        motion=settings.motion,
        effector=Effector.load(load(world)),
    )
    assert arm.step(queue_of(), PARK, 0.0).phase in {Phase.PARK, Phase.STANDBY}


def test_the_full_visit_profile_needs_the_jaw_geometry() -> None:
    """AC-MOVE-50: the full-visit profile needs the jaw's geometry.

    It descends onto a belt, and the pose it descends to is the flange rather
    than the jaw, so without the jaw's own dimensions the machine cannot know
    how close the pads come to the surface. Refusing at construction is the
    same answer the profile gives for missing ceilings.
    """
    with pytest.raises(TaskError, match="jaw's geometry"):
        TaskMachine(
            task_settings(profile=Profile.FULL_VISIT),
            CalibrationSettings(
                flange_offset=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
            ),
            belt_surface=BELT_SURFACE,
            belt_speed=BELT_SPEED,
            motion=LIMITS,
        )


def test_a_grasp_the_shut_jaw_can_clear_does_not_rise() -> None:
    """AC-MOVE-67: a grasp the shut jaw can clear does not rise.

    The pads drop as the jaw shuts. Lifting the flange on a grasp that
    already clears the belt lifts the pads off the object.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    arm.step(queue_of(candidate(1, x=0.30)), PARK, 0.0)
    assert arm.plan is not None
    holds = [leg for leg in arm.plan.legs if leg.phase is Phase.HOLD]
    assert len(holds) == 1
    assert holds[0].segment.end.position[2] == pytest.approx(
        holds[0].segment.start.position[2]
    )


def test_a_grasp_below_the_shut_floor_rises_back_to_it() -> None:
    """AC-MOVE-67: a grasp below the shut floor rises back to it.

    Arriving at the open jaw and staying there puts the pads through the
    clearance as the linkage hangs further. The hold climbs that difference
    and no more.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    low = BELT_SURFACE + EFFECTOR.open_flange_floor
    short = Candidate(
        track_id=3,
        anchor=np.asarray((0.30, 0.0, low - EFFECTOR.finger_length), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, low), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
    )
    arm.step(queue_of(short), PARK, 0.0)
    assert arm.plan is not None
    holds = [leg for leg in arm.plan.legs if leg.phase is Phase.HOLD]
    rise, carry = (leg.segment for leg in holds)
    assert rise.duration == pytest.approx(0.20)
    climbed = rise.end.position[2] - rise.start.position[2]
    assert climbed == pytest.approx(EFFECTOR.flange_floor - EFFECTOR.open_flange_floor)
    assert carry.end.position[2] == pytest.approx(rise.end.position[2])


def test_a_grasp_pose_below_the_jaw_clearance_is_refused() -> None:
    """AC-MOVE-50: a grasp pose below the jaw clearance is refused.

    The candidate's pose puts the pads 20 mm under the belt surface, which no
    marker would grant and which a plan that drifted can still ask for. It is
    recorded against the candidate and no plan is built from it.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    under = Candidate(
        track_id=9,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, BELT_SURFACE + 0.010), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
    )
    goal = arm.step(queue_of(under), PARK, 0.0)
    assert arm.active is False
    assert arm.missed == (9,)
    assert goal.position == pytest.approx(PARK)


def test_a_grasp_pose_at_the_jaw_clearance_is_taken() -> None:
    """AC-MOVE-50: a grasp pose exactly at the clearance is taken.

    The marker clamps its plane to this height, so a guard that refused the
    boundary would refuse every marker for the shortest object in the set.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    floor = BELT_SURFACE + EFFECTOR.open_flange_floor
    shortest = Candidate(
        track_id=3,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, floor), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
    )
    arm.step(queue_of(shortest), PARK, 0.0)
    assert arm.active is True
    assert arm.missed == ()


def test_a_refusal_with_nothing_queued_is_still_recorded() -> None:
    """AC-MOVE-05: a refusal with nothing queued is still recorded.

    A refusal nobody counted is the one way the arm can stop working and
    leave no trace of why, so it goes down against no track rather than
    being dropped.
    """
    arm = machine()
    goal = arm.step(queue_of(), PARK, 0.0, refusal="the envelope said no")
    assert goal.phase is Phase.FAULT
    assert arm.faults == ((None, "the envelope said no"),)


def test_a_plan_is_re_aimed_while_the_arm_is_still_approaching() -> None:
    """AC-MOVE-40: a plan is re-aimed while the arm is still approaching.

    The arrival time does not move with it, so everything downstream stays
    scheduled against the same instant.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    first = candidate(1, x=0.30)
    arm.step(queue_of(first), PARK, 0.0)
    before = arm.step(queue_of(first), PARK, 0.01)
    drifted = Candidate(
        track_id=1,
        anchor=np.asarray((0.34, 0.05, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.34, 0.05, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
    )
    arm.step(queue_of(drifted), PARK, 0.50)
    arm.tick(0.50, PARK)
    assert arm.active is True
    after = arm.step(queue_of(drifted), PARK, 0.51)
    assert math.dist(before.position, after.position) > 0.010


def test_a_pick_outside_the_trusted_region_is_never_planned() -> None:
    """AC-MOVE-43: a pick outside the trusted region is never planned.

    The arm is where it should be and the grasp pose is not, which is the
    candidate's problem and is recorded against it. Contrast with the test
    below, where the arm itself is out of place and the candidate is fine.
    """
    arm = TaskMachine(
        task_settings(profile=Profile.FULL_VISIT),
        CalibrationSettings(
            flange_offset=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        ),
        belt_surface=BELT_SURFACE,
        belt_speed=BELT_SPEED,
        motion=LIMITS,
        effector=EFFECTOR,
        admits=lambda pose: same(pose, PARK),
    )
    only = candidate(1, x=0.30)
    assert arm.step(queue_of(only), PARK, 0.0).position == pytest.approx(PARK)
    assert arm.active is False
    assert arm.missed == (1,)


def test_no_plan_is_built_from_a_pose_the_arm_should_not_be_in() -> None:
    """AC-MOVE-43: no plan is built from a pose the arm should not be in.

    A plan begins where the arm is, so planning from outside the trusted
    region produces a first pose the servo refuses, and the refusal tears up
    the plan that caused it. The next capture builds the same one again.
    Measured on the shipped line that latched: one excursion became 46
    faults and the arm never moved again.
    """
    outside = np.asarray((0.45, -1.00, 1.48), dtype=np.float64)
    arm = TaskMachine(
        task_settings(profile=Profile.FULL_VISIT),
        CalibrationSettings(
            flange_offset=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        ),
        belt_surface=BELT_SURFACE,
        belt_speed=BELT_SPEED,
        motion=LIMITS,
        effector=EFFECTOR,
        admits=lambda pose: not np.array_equal(pose, outside),
    )
    goal = arm.step(queue_of(candidate(1, x=0.30)), outside, 0.0)
    assert arm.active is False
    assert goal.position == pytest.approx(PARK), "the arm was not sent home to recover"
    # And the candidate is not blamed: nothing is wrong with it.
    assert arm.missed == ()


def test_task_machine_accepts_custom_belt_parameters() -> None:
    """AC-MOVE-43: TaskMachine respects custom belt dimensions and border position."""
    arm = TaskMachine(
        task_settings(profile=Profile.FULL_VISIT),
        CalibrationSettings(
            flange_offset=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        ),
        belt_surface=BELT_SURFACE,
        belt_speed=BELT_SPEED,
        motion=LIMITS,
        effector=EFFECTOR,
        belt_width=0.70,
        belt_center_y=0.05,
    )
    # Expected border: 0.05 - 0.70 / 2.0 = -0.30
    assert arm._belt_border_y == pytest.approx(-0.30)


def test_reaim_refreshes_plan_yaw_in_flight() -> None:
    """AC-MOVE-40: re-aiming an approach updates target yaw and visit-tick commands."""
    arm = machine(profile=Profile.FULL_VISIT)
    first = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.0,
        distance_before_leaving=1.5,
    )
    arm.step(queue_of(first), PARK, 0.0)
    initial_flight = arm.tick(0.01, PARK)
    assert initial_flight is not None

    rotated = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 4.0,
        distance_before_leaving=1.5,
    )
    arm.step(queue_of(rotated), PARK, 0.20)
    assert arm._plan_yaw == pytest.approx(math.pi / 4.0)
    later_flight = arm.tick(0.20, PARK)
    assert later_flight is not None
    assert later_flight.yaw is not None


def test_a_plan_is_never_flowed_onto_an_object_that_has_moved_away() -> None:
    """AC-MOVE-56: a re-aim that cannot follow the object is answered.

    The regression this exists for: a plan committed to intercepting a moving
    object, and an object that then fails to arrive there -- one dragging on
    the belt, or one that has stopped drifting across it. The refinement is
    refused in that case, because the arm is already committed further
    downstream than the object will reach and no arc takes it back up the
    belt. The machine used to hold the stale plan and fly it, and the arm
    descended onto bare belt while reporting a millimetre arrival error
    against its own commands.

    Whatever it does instead has to leave the plan aimed at the object: either
    a active plan whose aim is within the tolerance of the freshest
    estimate, or no plan at all with the visit recorded as given up.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    riding = Candidate(
        track_id=1,
        anchor=np.asarray((0.10, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.10, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
    )
    arm.step(queue_of(riding), PARK, 0.0)
    assert arm.active is True
    # Two and a half seconds later the object is barely past where it was,
    # against a plan that aimed it a belt's travel further downstream.
    dragging = Candidate(
        track_id=1,
        anchor=np.asarray((0.16, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.16, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((0.02, 0.0, 0.0), dtype=np.float64),
    )
    arm.step(queue_of(dragging), PARK, 2.50)
    assert max(refresh.drift or 0.0 for refresh in arm.refreshes) > 0.030, (
        "the object did not drift far enough from the aim to test anything"
    )
    if arm.active:
        plan = arm.plan
        assert plan is not None
        fresh = arm._fresh_aim(dragging, dragging.flange, 2.50)
        aim = next(
            leg.segment.end.position for leg in plan.legs if leg.phase is Phase.DESCEND
        )
        assert math.dist(fresh, aim) <= arm._settings.aim_tolerance
    else:
        given_up = [track for track, _ in arm.abandoned]
        assert given_up == [1]
        assert "mm out" in arm.abandoned[0][1]
        assert 1 in arm.missed
        assert arm.plan is None


def test_a_re_solved_visit_starts_from_the_motion_reference() -> None:
    """AC-MOVE-68: a mid-visit re-solve begins at the planned pose, not the flange.

    Motion is model-based in the flat output. The planned end-effector state
    advances as if tracking were perfect, and the plant's lag is a control
    problem. Re-solving from the measured flange writes that lag into the next
    plan: the projected trajectory rushes off the arc the jaw was already on,
    which is the defect this guards.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    riding = Candidate(
        track_id=1,
        anchor=np.asarray((0.10, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.10, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
    )
    arm.step(queue_of(riding), PARK, 0.0)
    assert arm.active is True
    at = 1.00
    guided = arm.tick(at, PARK)
    assert guided is not None
    reference_pose = guided.position
    # The flange has lagged sideways off the path. The object has also jumped
    # across the belt far enough that the approach arc cannot be bent onto it,
    # so the visit is solved again -- and that solve must not start at the lag.
    lagged = np.asarray(
        (reference_pose[0], reference_pose[1] - 0.35, reference_pose[2]),
        dtype=np.float64,
    )
    jumped = Candidate(
        track_id=1,
        anchor=np.asarray((0.10, 0.45, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.10, 0.45, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
    )
    arm.step(queue_of(jumped), lagged, at)
    solved = [refresh for refresh in arm.refreshes if refresh.action == "solved"]
    assert solved, (
        "the object did not force a re-solve; refreshes were "
        f"{[(r.action, r.drift) for r in arm.refreshes]}"
    )
    plan = arm.plan
    assert plan is not None
    start = plan.legs[0].segment.start.position
    assert math.dist(start, reference_pose) < 0.02, (
        f"re-solve began at {start}, {math.dist(start, reference_pose) * 1000:.0f} mm "
        f"from the motion reference {reference_pose}, so the plant lag was written "
        "into the plan"
    )
    assert math.dist(start, lagged) > 0.20, (
        f"re-solve began at the lagged flange {lagged} rather than at "
        "the motion reference"
    )


def test_a_served_object_that_leaves_the_queue_abandons_the_visit() -> None:
    """AC-MOVE-57: a visit to an object that is no longer a candidate is given up.

    A track that retired, left the window, or was replaced in its slot has no
    pose left to aim at, and a plan active at the last pose it saw is a plan
    descending onto belt. The arrival time cannot be kept either: there is
    nothing scheduled against it any more.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    arm.step(queue_of(candidate(1, x=0.30)), PARK, 0.0)
    assert arm.active is True
    arm.step(queue_of(), PARK, 0.50)
    assert arm.active is False
    assert arm.served == ()
    assert arm.abandoned == ((1, "the object is no longer a candidate"),)
    assert arm.missed == (1,)
    # And the arm is handed something to do rather than left running an active plan
    # nobody is aiming any more.
    goal = arm.step(queue_of(), PARK, 0.51)
    assert goal.phase in (Phase.STANDBY, Phase.PARK)


def test_the_tool_is_turned_to_where_the_object_will_be_facing() -> None:
    """AC-MOVE-62: the tool is turned to the yaw the object will hold at the pick.

    A grasp pose is a yaw claim made once per capture, and the jaws close when
    the plan says they will. Objects on this belt turn while that time passes --
    up to 5.4 radians per second, measured -- so commanding the claimed yaw
    sends the tool to where the object was facing: 22 to 40 degrees off its own
    axis at the instant the jaws shut, over nine grabs.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    turning = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.20,
        distance_before_leaving=1.5,
        yaw_rate_belt=0.10,
    )
    arm.step(queue_of(turning), PARK, 0.0)
    plan = arm.plan
    assert plan is not None
    assert arm._plan_yaw == pytest.approx(
        (0.20 + 0.10 * plan.pick_at + math.pi) % (2 * math.pi) - math.pi, abs=1e-9
    )
    # Past the 45 degrees a jaw's symmetry leaves useful, the claim is
    # commanded instead: a spin carried over a four second plan is not an
    # estimate of where the object will be facing, and the arm that acted on
    # one put its jaw 106.6 mm inside the belt over 67 ticks.
    spinning = Candidate(
        track_id=3,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.20,
        distance_before_leaving=1.5,
        yaw_rate_belt=5.40,
    )
    fast = machine(profile=Profile.FULL_VISIT)
    fast.step(queue_of(spinning), PARK, 0.0)
    assert fast._plan_yaw == pytest.approx(0.20)
    # And a marker claiming no rate is turned where it claimed, exactly as it
    # was before: the claim is all there is to go on.
    still = Candidate(
        track_id=2,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.20,
        distance_before_leaving=1.5,
    )
    other = machine(profile=Profile.FULL_VISIT)
    other.step(queue_of(still), PARK, 0.0)
    assert other._plan_yaw == pytest.approx(0.20)


def _into_the_descent(arm: TaskMachine) -> float:
    """Advance the clock to a tick on the descent and return that time."""
    plan = arm.plan
    assert plan is not None
    edges = plan._boundaries()
    index = next(i for i, leg in enumerate(plan.legs) if leg.phase is Phase.DESCEND)
    at = plan.started_at + edges[index] + 0.05
    flown = arm.tick(at, PARK)
    assert flown is not None
    assert flown.phase is Phase.DESCEND
    return at


def test_a_descent_moves_onto_where_the_object_is() -> None:
    """AC-MOVE-63: a descent in progress is aimed at the freshest estimate.

    The approach was the last time the aim could be rebuilt as a visit. After
    it, the object is still moving and the jaw has not shut, and a descent
    frozen at its start arrives where the object was. On a ground-truth run
    that staleness was centimetres of jaw gap with the flange already on its
    command.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    riding = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.20,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
    )
    arm.step(queue_of(riding), PARK, 0.0)
    at = _into_the_descent(arm)
    plan = arm.plan
    assert plan is not None
    before = next(
        leg.segment.end.position for leg in plan.legs if leg.phase is Phase.DESCEND
    )
    carried = 0.30 + BELT_SPEED * at
    shifted = Candidate(
        track_id=1,
        anchor=np.asarray((carried, 0.020, PINCH_Z), dtype=np.float64),
        flange=np.asarray((carried, 0.020, PICK_Z), dtype=np.float64),
        closing_axis=0.40,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
    )
    arm._reaim(
        queue_of(shifted),
        PARK,
        np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
        at,
    )
    assert arm.active is True
    assert arm.abandoned == ()
    steered = arm.plan
    assert steered is not None
    after = next(
        leg.segment.end.position for leg in steered.legs if leg.phase is Phase.DESCEND
    )
    assert after[1] == pytest.approx(0.020, abs=1e-6)
    assert abs(after[1] - before[1]) > 0.010
    assert steered.pick_at == pytest.approx(plan.pick_at, abs=1e-9)
    assert arm._plan_yaw == pytest.approx(0.40)


def test_a_descent_is_not_given_up_when_the_correction_will_not_fit() -> None:
    """AC-MOVE-63: a descent that cannot be bent is kept, not abandoned.

    Solving the visit again from mid-descent sends an arm that is already
    coming down back up the belt, or takes the visit away a fraction of a
    second before the jaw shuts. A miss a fraction of which still fits is
    taken that far, which is `AC-MOVE-65`. This miss is past every fraction
    the speed ceiling allows, so the plan already in hand is the one that
    flies, and the visit is not abandoned.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    riding = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
    )
    arm.step(queue_of(riding), PARK, 0.0)
    at = _into_the_descent(arm)
    plan = arm.plan
    wild = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 10.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 10.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        velocity_world=np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64),
    )
    arm._reaim(
        queue_of(wild), PARK, np.asarray((BELT_SPEED, 0.0, 0.0), dtype=np.float64), at
    )
    assert arm.active is True
    assert arm.abandoned == ()
    assert arm.plan is plan


def test_reaim_ignores_calls_once_the_jaw_is_shut() -> None:
    """_reaim does not modify the plan once the jaw is shut.

    The descent is still corrected, because the object is moving and the jaw
    has not closed. The hold is not: a closed jaw that keeps chasing pulls
    whatever it caught across the belt.
    """
    arm = machine(profile=Profile.FULL_VISIT)
    cand = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.20,
        distance_before_leaving=1.5,
    )
    arm.step(queue_of(cand), PARK, 0.0)
    assert arm._plan_yaw == pytest.approx(0.20)

    arm._phase = Phase.HOLD
    changed = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.80,
        distance_before_leaving=1.5,
    )
    arm._reaim(
        queue_of(changed), PARK, np.asarray((0.0, 0.0, 0.0), dtype=np.float64), 0.50
    )
    assert arm._plan_yaw == pytest.approx(0.20)


def test_rest_preserves_last_commanded_yaw() -> None:
    """_rest sets target_yaw_world to last known yaw rather than None."""
    arm = machine(profile=Profile.FULL_VISIT)
    arm._last_yaw = 0.75
    goal = arm._rest(PARK, at_nanos=1_000_000)
    assert goal.target_yaw_world == pytest.approx(0.75)


def test_the_task_machine_narrates_the_visit() -> None:
    """AC-STORY-03: commit, phases, a re-aim, a miss, a fault, and giving up."""
    lines: list[str] = []
    story = StoryLog(sink=lines.append)
    story.note(1, "M-01", "CH-PET")
    story.note(7, "M-11", "CH-REJECT")
    arm = machine(
        profile=Profile.FULL_VISIT,
        story=story,
        chutes={"CH-PET": np.asarray((0.20, -0.80, BELT_SURFACE), dtype=np.float64)},
    )
    only = Candidate(
        track_id=1,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        channel="CH-PET",
    )
    arm.step(queue_of(only), PARK, 0.0)
    assert any(
        "I will fly a" in line and "object 1 (PET, M-01)" in line for line in lines
    )
    assert any("track, descend, hold, retreat, deliver" in line for line in lines)

    carried = Candidate(
        track_id=1,
        anchor=np.asarray((0.30 + BELT_SPEED * 0.20, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30 + BELT_SPEED * 0.20, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=math.pi / 2.0,
        distance_before_leaving=1.5,
        channel="CH-PET",
    )
    arm.tick(0.05, PARK)
    arm.step(queue_of(carried), PARK, 0.20)
    assert any("fresh estimate" in line for line in lines)
    # AC-STORY-10: the same capture repeated is another refresh, not another line.
    for _ in range(4):
        arm.step(queue_of(carried), PARK, 0.20)
    decisions: list[tuple[int, str]] = []
    for refresh in arm.refreshes:
        decision = (refresh.track_id, refresh.action)
        if not decisions or decisions[-1] != decision:
            decisions.append(decision)
    assert sum("fresh estimate" in line for line in lines) == len(decisions)
    assert len(arm.refreshes) > len(decisions)

    seen: set[Phase] = set()
    at_seconds = 0.20
    while at_seconds < 8.0 and Phase.HOLD not in seen:
        at_seconds += 0.01
        flown = arm.tick(at_seconds, PARK)
        assert flown is not None
        seen.add(flown.phase)
    assert Phase.DESCEND in seen
    assert any("descend onto object 1" in line for line in lines)
    assert any("close the jaw on object 1" in line for line in lines)

    story.grasp_reading(at_seconds, 1, 0.080)
    while at_seconds < 8.0 and not any("fly the release" in line for line in lines):
        at_seconds += 0.01
        flown = arm.tick(at_seconds, PARK)
        if flown is None:
            break
    released = [line for line in lines if "fly the release of object 1" in line]
    assert released
    assert released[-1].endswith(": fail")

    given_lines: list[str] = []
    given_story = StoryLog(sink=given_lines.append)
    given_story.note(1, "M-01", "CH-PET")
    giving_up = machine(profile=Profile.FULL_VISIT, story=given_story)
    giving_up.step(queue_of(only), PARK, 0.0)
    giving_up.tick(0.05, PARK)
    giving_up.step(queue_of(), PARK, 0.20)
    assert any(
        "abandon object 1" in line and line.endswith(": fail") for line in given_lines
    )

    missed = machine(profile=Profile.FULL_VISIT, story=story)
    leaving = Candidate(
        track_id=7,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=0.0,
        distance_before_leaving=0.01,
        channel="CH-REJECT",
    )
    missed.step(queue_of(leaving), PARK, 1.0)
    assert any("skip object 7" in line and line.endswith(": fail") for line in lines)

    faulted = machine(story=story)
    faulted.step(queue_of(only), PARK, 2.0, refusal="outside the annulus")
    assert any(
        "refused (outside the annulus)" in line and "hold the arm" in line
        for line in lines
    )


def test_a_motion_only_visit_narrates_the_track_and_the_arrival() -> None:
    """AC-STORY-03: motion-only says it is tracking, and success is arriving."""
    lines: list[str] = []
    story = StoryLog(sink=lines.append)
    story.note(7, "M-08", "CH-FIBER")
    arm = machine(story=story)
    head = Candidate(
        track_id=7,
        anchor=np.asarray((0.30, 0.0, PINCH_Z), dtype=np.float64),
        flange=np.asarray((0.30, 0.0, PICK_Z), dtype=np.float64),
        closing_axis=None,
        distance_before_leaving=1.5,
        channel="CH-FIBER",
    )
    goal = arm.step(queue_of(head), PARK, 0.0)
    assert any("at approach height" in line and "object 7" in line for line in lines)
    arm.step(queue_of(head), goal.position, 0.0)
    arm.step(queue_of(head), goal.position, 0.31)
    assert any(
        "count the visit to object 7" in line and line.endswith(": success")
        for line in lines
    )
