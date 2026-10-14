"""The pose to command this tick, on a bounded path."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.control.motion import Reference, toward
from clave.control.settings import MotionSettings, Phase
from clave.control.task import Goal
from clave.world.arm import ReachBounds
from clave.world.config import load

TIMESTEP = 0.002
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
_ARM = load(
    Path(__file__).resolve().parents[2] / "configs" / "world" / "sorting_line.yml"
)["arm"]
BOUNDS = ReachBounds(
    float(_ARM["reach_min_meters"]),
    float(_ARM["reach_max_meters"]),
    (
        float(_ARM["tool_above_base_meters"][0]),
        float(_ARM["tool_above_base_meters"][1]),
    ),
)
BASE_XY = (0.0, -0.70)


def goal_at(
    x: float,
    y: float = 0.0,
    z: float = 1.12,
    yaw: float | None = 0.5,
    rides_belt: bool = False,
    observed_at_nanos: int = 0,
) -> Goal:
    """A tracking goal at one place."""
    return Goal(
        phase=Phase.TRACK,
        position=np.asarray((x, y, z), dtype=np.float64),
        yaw=yaw,
        track_id=1,
        observed_at_nanos=observed_at_nanos,
        rides_belt=rides_belt,
    )


def at(
    position: tuple[float, float, float] | NDArray[np.float64],
    speed: float = 0.0,
) -> Reference:
    """The reference, standing somewhere at some speed."""
    return Reference(position=np.asarray(position, dtype=np.float64), speed=speed)


def run(
    start: tuple[float, float, float] | NDArray[np.float64],
    goal: Goal,
    ticks: int,
    belt_speed: float = 0.0,
    limits: MotionSettings = LIMITS,
) -> list[Reference]:
    """Step the reference toward a goal and keep every state it passed."""
    motion = at(start)
    history = [motion]
    for _ in range(ticks):
        command = toward(motion, goal, TIMESTEP, limits, belt_speed, 0)
        motion = Reference(position=command.position, speed=command.speed)
        history.append(motion)
    return history


def test_a_step_never_exceeds_the_speed_ceiling() -> None:
    """AC-MOVE-09: a step never exceeds the speed ceiling."""
    for motion in run((0.0, 0.0, 1.12), goal_at(2.0), ticks=3000):
        assert motion.speed <= LIMITS.max_speed + 1e-9


def test_the_ceiling_is_respected_over_a_whole_traverse() -> None:
    """AC-MOVE-09: the ceiling is respected over a whole traverse.

    Swept rather than checked once, because a limiter that holds for the
    first step and not the last is a limiter that does not hold.
    """
    history = run((0.0, 0.0, 1.12), goal_at(1.30, y=0.40), ticks=4000)
    for before, after in zip(history[:-1], history[1:], strict=True):
        assert math.dist(before.position, after.position) <= (
            LIMITS.max_speed * TIMESTEP + 1e-9
        )
    assert math.dist(history[-1].position, (1.30, 0.40, 1.12)) < 1e-6


def test_the_acceleration_ceiling_holds_over_a_whole_traverse() -> None:
    """AC-MOVE-09: the acceleration ceiling holds over a whole traverse.

    Etapa A asked for full speed on the first tick from a standstill, which
    is a step in velocity the actuators answer with a lunge. Every change of
    speed now fits inside the configured acceleration.
    """
    history = run((0.0, 0.0, 1.12), goal_at(1.30, y=0.40), ticks=4000)
    for before, after in zip(history[:-1], history[1:], strict=True):
        assert abs(after.speed - before.speed) <= (
            LIMITS.max_acceleration * TIMESTEP + 1e-9
        )


def test_the_reference_starts_from_rest_and_ends_at_rest() -> None:
    """AC-MOVE-09: the reference starts from rest and ends at rest.

    A profile that arrives at speed has not arrived: it overshoots and comes
    back, which reads on a plot as an arm that cannot settle.
    """
    history = run((0.0, 0.0, 1.12), goal_at(0.80), ticks=4000)
    assert history[0].speed == pytest.approx(0.0)
    assert history[-1].speed == pytest.approx(0.0, abs=1e-6)
    assert max(state.speed for state in history) > 0.5 * LIMITS.max_speed


def test_a_tighter_acceleration_takes_longer_to_arrive() -> None:
    """A tighter acceleration takes longer to arrive.

    The bound is a physical limit rather than a decoration, so halving it has
    to cost time.
    """
    goal = goal_at(1.00)
    brisk = run((0.0, 0.0, 1.12), goal, ticks=4000, limits=LIMITS)
    gentle = run(
        (0.0, 0.0, 1.12),
        goal,
        ticks=4000,
        limits=MotionSettings(
            max_speed=LIMITS.max_speed,
            max_acceleration=0.5,
            intercept_passes=LIMITS.intercept_passes,
            minimum_segment_seconds=LIMITS.minimum_segment_seconds,
            segment_sample_count=LIMITS.segment_sample_count,
            bisection_passes=LIMITS.bisection_passes,
            minimum_delivery_seconds=0.05,
            correction_steps=32,
        ),
    )

    def ticks_to_arrive(history: list[Reference]) -> int:
        for index, state in enumerate(history):
            if math.dist(state.position, goal.position) < 1e-6:
                return index
        return len(history)

    assert ticks_to_arrive(gentle) > ticks_to_arrive(brisk)


def test_a_near_goal_is_approached_and_never_overshot() -> None:
    """A near goal is approached and never overshot.

    Under an acceleration bound a reference at rest cannot arrive in one
    tick, so the property is not that it jumps there. It is that it closes
    the gap monotonically and never passes the goal, because stepping a fixed
    distance past a near target is what makes a flange buzz around something
    it has already met.
    """
    near = goal_at(0.0005)
    history = run((0.0, 0.0, 1.12), near, ticks=400)
    gaps = [math.dist(state.position, near.position) for state in history]
    for before, after in zip(gaps[:-1], gaps[1:], strict=True):
        assert after <= before + 1e-12
    assert gaps[-1] < 1e-9
    assert max(state.position[0] for state in history) <= near.position[0] + 1e-12


def test_the_step_runs_straight_at_the_goal() -> None:
    """The step runs straight at the goal.

    A straight line in task space is what an industrial linear move is, and
    it is what makes a descent a descent rather than an arc.
    """
    flange = np.asarray((0.0, 0.0, 1.12), dtype=np.float64)
    goal = goal_at(0.60, y=0.80, z=1.12)
    command = toward(at(flange, speed=LIMITS.max_speed), goal, TIMESTEP, LIMITS, 0.0, 0)
    travelled = [command.position[axis] - flange[axis] for axis in range(3)]
    remaining = [goal.position[axis] - flange[axis] for axis in range(3)]
    scale = math.dist((0, 0, 0), travelled) / math.dist((0, 0, 0), remaining)
    for moved, wanted in zip(travelled, remaining, strict=True):
        assert moved == pytest.approx(wanted * scale)


def test_the_yaw_the_goal_asked_for_is_carried_through() -> None:
    """AC-MOVE-07: the yaw the goal asked for is carried through."""
    command = toward(
        at((0.0, 0.0, 1.12)), goal_at(0.3, yaw=1.25), TIMESTEP, LIMITS, 0.0, 0
    )
    assert command.yaw == pytest.approx(1.25)


def test_a_goal_asking_for_no_rotation_commands_none() -> None:
    """AC-MOVE-08: a goal asking for no rotation commands none."""
    command = toward(
        at((0.0, 0.0, 1.12)), goal_at(0.3, yaw=None), TIMESTEP, LIMITS, 0.0, 0
    )
    assert command.yaw is None


def test_a_fault_goal_commands_no_motion() -> None:
    """AC-MOVE-05: a fault goal commands no motion.

    The task machine holds the flange where it is by asking for the pose it
    already has, and motion has to honour that rather than step toward it.
    """
    flange = np.asarray((0.11, -0.22, 1.09), dtype=np.float64)
    held = Goal(
        phase=Phase.FAULT, position=flange, yaw=None, track_id=3, observed_at_nanos=0
    )
    motion = at(flange, speed=0.4)
    for _ in range(400):
        command = toward(motion, held, TIMESTEP, LIMITS, 0.0, 0)
        assert command.position == pytest.approx(flange)
        assert command.speed <= motion.speed
        motion = Reference(position=command.position, speed=command.speed)
    assert motion.speed == pytest.approx(0.0)


def test_a_longer_tick_travels_further() -> None:
    """A longer tick travels further.

    The bound is a speed and not a step length, so the distance covered has
    to scale with the tick it was given.
    """
    flange = np.asarray((0.0, 0.0, 1.12), dtype=np.float64)
    goal = goal_at(2.0)
    moving = at(flange, speed=LIMITS.max_speed)
    short = toward(moving, goal, 0.002, LIMITS, 0.0, 0)
    long = toward(moving, goal, 0.010, LIMITS, 0.0, 0)
    assert math.dist(flange, long.position) == pytest.approx(
        5.0 * math.dist(flange, short.position)
    )


INNER = BOUNDS.reach_min


def keep_inside(pose: NDArray[np.float64]) -> NDArray[np.float64]:
    """The shipped projection, over the shipped arm base."""
    from clave.world.arm import project_into_reach

    x, y = project_into_reach(BASE_XY, float(pose[0]), float(pose[1]), BOUNDS)
    return np.asarray((x, y, float(pose[2])), dtype=np.float64)


def test_a_path_between_two_admitted_poses_stays_out_of_the_hole() -> None:
    """AC-MOVE-23: a path between two admitted poses stays out of the hole.

    The arm is trusted over an annulus, so its workspace is not convex and a
    straight line between two poses inside it can cross the hole around the
    base. Measured on the shipped line, a traverse from the park pose to the
    far side of the belt passed within 0.10 m of a base the arm is not
    trusted inside 0.25 m of, and every pose along that stretch was refused.
    """
    park = np.asarray((0.45, -1.00, 1.20), dtype=np.float64)
    across = np.asarray((-0.50, 0.42, 1.12), dtype=np.float64)
    goal = Goal(
        phase=Phase.TRACK,
        position=across,
        yaw=0.0,
        track_id=1,
        observed_at_nanos=0,
    )

    naive, guarded = at(park), at(park)
    breached = False
    for _ in range(4000):
        loose = toward(naive, goal, TIMESTEP, LIMITS, 0.0, 0)
        tight = toward(guarded, goal, TIMESTEP, LIMITS, 0.0, 0, keep_inside)
        naive = Reference(position=loose.position, speed=loose.speed)
        guarded = Reference(position=tight.position, speed=tight.speed)
        if math.dist(naive.position[:2], BASE_XY) < INNER:
            breached = True
        assert math.dist(guarded.position[:2], BASE_XY) >= INNER - 1e-9

    assert breached, "the straight line no longer crosses the hole to begin with"
    assert math.dist(guarded.position, across) < 1e-6, "the guarded path arrives"


def test_the_projection_leaves_a_pose_outside_the_hole_alone() -> None:
    """The projection leaves a pose outside the hole alone.

    It is a constraint and not a filter: everywhere the arm is trusted, the
    path is the straight line and nothing rounds it off.
    """
    outside = np.asarray((0.60, -0.20, 1.12), dtype=np.float64)
    assert keep_inside(outside) == pytest.approx(outside)


def test_the_projection_keeps_the_height_it_was_given() -> None:
    """The projection keeps the height it was given.

    The hole is a cylinder about the base, so leaving it is a horizontal
    move. Changing the height here would quietly undo a descent.
    """
    assert keep_inside(np.asarray((0.0, -0.68, 1.07), dtype=np.float64))[
        2
    ] == pytest.approx(1.07)


BELT = 0.31


def test_a_goal_that_rides_the_belt_is_aimed_ahead_of_itself() -> None:
    """AC-MOVE-10: a goal that rides the belt is aimed ahead of itself.

    Commanding where the object is aims at where it was by the time the arm
    arrives. Etapa A measured that lag at 102 mm against a 10 mm arrival
    tolerance, which is why this exists.
    """
    standing = goal_at(0.60, rides_belt=False)
    riding = goal_at(0.60, rides_belt=True)
    start = at((-0.40, 0.0, 1.12))

    still = toward(start, standing, TIMESTEP, LIMITS, BELT, 0)
    ahead = toward(start, riding, TIMESTEP, LIMITS, BELT, 0)
    assert ahead.aim is not None and still.aim is not None
    # Both take the same first step; what differs is the direction, because
    # the intercept sits downstream of the pose the goal reports.
    assert ahead.aim[0] > riding.position[0]
    assert ahead.aim[0] > still.aim[0]


def test_the_intercept_is_where_the_object_will_be_when_the_arm_gets_there() -> None:
    """AC-MOVE-10: the intercept is where the object will be on arrival.

    Checked against the definition rather than against a constant: the time
    the arm needs to reach the intercept is the same time the belt needs to
    carry the object there.
    """
    goal = goal_at(0.60, rides_belt=True)
    start = at((-0.40, 0.0, 1.12))
    aim = toward(start, goal, TIMESTEP, LIMITS, BELT, 0).aim
    assert aim is not None

    carried = (aim[0] - goal.position[0]) / BELT
    travel = math.dist(start.position, aim) / LIMITS.max_speed
    assert carried == pytest.approx(travel, rel=0.05)


def test_a_standing_goal_is_aimed_at_itself() -> None:
    """AC-MOVE-10: a standing goal is aimed at itself.

    The park pose does not ride the belt, and predicting it downstream would
    send the arm past the place it was told to rest.
    """
    park = Goal(
        phase=Phase.PARK,
        position=np.asarray((0.45, -1.00, 1.20), dtype=np.float64),
        yaw=None,
        track_id=None,
        observed_at_nanos=0,
        rides_belt=False,
    )
    aim = toward(at((0.0, 0.0, 1.12)), park, TIMESTEP, LIMITS, BELT, 0).aim
    assert aim is not None
    assert aim == pytest.approx(park.position)


def test_a_stopped_belt_intercepts_where_the_object_stands() -> None:
    """A stopped belt intercepts where the object stands.

    The degenerate case the prediction has to survive rather than divide by.
    """
    goal = goal_at(0.60, rides_belt=True)
    aim = toward(at((-0.40, 0.0, 1.12)), goal, TIMESTEP, LIMITS, 0.0, 0).aim
    assert aim is not None
    assert aim == pytest.approx(goal.position)


CAPTURE_TICKS = 250
"""Ticks between decisions, matching the run's 0.5 s capture cadence."""


def test_interception_closes_the_lag_a_chasing_command_leaves() -> None:
    """AC-MOVE-10: interception closes the lag a chasing command leaves.

    Modelled the way the loop actually runs, because that is where the lag
    comes from: a goal is decided once per capture and held for 0.5 s while
    the belt carries the object 157 mm. Refreshing the goal every tick would
    hide the effect this exists to fix.
    """
    start = np.asarray((-0.40, 0.0, 1.12), dtype=np.float64)
    origin = np.asarray((0.60, 0.0, 1.12), dtype=np.float64)

    def lag(rides_belt: bool) -> float:
        motion = at(start)
        goal = None
        for tick in range(3000):
            now = int(tick * TIMESTEP * 1_000_000_000)
            if tick % CAPTURE_TICKS == 0:
                seen = np.asarray(
                    (origin[0] + BELT * tick * TIMESTEP, origin[1], origin[2]),
                    dtype=np.float64,
                )
                goal = Goal(
                    phase=Phase.TRACK,
                    position=seen,
                    yaw=0.0,
                    track_id=1,
                    observed_at_nanos=now,
                    rides_belt=rides_belt,
                )
            assert goal is not None
            command = toward(motion, goal, TIMESTEP, LIMITS, BELT, now)
            motion = Reference(position=command.position, speed=command.speed)
            live = np.asarray(
                (origin[0] + BELT * tick * TIMESTEP, origin[1], origin[2]),
                dtype=np.float64,
            )
            if tick > CAPTURE_TICKS * 4:
                return math.dist(motion.position, live)
        raise AssertionError("the run never settled")

    chasing, intercepting = lag(False), lag(True)
    assert intercepting < 0.5 * chasing, f"{intercepting=} {chasing=}"


def test_an_intercept_beyond_reach_is_pulled_back_to_the_edge() -> None:
    """AC-MOVE-23: an intercept beyond reach is pulled back to the edge.

    A chord between two points inside a disc stays inside it, so a path
    never leaves by the outer radius. An interception does: aiming ahead of
    an object the belt is carrying puts the aim downstream of a pose that
    was reachable. Four visits in a sixteen second run were refused that way
    at 1.266 m to 1.287 m against a 1.25 m limit.
    """

    far = np.asarray((BASE_XY[0] + 1.40, BASE_XY[1], 1.12), dtype=np.float64)
    pulled = keep_inside(far)
    assert math.dist(pulled[:2], BASE_XY) <= BOUNDS.reach_max
    assert pulled[2] == pytest.approx(1.12)


def test_a_pose_inside_the_annulus_is_left_alone() -> None:
    """A pose inside the annulus is left alone.

    The projection is a constraint and not a filter: everywhere the arm is
    trusted, the path is the straight line and nothing rounds it off.
    """
    for radius in (0.30, 0.70, 1.20):
        pose = np.asarray((BASE_XY[0] + radius, BASE_XY[1], 1.12), dtype=np.float64)
        assert keep_inside(pose) == pytest.approx(pose)
