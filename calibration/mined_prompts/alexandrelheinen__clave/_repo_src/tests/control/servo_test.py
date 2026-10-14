"""Turning a commanded pose into joint angles, against the compiled model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.control.motion import Command
from clave.control.servo import follow
from clave.world.config import load

ROOT = Path(__file__).resolve().parents[2]
GAIN = 1.0


@pytest.fixture(scope="module")
def world() -> Any:
    """The shipped world, compiled once for the whole module."""
    numpy = pytest.importorskip("numpy")
    mujoco = pytest.importorskip("mujoco")
    from clave.world import arm as armmod
    from clave.world import scene

    raw = load(ROOT / "configs" / "world" / "sorting_line.yml")
    model, data, plan = scene.build(raw, numpy.random.default_rng(0), ROOT)
    mujoco.mj_forward(model, data)
    return (
        model,
        data,
        armmod.locate(
            model,
            armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
        ),
    )


def flange_of(world: Any) -> NDArray[np.float64]:
    """Where the flange stands right now."""
    from clave.world import arm as armmod

    model, data, arm = world
    del model
    place = armmod.end_effector_position(data, arm)
    return np.asarray(place, dtype=np.float64)


def rested(world: Any) -> Any:
    """Return the compiled model with a fresh arm at the configured park pose.

    The module fixture shares one `MjData` across tests, and the zero home
    pose is not how a run starts: `_park_the_arm` solves the park first. A
    short settle from zeros toward an extended IK branch can jam the arm
    into the belt, so arrival tests start from park the way the sim does.
    """
    mujoco = pytest.importorskip("mujoco")
    from clave.control.settings import ControlSettings
    from clave.world import arm as armmod

    model, _, arm = world
    data = mujoco.MjData(model)
    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    park = np.asarray(settings.task.park_position, dtype=np.float64)
    angles = armmod.solve(model, data, arm, park)
    for slot, joint in enumerate(arm.joint_ids):
        data.qpos[model.jnt_qposadr[joint]] = angles[slot]
        data.ctrl[arm.actuator_ids[slot]] = angles[slot]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    armmod._TRACKING[(id(model), id(data))] = angles.copy()
    return model, data, arm


def settle(world: Any, command: Command, steps: int = 400) -> Any:
    """Command one pose repeatedly and let the physics catch up."""
    mujoco = pytest.importorskip("mujoco")
    model, data, arm = world
    step = None
    for _ in range(steps):
        step = follow(model, data, arm, command, GAIN)
        mujoco.mj_step(model, data)
    return step


def test_a_reachable_pose_moves_the_flange_toward_it(world: Any) -> None:
    """AC-MOVE-04: a reachable pose moves the flange toward it."""
    target = np.asarray((0.30, -0.20, 1.12), dtype=np.float64)
    before = math.dist(flange_of(world), target)
    settle(world, Command(position=target, yaw=0.0))
    assert math.dist(flange_of(world), target) < before


def test_the_flange_arrives_inside_the_tolerance_the_task_layer_uses(
    world: Any,
) -> None:
    """AC-MOVE-12: the flange arrives inside the tolerance the task layer uses.

    Ten millimetres is what `task.arrival_tolerance_meters` calls arrived. A
    servo that cannot get inside it makes every visit hang, so the number and
    the mechanism are checked against each other here rather than separately.
    """
    from clave.control.settings import ControlSettings

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    target = np.asarray((0.20, -0.30, 1.15), dtype=np.float64)
    arm = rested(world)
    settle(arm, Command(position=target, yaw=0.0), steps=1200)
    assert math.dist(flange_of(arm), target) <= settings.task.arrival_tolerance


def test_a_pose_outside_the_annulus_is_refused_and_named(world: Any) -> None:
    """AC-MOVE-05: a pose outside the annulus is refused and named."""
    model, data, arm = world
    step = follow(
        model,
        data,
        arm,
        Command(position=np.asarray((3.0, 0.0, 1.10), dtype=np.float64), yaw=0.0),
        GAIN,
    )
    assert step.refusal is not None
    assert "annulus" in step.refusal


def test_a_pose_outside_the_vertical_band_is_refused_and_named(world: Any) -> None:
    """AC-MOVE-05: a pose outside the vertical band is refused and named."""
    model, data, arm = world
    step = follow(
        model,
        data,
        arm,
        Command(position=np.asarray((0.30, -0.30, 2.40), dtype=np.float64), yaw=0.0),
        GAIN,
    )
    assert step.refusal is not None
    assert "band" in step.refusal


def test_a_refused_pose_commands_a_hold_rather_than_the_pose(world: Any) -> None:
    """AC-MOVE-05: a refused pose commands a hold rather than the pose.

    Writing nothing looks safer and is not. An uncommanded arm sags under
    gravity, the sag puts the flange outside the trusted vertical band, and
    from there every pose is refused for a reason the controller caused. That
    deadlock is what this pins down: the command written is where the arm
    already is, derived from the arm and not from the pose that was refused.
    """
    from clave.world import arm as armmod

    model, data, arm = world
    step = follow(
        model,
        data,
        arm,
        Command(position=np.asarray((3.0, 0.0, 1.10), dtype=np.float64), yaw=0.0),
        GAIN,
    )
    assert step.refusal is not None
    held = armmod.joint_positions(model, data, arm)
    for slot, actuator in enumerate(arm.actuator_ids):
        assert float(data.ctrl[actuator]) == pytest.approx(held[slot])


def test_an_arm_left_refusing_does_not_sag_out_of_its_own_workspace(
    world: Any,
) -> None:
    """An arm left refusing does not sag out of its own workspace.

    The regression behind the hold. Two hundred ticks of refusal used to drop
    the flange below the trusted band, after which nothing the controller
    asked for could ever be accepted again.
    """
    mujoco = pytest.importorskip("mujoco")
    from clave.world import arm as armmod

    model, data, arm = world
    settle(
        world,
        Command(position=np.asarray((0.30, -0.30, 1.15), dtype=np.float64), yaw=0.0),
        steps=900,
    )
    for _ in range(200):
        follow(
            model,
            data,
            arm,
            Command(position=np.asarray((3.0, 0.0, 1.10), dtype=np.float64), yaw=0.0),
            GAIN,
        )
        mujoco.mj_step(model, data)
    import numpy

    assert armmod.reachable(arm, numpy.array(flange_of(world)))


def test_a_command_with_no_yaw_holds_the_rotation_the_arm_has(world: Any) -> None:
    """AC-MOVE-08: a command with no yaw holds the rotation the arm has.

    An unoriented footprint claims no angle, so the arm keeps the one it is
    already turned to rather than snapping to zero.
    """
    from clave.world import arm as armmod

    model, data, arm = world
    settle(
        world,
        Command(position=np.asarray((0.30, -0.25, 1.15), dtype=np.float64), yaw=1.0),
        steps=900,
    )
    held = armmod.tool_yaw(data, arm)
    settle(
        world,
        Command(position=np.asarray((0.32, -0.25, 1.15), dtype=np.float64), yaw=None),
        steps=300,
    )
    assert armmod.tool_yaw(data, arm) == pytest.approx(held, abs=0.15)


def test_the_commanded_joints_stay_inside_their_limits(world: Any) -> None:
    """The commanded joints stay inside their limits."""
    model, data, arm = world
    step = follow(
        model,
        data,
        arm,
        Command(position=np.asarray((0.40, -0.40, 1.05), dtype=np.float64), yaw=0.3),
        GAIN,
    )
    assert step.refusal is None
    for angle, low, high in zip(step.joints, arm.lower, arm.upper, strict=True):
        assert low <= angle <= high


def test_the_arm_keeps_up_with_a_pose_moving_at_the_speed_ceiling(
    world: Any,
) -> None:
    """AC-MOVE-09: the arm keeps up with a pose moving at the speed ceiling.

    This is the question etapa A exists to answer. The belt moves a target
    0.5 mm per physics step and the solver's warm start was sized for that;
    a flange traversing at the configured 1 m/s moves 2 mm per step, three to
    four times further. If the descent stops converging there, the answer is
    to slow the belt rather than to raise the ceiling, and this is what would
    say so.
    """
    mujoco = pytest.importorskip("mujoco")
    from clave.control.motion import Reference, toward
    from clave.control.settings import ControlSettings, Phase
    from clave.control.task import Goal

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    model, data, arm = rested(world)
    settle(
        (model, data, arm),
        Command(position=np.asarray((0.30, -0.30, 1.15), dtype=np.float64), yaw=0.0),
        steps=900,
    )

    goal = Goal(
        phase=Phase.TRACK,
        position=np.asarray((-0.40, -0.20, 1.15), dtype=np.float64),
        yaw=0.0,
        track_id=1,
        observed_at_nanos=0,
    )
    motion = Reference(position=flange_of((model, data, arm)), speed=0.0)
    lag = 0.0
    for _ in range(1500):
        command = toward(motion, goal, 0.002, settings.motion, 0.0, 0)
        motion = Reference(position=command.position, speed=command.speed)
        step = follow(model, data, arm, command, settings.servo.gain)
        assert step.refusal is None
        mujoco.mj_step(model, data)
        lag = max(lag, math.dist(flange_of((model, data, arm)), command.position))
    assert (
        math.dist(flange_of((model, data, arm)), goal.position)
        <= settings.task.arrival_tolerance
    )
    # Recorded rather than bounded tightly: what matters here is that the
    # solver kept converging and the flange arrived, not the exact tracking
    # error of a position-controlled arm under its own actuator dynamics.
    assert lag < 0.30


def test_stepping_motion_from_the_measurement_crawls(world: Any) -> None:
    """Stepping the motion reference from the measurement crawls.

    The defect this pins down is easy to write and hard to see: feeding the
    measured flange back into motion makes the reference restart from
    wherever the arm lagged to, so it never leads the plant and the traverse
    runs at the tracking error rather than at the ceiling. Keeping the test
    keeps somebody from simplifying the reference away again.
    """
    mujoco = pytest.importorskip("mujoco")
    from clave.control.motion import Reference, toward
    from clave.control.settings import ControlSettings, Phase
    from clave.control.task import Goal

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    model, data, arm = world
    settle(
        world,
        Command(position=np.asarray((0.30, -0.30, 1.15), dtype=np.float64), yaw=0.0),
        steps=900,
    )

    goal = Goal(
        phase=Phase.TRACK,
        position=np.asarray((-0.40, -0.20, 1.15), dtype=np.float64),
        yaw=0.0,
        track_id=1,
        observed_at_nanos=0,
    )
    start = flange_of(world)
    for _ in range(500):
        chasing_motion = Reference(position=flange_of(world), speed=0.0)
        command = toward(chasing_motion, goal, 0.002, settings.motion, 0.0, 0)
        follow(model, data, arm, command, settings.servo.gain)
        mujoco.mj_step(model, data)
    chasing = math.dist(start, flange_of(world))

    # One second of ticks at the ceiling covers a metre. Chasing the plant
    # covers a fraction of that, and this is the fraction that matters.
    assert chasing < 0.5 * settings.motion.max_speed * 500 * 0.002


def place(world: Any, angles: list[float]) -> None:
    """Put the arm at one joint configuration and command it to stay."""
    mujoco = pytest.importorskip("mujoco")
    model, data, arm = world
    for slot, joint in enumerate(arm.joint_ids):
        data.qpos[model.jnt_qposadr[joint]] = angles[slot]
    for slot, actuator in enumerate(arm.actuator_ids):
        data.ctrl[actuator] = angles[slot]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def near_singular(world: Any) -> list[float]:
    """Joint angles with the fifth axis at zero, where the wrist degenerates.

    A six-revolute arm loses a degree of freedom when the fourth and sixth
    axes become collinear, which is what the fifth going to zero does. These
    angles were found by sweeping the compiled model for a degenerate wrist
    whose flange still lands inside the trusted annulus, because a singular
    configuration the arm refuses on reach proves nothing about the cap.
    """
    model, data, arm = world
    del model, data, arm
    return [-0.6, -0.8, 1.2, -1.4, 0.0, 0.0]


def test_a_wrist_singularity_bounds_the_joint_command_rather_than_the_pose(
    world: Any,
) -> None:
    """AC-MOVE-11: a wrist singularity bounds the joint command, not the pose.

    Where the Jacobian loses rank the damped solve still asks for a large
    joint motion to buy a small Cartesian one. The requirement is that the
    command stays inside what the actuators turn at, and explicitly not that
    the pose is met.
    """
    from clave.control.settings import ControlSettings

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    timestep = 0.002
    cap = settings.servo.max_joint_speed * timestep

    model, data, arm = world
    place(world, near_singular(world))
    before = [float(data.ctrl[actuator]) for actuator in arm.actuator_ids]

    # A small move across the degenerate direction, which is what asks the
    # wrist for a large turn.
    here = flange_of(world)
    nudged = np.asarray((here[0] + 0.004, here[1] + 0.004, here[2]), dtype=np.float64)
    step = follow(
        model, data, arm, Command(position=nudged, yaw=0.0), 1.0, max_joint_step=cap
    )
    if step.refusal is not None:
        pytest.skip(
            f"the singular configuration is outside the workspace: {step.refusal}"
        )

    after = [float(data.ctrl[actuator]) for actuator in arm.actuator_ids]
    for was, now in zip(before, after, strict=True):
        assert abs(now - was) <= cap + 1e-12


def test_without_the_cap_that_command_can_run_away(world: Any) -> None:
    """Without the cap that command can run away.

    The companion to the bound above. A test that passes whether or not the
    cap is applied proves nothing about the cap, so this shows the same
    configuration asking for more than the bound when nothing holds it.
    """
    from clave.control.settings import ControlSettings

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    cap = settings.servo.max_joint_speed * 0.002

    model, data, arm = world
    place(world, near_singular(world))
    before = [float(data.ctrl[actuator]) for actuator in arm.actuator_ids]
    here = flange_of(world)
    step = follow(
        model,
        data,
        arm,
        Command(
            position=np.asarray(
                (here[0] + 0.004, here[1] + 0.004, here[2]), dtype=np.float64
            ),
            yaw=0.0,
        ),
        1.0,
    )
    assert step.refusal is None, step.refusal
    after = [float(data.ctrl[actuator]) for actuator in arm.actuator_ids]
    assert max(abs(now - was) for was, now in zip(before, after, strict=True)) > cap


def test_the_cap_is_measured_against_the_previous_command(world: Any) -> None:
    """AC-MOVE-11: the cap is measured against the previous command.

    Not against the measured position, which is the same distinction
    motion makes one layer up and matters for the same reason. These are
    position actuators running a proportional-derivative loop, so the
    command has to lead the position to produce force. Capping that lead
    instead of the command rate left almost no driving error: a flange asked
    to follow the belt at 0.31 m/s fell a metre behind inside two seconds.
    """
    mujoco = pytest.importorskip("mujoco")
    from clave.world import arm as armmod

    model, data, arm = world
    settle(
        world,
        Command(position=np.asarray((0.30, -0.30, 1.15), dtype=np.float64), yaw=0.0),
        steps=900,
    )
    cap = 2.09 * 0.002

    # Hold the arm still while the command walks away from it, then confirm
    # the command kept moving at the cap rather than stalling against the
    # frozen measurement.
    target = np.asarray((0.30, -0.30, 1.15), dtype=np.float64)
    for _ in range(50):
        follow(model, data, arm, Command(position=target, yaw=0.0), 1.0, cap)
    walked = [float(data.ctrl[actuator]) for actuator in arm.actuator_ids]
    for _ in range(50):
        follow(
            model,
            data,
            arm,
            Command(
                position=np.asarray((0.10, -0.40, 1.20), dtype=np.float64), yaw=0.0
            ),
            1.0,
            cap,
        )
        mujoco.mj_step(model, data)
    moved = [float(data.ctrl[actuator]) for actuator in arm.actuator_ids]
    del armmod
    assert max(abs(a - b) for a, b in zip(walked, moved, strict=True)) > cap


def track(world: Any, speed: float, lead: float) -> float:
    """Follow a target crossing the workspace and return the settled lag.

    On its own state rather than the module's. These runs care about the
    arm starting from rest at a known pose, and every test before them
    leaves it somewhere else, so sharing the fixture's `MjData` makes the
    result depend on collection order.
    """
    import statistics

    mujoco = pytest.importorskip("mujoco")
    numpy = pytest.importorskip("numpy")
    from clave.world import arm as armmod

    model, _, arm = world
    data = mujoco.MjData(model)
    start = numpy.array([-0.90, -0.20, 1.12])
    angles = armmod.solve(model, data, arm, start)
    for slot, joint in enumerate(arm.joint_ids):
        data.qpos[model.jnt_qposadr[joint]] = angles[slot]
    for slot, actuator in enumerate(arm.actuator_ids):
        data.ctrl[actuator] = angles[slot]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    span, errors = 1.50 / speed, []
    for tick in range(int(span / 0.002)):
        at_seconds = tick * 0.002
        target = start + numpy.array([speed * at_seconds, 0.0, 0.0])
        if not armmod.reachable(arm, target):
            break
        follow(
            model,
            data,
            arm,
            Command(
                position=np.asarray(
                    (target[0], target[1], target[2]), dtype=np.float64
                ),
                yaw=0.0,
                velocity=np.asarray((speed, 0.0, 0.0), dtype=np.float64),
            ),
            1.0,
            lead_seconds=lead,
        )
        mujoco.mj_step(model, data)
        if at_seconds > 0.6 * span:
            here = armmod.end_effector_position(data, arm)
            errors.append(math.dist(here, target))
    return statistics.median(errors)


def test_the_lag_against_a_moving_command_is_proportional_to_its_speed(
    world: Any,
) -> None:
    """The lag against a moving command is proportional to its speed.

    Which is what makes it a time constant rather than a distance, and
    therefore what makes a single lead cancel it at every speed. If this
    ever stops holding, one configured lead is the wrong shape of fix.
    """
    constants = [track(world, speed, 0.0) / speed for speed in (0.31, 0.50, 1.00)]
    assert max(constants) / min(constants) < 1.3, constants
    for constant in constants:
        assert 0.025 < constant < 0.045


def test_leading_the_plant_cancels_most_of_that_lag(world: Any) -> None:
    """AC-MOVE-36: leading the plant cancels most of that lag."""
    from clave.control.settings import ControlSettings

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    for speed in (0.31, 1.00):
        plain = track(world, speed, 0.0)
        led = track(world, speed, settings.servo.lead_seconds)
        assert led < 0.2 * plain, f"{speed} m/s: {plain * 1000:.1f} -> {led * 1000:.1f}"


def test_what_is_left_fits_inside_the_jaw_clearance(world: Any) -> None:
    """AC-MOVE-36: what is left fits inside the jaw clearance.

    The number the lead exists for. The jaw's clear opening is 85.2 mm and
    the widest object left in the set is 67.8 mm, so the tightest side
    clearance is 8.7 mm. A flange trailing by more than that closes the jaw
    onto an object rather than around it.
    """
    from clave.control.settings import ControlSettings

    settings = ControlSettings.load(load(ROOT / "configs" / "runtime" / "control.yml"))
    tightest = (0.0852 - 0.0678) / 2.0
    for speed in (0.31, 0.50, 1.00):
        assert track(world, speed, settings.servo.lead_seconds) < tightest


def test_a_lead_at_the_edge_of_reach_is_projected_rather_than_refused(
    world: Any,
) -> None:
    """AC-MOVE-36: a lead at the edge of reach is projected, not refused.

    Leading a pose that sits just inside the annulus can push it just
    outside. Faulting the track for that would blame the object for the
    correction, so the led pose is projected back and the pose the phase
    asked for still decides whether it is reachable.
    """
    from clave.world import arm as armmod

    model, data, arm = world
    base = (float(arm.base_position[0]), float(arm.base_position[1]))

    def keep_inside(pose: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = armmod.project_into_reach(
            base, float(pose[0]), float(pose[1]), arm.reach
        )
        return np.asarray((x, y, float(pose[2])), dtype=np.float64)

    edge = np.asarray(
        (base[0] + arm.reach.reach_max - 0.005, base[1], 1.15),
        dtype=np.float64,
    )
    step = follow(
        model,
        data,
        arm,
        Command(
            position=edge,
            yaw=0.0,
            velocity=np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        ),
        1.0,
        lead_seconds=0.033,
        keep_inside=keep_inside,
    )
    assert step.refusal is None
