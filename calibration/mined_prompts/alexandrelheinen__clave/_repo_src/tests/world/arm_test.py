"""Tests for manipulator actuation."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.world import arm as armmod
from clave.world import config, scene

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"


def built() -> tuple[Any, Any, armmod.ArmIndices]:
    """Build the shipped world and locate the arm in it."""

    import mujoco

    raw = config.load(CONFIG)
    model, data, plan = scene.build(raw, np.random.default_rng(0), ROOT)
    indices = armmod.locate(
        model,
        armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
    )
    mujoco.mj_forward(model, data)
    return model, data, indices


def test_the_arm_is_found_with_its_six_axes() -> None:
    """The controller drives the six axes a UR10e has."""
    pytest.importorskip("mujoco")
    _, _, indices = built()
    assert len(indices.joint_ids) == 6
    assert len(indices.actuator_ids) == 6
    assert indices.tool_site >= 0


def test_joint_limits_are_read_from_the_model() -> None:
    """Limits come from the vendored model, never from a number written here."""
    pytest.importorskip("mujoco")
    _, _, indices = built()
    assert np.all(indices.lower < indices.upper)


def test_the_base_is_the_mounting_face_not_the_first_joint() -> None:
    """Every trusted bound is measured from the pedestal top.

    The shoulder-pan anchor sits 0.181 m above it, and reporting that instead
    would shift the whole vertical band by that much without failing anything
    loudly.
    """
    pytest.importorskip("mujoco")
    _, _, indices = built()
    raw = config.load(ROOT / "configs" / "world" / "sorting_line.yml")
    configured = config.require(
        config.require(raw, "arm"), "base_position_meters", "arm"
    )
    assert np.allclose(indices.base_position, [float(v) for v in configured])


def test_proprioception_reports_the_arm_joint_angles() -> None:
    """The dataset had none of this, which is why policies were vision only."""
    pytest.importorskip("mujoco")
    model, data, indices = built()
    angles = armmod.joint_positions(model, data, indices)
    assert angles.shape == (6,)
    assert np.all(np.isfinite(angles))


def _pick_target(indices: armmod.ArmIndices) -> NDArray[np.float64]:
    """A point on the belt the arm is trusted over."""
    base = indices.base_position
    return np.array([0.0, base[1] + 0.70, base[2] + 0.05])


def test_the_solver_places_the_tool_on_the_target() -> None:
    """Solved with the tool held vertical, not compared to a radius."""
    pytest.importorskip("mujoco")
    import mujoco

    model, data, indices = built()
    target = _pick_target(indices)
    solved = armmod.solve(model, data, indices, target, yaw=0.3)
    for joint, value in zip(indices.joint_ids, solved, strict=True):
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(armmod.end_effector_position(data, indices) - target) < 2e-3


def test_the_solver_holds_the_tool_vertical() -> None:
    """The constraint that makes a six-axis arm serve a four-axis task."""
    pytest.importorskip("mujoco")
    import mujoco

    model, data, indices = built()
    solved = armmod.solve(model, data, indices, _pick_target(indices))
    for joint, value in zip(indices.joint_ids, solved, strict=True):
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    axis = data.site_xmat[indices.tool_site].reshape(3, 3)[:, 2]
    assert axis[2] < -0.99, f"tool axis is not pointing down: {axis}"


def test_the_solver_leaves_the_simulation_state_alone() -> None:
    """Asking a question must not advance anybody's world."""
    pytest.importorskip("mujoco")
    model, data, indices = built()
    before = data.qpos.copy()
    armmod.solve(model, data, indices, _pick_target(indices))
    assert np.array_equal(data.qpos, before)


def test_a_target_under_the_base_is_refused() -> None:
    """The inner radius is a real property of the arm, not an approximation."""
    pytest.importorskip("mujoco")
    _, _, indices = built()
    base = indices.base_position
    under = np.array([base[0], base[1], base[2] + 0.05])
    assert not armmod.reachable(indices, under)


def test_a_target_past_the_outer_radius_is_refused() -> None:
    """Beyond the annulus the solver refuses rather than clips."""
    pytest.importorskip("mujoco")
    model, data, indices = built()
    base = indices.base_position
    far = np.array([0.0, base[1] + indices.reach.reach_max + 0.3, base[2]])
    assert not armmod.reachable(indices, far)
    with pytest.raises(armmod.ReachError):
        armmod.solve(model, data, indices, far)


def test_the_trusted_region_is_inside_what_the_arm_can_reach() -> None:
    """The region is proven, not asserted.

    `reaches` applies a fixed annulus rather than solving, because a per-call
    iterative solve would be slow and seed-dependent. That is only honest while
    every point the annulus admits really does admit a solution, which is what
    this re-establishes.
    """
    pytest.importorskip("mujoco")
    model, data, indices = built()
    refused = []
    for x in np.arange(-1.1, 1.11, 0.275):
        for y in np.arange(-0.5, 0.51, 0.25):
            target = np.array([x, y, indices.base_position[2] + 0.05])
            if not armmod.reachable(indices, target):
                continue
            try:
                armmod.solve(model, data, indices, target)
            except armmod.ReachError:
                refused.append((round(float(x), 2), round(float(y), 2)))
    assert not refused, f"trusted but unreachable: {refused}"


def test_the_solver_prefers_an_extended_reach_over_a_folded_wrist() -> None:
    """AC-BEHAVE-04: park IK prefers the extended branch over a folded wrist.

    The default warm start from the zero pose lands on a negative elbow with
    wrist_2 near 270°. Ranking converged restarts by posture cost must pick
    the extended branch instead: elbow positive, shoulder lift at or below
    zero, and wrist_2 near ±90° after folding into (−π, π].
    """
    pytest.importorskip("mujoco")
    import math

    import mujoco

    model, data, indices = built()
    park = np.array([0.72, -0.42, 1.20], dtype=np.float64)
    solved = armmod.solve(model, data, indices, park)
    elbow = armmod._principal(float(solved[2]))
    shoulder_lift = armmod._principal(float(solved[1]))
    wrist_2 = armmod._principal(float(solved[4]))
    assert elbow > 0.0, f"elbow stayed folded: {math.degrees(elbow):.1f}°"
    assert shoulder_lift <= 0.0, (
        f"shoulder lift flipped overhead: {math.degrees(shoulder_lift):.1f}°"
    )
    assert abs(abs(wrist_2) - math.pi / 2.0) < math.radians(5.0), (
        f"wrist_2 not near ±90°: {math.degrees(wrist_2):.1f}°"
    )
    assert abs(wrist_2) < math.pi, "wrist_2 was not folded into (−π, π]"
    for joint, value in zip(indices.joint_ids, solved, strict=True):
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(armmod.end_effector_position(data, indices) - park) < 2e-3


def test_commanding_a_target_moves_the_end_effector_toward_it() -> None:
    """The point of the whole module: something writes data.ctrl."""
    pytest.importorskip("mujoco")
    import mujoco

    model, data, indices = built()
    target = _pick_target(indices)
    before = np.linalg.norm(target - armmod.end_effector_position(data, indices))
    for _ in range(1500):
        armmod.step_toward(model, data, indices, target, gain=0.4)
        mujoco.mj_step(model, data)
    after = np.linalg.norm(target - armmod.end_effector_position(data, indices))
    assert after < before / 2.0


def test_commanded_angles_stay_inside_the_joint_limits() -> None:
    """A target outside the workspace must not drive a joint past its stop."""
    pytest.importorskip("mujoco")
    import mujoco

    model, data, indices = built()
    unreachable = np.array([5.0, 5.0, 5.0])
    for _ in range(200):
        commanded = armmod.step_toward(model, data, indices, unreachable, gain=1.0)
        mujoco.mj_step(model, data)
        assert np.all(commanded >= indices.lower - 1e-9)
        assert np.all(commanded <= indices.upper + 1e-9)


def test_a_missing_arm_is_reported_by_name() -> None:
    """A moved model or a changed prefix should fail legibly."""
    pytest.importorskip("mujoco")
    import mujoco

    empty = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(KeyError, match="arm_shoulder_pan_joint"):
        armmod.locate(empty, armmod.ReachBounds(0.25, 1.25, (-0.05, 0.45)))


# The stall measured when the command was already on the mass. The roll sits
# on its upper stop. Clipped to the compiled limits before it is used.
_STALL_JOINTS = np.array(
    [
        2.645491048232159,
        -2.883348142443644,
        -0.001063485045729,
        -4.969290856013832,
        4.712429309536971,
        6.28319,
    ]
)
_STALL_TARGET = np.array([-0.823157126636458, -0.05745090470265107, 1.2838158909149542])
_STALL_YAW = -2.3343556939486056


def test_cross3_matches_numpy_for_length_three_vectors() -> None:
    """The local cross matches numpy on the three-axis case the arm uses."""
    left = np.array([0.2, -0.5, 0.9], dtype=np.float64)
    right = np.array([-0.3, 0.4, 0.1], dtype=np.float64)
    out = np.empty(3, dtype=np.float64)
    assert armmod._cross3(left, right, out=out) is out
    assert out == pytest.approx(np.cross(left, right))


def test_a_wrist_on_its_stop_is_not_walked_off_the_command() -> None:
    """AC-MOVE-64: a wrist on its stop is not walked off the command.

    These joint angles are the stall measured when the command was already on
    the mass. The roll sits on its upper stop, and a further descent cannot
    spend it. The same jaw the other way round does move the roll, and it is
    not taken: that pose is three quarters of a radian away, and the command
    can only approach it at the joint-speed cap, which flies the tilt between
    the two. What is taken is the descent that does not increase the miss.
    """
    pytest.importorskip("mujoco")
    model, _, indices = built()
    seed = np.clip(_STALL_JOINTS, indices.lower, indices.upper)
    plain = armmod._descend(model, indices, seed, _STALL_TARGET, _STALL_YAW, 8)
    tracked = armmod._track_pose(model, indices, seed, _STALL_TARGET, _STALL_YAW, 8)
    plain_gap = armmod._tool_distance(model, indices, plain, _STALL_TARGET)
    tracked_gap = armmod._tool_distance(model, indices, tracked, _STALL_TARGET)
    seed_gap = armmod._tool_distance(model, indices, seed, _STALL_TARGET)
    assert abs(float(tracked[5]) - float(seed[5])) < 0.05
    assert tracked_gap <= seed_gap + 1e-9
    assert tracked_gap <= plain_gap + 1e-9


def test_a_descent_that_walks_off_the_command_is_discarded() -> None:
    """AC-MOVE-64: a descent that walks the tool off the command is discarded.

    At the same stall the tool is already on a point, and a descent that
    chases the remaining yaw walks it off that point. That step is not the
    one the next tick starts from.
    """
    pytest.importorskip("mujoco")
    import mujoco

    model, data, indices = built()
    seed = np.clip(_STALL_JOINTS, indices.lower, indices.upper)
    for slot, joint in enumerate(indices.joint_ids):
        data.qpos[model.jnt_qposadr[joint]] = seed[slot]
    mujoco.mj_forward(model, data)
    here = np.array(data.site_xpos[indices.tool_site], dtype=np.float64)
    plain = armmod._descend(model, indices, seed, here, _STALL_YAW, 8)
    tracked = armmod._track_pose(model, indices, seed, here, _STALL_YAW, 8)
    plain_gap = armmod._tool_distance(model, indices, plain, here)
    tracked_gap = armmod._tool_distance(model, indices, tracked, here)
    assert plain_gap > 1e-4
    assert tracked_gap <= 1e-6
