"""The jaw on the flange, and whether it holds anything.

This is the first thing in CLAVE that can fail for a physical reason rather
than a logical one. A pose outside the envelope is refused by arithmetic; a
jaw that closes on a package and comes away empty is the simulator
disagreeing with the plan.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from clave.world.config import load

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "configs" / "world" / "sorting_line.yml"


@pytest.fixture(scope="module")
def world() -> Any:
    """The shipped world, compiled once for the module."""
    numpy = pytest.importorskip("numpy")
    mujoco = pytest.importorskip("mujoco")
    from clave.world import arm as armmod
    from clave.world import scene

    model, data, plan = scene.build(load(WORLD), numpy.random.default_rng(0), ROOT)
    mujoco.mj_forward(model, data)
    return (
        model,
        data,
        plan,
        armmod.locate(
            model,
            armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
        ),
    )


def test_the_gripper_is_on_the_flange(world: Any) -> None:
    """AC-GRIP-01: the gripper is on the flange."""
    from clave.world import arm as armmod

    model, data, _, arm = world
    del model
    assert arm.gripper_actuator >= 0
    assert arm.pinch_site >= 0
    flange = armmod.end_effector_position(data, arm)
    pinch = armmod.pinch_position(data, arm)
    assert math.dist(flange, pinch) > 0.10


def test_the_configured_finger_length_is_the_compiled_one(world: Any) -> None:
    """AC-GRIP-01: the configured finger length is the compiled one.

    The offset between the flange and the pinch point is a property of the
    vendored model, and the controller reads it from configuration. This is
    what stops the two drifting when the model is re-copied.
    """
    from clave.world import arm as armmod
    from clave.world.effector import Effector

    model, data, _, arm = world
    del model
    effector = Effector.load(load(WORLD))
    measured = math.dist(
        armmod.end_effector_position(data, arm), armmod.pinch_position(data, arm)
    )
    assert effector.finger_length == pytest.approx(measured, abs=0.001)
    assert effector.opening == pytest.approx(0.085)


def test_the_tool_site_every_pose_is_expressed_against_still_moves(
    world: Any,
) -> None:
    """AC-GRIP-01: the tool site every pose is expressed against still moves.

    The gripper hangs off the flange rather than displacing it, which is why
    the trusted reach bounds did not have to be re-swept. This fails if a
    future mounting moves the site instead.
    """
    mujoco = pytest.importorskip("mujoco")
    numpy = pytest.importorskip("numpy")
    from clave.world import arm as armmod

    model, data, _, arm = world
    before = armmod.end_effector_position(data, arm).copy()
    target = numpy.array([0.20, -0.30, 1.15])
    for _ in range(900):
        armmod.step_toward(model, data, arm, target, gain=1.0)
        mujoco.mj_step(model, data)
    after = armmod.end_effector_position(data, arm)
    assert math.dist(before, after) > 0.05
    assert math.dist(after, target) < 0.05


def test_the_jaw_opens_and_closes(world: Any) -> None:
    """AC-GRIP-05: the jaw opens and closes."""
    mujoco = pytest.importorskip("mujoco")
    from clave.world import arm as armmod

    model, data, _, arm = world

    def separation() -> float:
        left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_grip_left_pad")
        right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_grip_right_pad")
        return float(math.dist(data.xpos[left], data.xpos[right]))

    armmod.hold(data, arm, closed=0.0)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    wide = separation()

    armmod.hold(data, arm, closed=1.0)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    shut = separation()

    assert wide > shut + 0.05, f"open {wide * 1000:.1f} mm, shut {shut * 1000:.1f} mm"


def test_no_object_in_the_set_is_wider_than_the_jaw() -> None:
    """AC-GRIP-03: no object in the set is wider than the jaw.

    Enforced at load by the scene, which is why building the world is the
    test. This states the property so a reader finds it here rather than
    inferring it from a build that happened not to fail.
    """
    raw = load(WORLD)
    assert float(raw["arm"]["max_grasp_width_meters"]) == pytest.approx(0.085)
    assert not [
        entry for entry in raw["objects"] if entry.get("name") == "master_chef_can"
    ], "an object the jaw cannot close on is back in the set"


def _pad_box(world: Any) -> Any:
    """Return one pad's corners in the tool frame, flange at the origin.

    A finger's collision shape is two boxes, so the corners are taken over
    both, in the tool's own frame: the jaw closes along the tool's x, its
    depth is y, its height is z.

    Args:
        world: The compiled model, its data and the arm indices.

    Returns:
        The corners, one row per corner, in meters.
    """
    numpy = pytest.importorskip("numpy")
    import mujoco

    model, data, _, arm = world
    axes = data.site_xmat[arm.tool_site].reshape(3, 3)
    origin = data.site_xpos[arm.tool_site]
    corners = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if not name.startswith("arm_grip_") or "pad" not in name:
            continue
        if model.geom_contype[geom] == 0:
            continue
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
        )
        if body != "arm_grip_right_pad":
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        size = numpy.asarray(model.geom_size[geom])
        centre = data.geom_xpos[geom]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    offset = rotation @ numpy.array(
                        [sx * size[0], sy * size[1], sz * size[2]]
                    )
                    corners.append(axes.T @ (centre + offset - origin))
    return numpy.asarray(corners)


def _pad_geometry(world: Any, grip: float) -> tuple[tuple[float, float, float], float]:
    """Return one pad's extents in the tool frame and its lowest depth below it.

    Measured from the compiled model rather than read from the configuration,
    because the configuration is what is under test.

    The linkage moves, so the answer depends on where it stands: this settles
    the jaw at the commanded grip for long enough to be shut or open rather than
    asking `mj_forward`, which computes positions and never moves a finger.

    Args:
        world: The compiled model, its data and the arm indices.
        grip: What to command the jaw, from zero open to one shut.

    Returns:
        The extents along `(x, y, z)` of the tool frame, and how far the lowest
        corner of the pad reaches below the flange. Both in meters.
    """
    import mujoco

    from clave.world import arm as armmod

    model, data, _, arm = world
    for slot, actuator in enumerate(arm.actuator_ids):
        data.ctrl[actuator] = armmod.joint_positions(model, data, arm)[slot]
    armmod.hold(data, arm, closed=grip)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    box = _pad_box(world)
    span = box.max(axis=0) - box.min(axis=0)
    return (float(span[0]), float(span[1]), float(span[2])), float(box[:, 2].max())


def test_the_configured_pad_dimensions_are_the_compiled_ones(world: Any) -> None:
    """AC-GRIP-12: the configured pad dimensions are the compiled ones.

    The configuration is what a marker draws and what sizes the clearance it
    grants. It read 12 x 40 x 50 mm while the effector was hypothetical: a
    quarter turn from the real pad and a different size in every direction.

    Measured with the jaw open. Each pad is two collision boxes, one above the
    other on the same finger, and the linkage turns them relative to each other
    as the jaw shuts: the span of the pair grows by up to a millimetre with
    where the stop holds them, which is a property of the linkage and not a pad
    dimension. The open state is the one where the two are in line, so it is
    the one this reads.
    """
    from clave.world.effector import Effector

    jaw = Effector.load(load(WORLD))
    (thickness, depth, height), _ = _pad_geometry(world, 0.0)
    assert thickness == pytest.approx(jaw.pad_thickness, abs=0.001)
    assert depth == pytest.approx(jaw.pad_depth, abs=0.001)
    assert height == pytest.approx(jaw.pad_height, abs=0.001)


def test_the_configured_lowest_geometry_is_the_compiled_worst_case(world: Any) -> None:
    """AC-GRIP-13: the configured lowest geometry is the compiled worst case.

    The pads hang below the pinch point the arm is commanded to, and how far
    depends on where the linkage stands, so the number the configuration
    carries has to be the worst of the two states: the shut one, which is what
    the jaw is in when it is holding something.
    """
    from clave.world.effector import Effector

    jaw = Effector.load(load(WORLD))
    _, open_depth = _pad_geometry(world, 0.0)
    _, shut_depth = _pad_geometry(world, 1.0)
    assert shut_depth == pytest.approx(jaw.lowest_below_flange, abs=0.001)
    assert open_depth < shut_depth, "the shut jaw is not the lowest it gets"
    assert shut_depth > jaw.finger_length, "the pads are above the pinch point"


def test_the_open_jaw_is_the_configured_open_reach(world: Any) -> None:
    """AC-GRIP-14: the open jaw's lowest geometry is the compiled one.

    The descent arrives against this reach. The shut reach is what the hold
    rises to, and the two are not the same number.
    """
    from clave.world.effector import Effector

    jaw = Effector.load(load(WORLD))
    _, open_depth = _pad_geometry(world, 0.0)
    assert open_depth == pytest.approx(jaw.open_lowest_below_flange, abs=0.001)
    assert jaw.open_lowest_below_flange < jaw.lowest_below_flange


def test_the_rise_leads_the_hang_the_jaw_adds_as_it_shuts(world: Any) -> None:
    """AC-MOVE-67: the rise leads the hang the jaw adds as it shuts.

    A rest-to-rest quintic over the first 0.20 s of the hold climbs the extra
    hang of the shut jaw. The linkage adds that hang more slowly than the
    quintic, except at one sample, and the millimetre the descent stops short
    covers that sample. Past the rise the hold is already at the full climb.
    """
    import mujoco
    import numpy as np

    from clave.control.trajectory import Segment, State
    from clave.world import arm as armmod
    from clave.world.effector import ARRIVAL_LEAD_METERS, Effector

    jaw = Effector.load(load(WORLD))
    _, open_hang = _pad_geometry(world, 0.0)
    model, data, _, arm = world
    armmod.hold(data, arm, closed=1.0)
    rise = Segment(
        start=State.at_rest(np.zeros(3, dtype=np.float64)),
        end=State.at_rest(np.asarray((0.0, 0.0, jaw.closing_drop), dtype=np.float64)),
        duration=0.20,
    )
    step = float(model.opt.timestep)
    for index in range(1, int(round(0.40 / step)) + 1):
        mujoco.mj_step(model, data)
        growth = float(_pad_box(world)[:, 2].max()) - open_hang
        climbed = rise.at(index * step).position[2]
        assert growth <= climbed + ARRIVAL_LEAD_METERS + 1e-6
