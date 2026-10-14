"""The effector the pick geometry assumes."""

from __future__ import annotations

from pathlib import Path

import pytest

from clave.errors import ClaveError
from clave.world.config import load
from clave.world.effector import Effector

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"


def test_the_shipped_configuration_describes_an_effector() -> None:
    """The shipped configuration describes an effector."""
    effector = Effector.load(load(CONFIG))
    assert effector.finger_length > 0.0
    assert effector.pad_thickness > 0.0
    assert effector.pad_depth > 0.0
    assert effector.grasp_height > 0.0
    assert effector.lowest_below_flange > 0.0
    assert effector.jaw_clearance > 0.0


def test_the_jaw_closes_with_the_force_the_world_asks_for() -> None:
    """AC-EFF-01: the compiled model carries the closing force the world asks for.

    The vendored 2F-85 ships ±5 N·m, which is around a hundred newtons at the
    pads against objects that weigh ten to twenty grams. Left at that, closing
    on one presses it 27 mm through the belt surface and rides down with it,
    which is measured in the world configuration beside the number that
    replaced it. The limit is applied to the spec by the scene, so this reads
    it back out of the compiled model rather than out of the file that asked
    for it.
    """
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("mujoco")

    from clave.world import arm as armmod
    from clave.world import scene

    raw = load(CONFIG)
    compiled, _data, _plan = scene.build(raw, numpy.random.default_rng(0), ROOT)
    index = armmod.locate(
        compiled,
        armmod.ReachBounds(_plan.reach_min, _plan.reach_max, _plan.tool_above_base),
    ).gripper_actuator
    wanted = float(raw["effector"]["closing_torque_newton_meters"])
    assert compiled.actuator_forcerange[index][1] == pytest.approx(wanted)
    assert compiled.actuator_forcerange[index][0] == pytest.approx(-wanted)


def test_a_jaw_that_may_not_close_is_refused() -> None:
    """AC-EFF-01: a closing force of nothing is refused, not silently applied."""
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("mujoco")

    from clave.world import scene

    raw = load(CONFIG)
    raw["effector"]["closing_torque_newton_meters"] = 0.0
    with pytest.raises(ClaveError, match="closing_torque_newton_meters"):
        scene.build(raw, numpy.random.default_rng(0), ROOT)


def test_the_opening_comes_from_the_arm_rather_than_a_second_number() -> None:
    """The opening comes from the arm rather than a second number.

    `arm.max_grasp_width_meters` already bounds the object set. Repeating it
    here would let the two drift and let a world spawn objects its own marker
    calls unreachable.
    """
    raw = load(CONFIG)
    assert Effector.load(raw).opening == pytest.approx(
        float(raw["arm"]["max_grasp_width_meters"])
    )


def test_an_absent_key_fails_at_load_naming_itself() -> None:
    """An absent key fails at load naming itself."""
    raw = load(CONFIG)
    del raw["effector"]["finger_length_meters"]
    with pytest.raises(ClaveError, match="finger_length_meters"):
        Effector.load(raw)


def test_an_absent_block_fails_at_load_naming_itself() -> None:
    """An absent block fails at load naming itself."""
    raw = load(CONFIG)
    del raw["effector"]
    with pytest.raises(ClaveError, match="effector"):
        Effector.load(raw)


def test_a_pad_no_thicker_than_nothing_is_refused() -> None:
    """A pad no thicker than nothing is refused."""
    raw = load(CONFIG)
    raw["effector"]["pad_thickness_meters"] = 0.0
    with pytest.raises(ClaveError, match="pad_thickness_meters"):
        Effector.load(raw)


def test_the_grasp_height_stays_above_the_belt() -> None:
    """The grasp height stays above the belt.

    A pad plane at or below the surface describes a jaw closing through the
    belt, which is a configuration error rather than a tight grasp.
    """
    raw = load(CONFIG)
    raw["effector"]["grasp_height_meters"] = -0.01
    with pytest.raises(ClaveError, match="grasp_height_meters"):
        Effector.load(raw)


def test_a_grasp_plane_inside_the_jaw_clearance_is_refused() -> None:
    """AC-GRIP-13: a grasp plane inside the jaw clearance is refused.

    The jaw's lowest geometry hangs below the pinch point, so the plane the
    pinch point may stand at is the clearance plus that overhang. A configured
    grasp height below it is a run that closes the pads on the belt, and it is
    refused where the numbers are rather than where the contact is.
    """
    raw = load(CONFIG)
    jaw = Effector.load(raw)
    raw["effector"]["grasp_height_meters"] = jaw.pinch_floor - 0.001
    with pytest.raises(ClaveError, match="pads would close"):
        Effector.load(raw)


def test_an_open_reach_past_the_shut_one_is_refused() -> None:
    """AC-GRIP-14: an open reach past the shut one is refused.

    Closing would not drop the pads, and the hold would have nothing to rise
    by. That is a configuration that describes the linkage backwards.
    """
    raw = load(CONFIG)
    raw["effector"]["open_lowest_below_flange_meters"] = (
        float(raw["effector"]["lowest_below_flange_meters"]) + 0.001
    )
    with pytest.raises(ClaveError, match="open_lowest_below_flange_meters"):
        Effector.load(raw)


def test_an_absent_open_reach_fails_at_load_naming_itself() -> None:
    """AC-GRIP-14: an absent open reach fails at load naming itself."""
    raw = load(CONFIG)
    del raw["effector"]["open_lowest_below_flange_meters"]
    with pytest.raises(ClaveError, match="open_lowest_below_flange_meters"):
        Effector.load(raw)


def test_an_absent_clearance_key_fails_at_load_naming_itself() -> None:
    """AC-GRIP-13: an absent clearance key fails at load naming itself."""
    raw = load(CONFIG)
    del raw["effector"]["jaw_clearance_meters"]
    with pytest.raises(ClaveError, match="jaw_clearance_meters"):
        Effector.load(raw)
