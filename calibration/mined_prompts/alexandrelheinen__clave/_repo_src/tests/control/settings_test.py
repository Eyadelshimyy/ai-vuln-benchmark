"""Every tunable the arm controller reads, and the refusals when one is absent."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from clave.control.settings import ControlSettings, Profile
from clave.errors import ClaveError
from clave.world.config import load

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "configs" / "runtime" / "control.yml"
WORLD = ROOT / "configs" / "world" / "sorting_line.yml"


def shipped() -> ControlSettings:
    """The settings the repository ships."""
    return ControlSettings.load(load(CONTROL))


def raw() -> dict[str, Any]:
    """The shipped file, parsed, for a test to damage a copy of."""
    return load(CONTROL)


def test_the_shipped_configuration_loads() -> None:
    """AC-MOVE-14: the shipped configuration loads."""
    settings = shipped()
    assert settings.selection.exit_weight > 0.0
    assert settings.selection.anchor_radius > 0.0
    assert settings.task.profile is Profile.FULL_VISIT
    assert settings.task.grasp_clearance > 0.0
    assert settings.task.approach_speed > 0.0
    assert settings.task.interception_limit > 0.0
    assert settings.motion.max_speed > 0.0
    assert settings.servo.gain > 0.0


@pytest.mark.parametrize(
    ("block", "key"),
    [
        ("selection", "exit_weight"),
        ("selection", "anchor_radius_meters"),
        ("task", "profile"),
        ("task", "approach_height_meters"),
        ("task", "dwell_seconds"),
        ("task", "park_position_meters"),
        ("task", "park_marker_color"),
        ("motion", "max_speed_meters_per_second"),
        ("motion", "max_acceleration_meters_per_second_squared"),
        ("servo", "gain"),
        ("servo", "max_joint_speed_radians_per_second"),
        ("calibration", "flange_offset_meters"),
    ],
)
def test_an_absent_key_fails_at_load_naming_itself(block: str, key: str) -> None:
    """AC-MOVE-14: an absent key fails at load naming itself."""
    parsed = raw()
    del parsed[block][key]
    with pytest.raises(ClaveError, match=key):
        ControlSettings.load(parsed)


def test_an_absent_block_fails_at_load_naming_itself() -> None:
    """AC-MOVE-14: an absent block fails at load naming itself."""
    parsed = raw()
    del parsed["motion"]
    with pytest.raises(ClaveError, match="motion"):
        ControlSettings.load(parsed)


def test_an_unknown_profile_names_the_ones_that_exist() -> None:
    """An unknown profile names the ones that exist.

    A typo in a profile name is a run that silently did something else, so the
    error lists what it could have been.
    """
    parsed = raw()
    parsed["task"]["profile"] = "full_send"
    with pytest.raises(ClaveError, match="motion_only"):
        ControlSettings.load(parsed)


def test_both_profiles_are_selectable() -> None:
    """AC-MOVE-18: both profiles are selectable."""
    parsed = raw()
    parsed["task"]["profile"] = "full_visit"
    assert ControlSettings.load(parsed).task.profile is Profile.FULL_VISIT


@pytest.mark.parametrize(
    ("block", "key"),
    [
        ("selection", "anchor_radius_meters"),
        ("task", "approach_height_meters"),
        ("motion", "max_speed_meters_per_second"),
        ("motion", "max_acceleration_meters_per_second_squared"),
        ("servo", "max_joint_speed_radians_per_second"),
    ],
)
def test_a_bound_at_or_below_zero_is_refused(block: str, key: str) -> None:
    """A bound at or below zero is refused.

    A speed limit of zero is an arm that never moves and a run that looks like
    a controller failure. Refusing at load says which it was.
    """
    parsed = raw()
    parsed[block][key] = 0.0
    with pytest.raises(ClaveError, match=key):
        ControlSettings.load(parsed)


def test_a_point_that_is_not_three_numbers_is_refused() -> None:
    """A point that is not three numbers is refused."""
    parsed = raw()
    parsed["task"]["park_position_meters"] = [0.45, -1.00]
    with pytest.raises(ClaveError, match="park_position_meters"):
        ControlSettings.load(parsed)


def test_the_calibration_offset_is_zero_until_something_measures_it() -> None:
    """The calibration offset is zero until something measures it.

    There is no effector on this arm, so a non-zero offset would correct for a
    mounting that does not exist. This fails when one appears without the
    measurement that justifies it.
    """
    assert shipped().calibration.flange_offset == pytest.approx(
        np.zeros(3, dtype=np.float64),
    )


def test_the_park_pose_lies_inside_the_region_the_arm_is_trusted_over() -> None:
    """AC-MOVE-03: the park pose lies inside the region the arm is trusted over.

    Read from the world rather than restated here, so moving the arm or the
    annulus fails this instead of leaving the park pose stranded outside it.
    """
    from clave.world import arm as armmod

    park = shipped().task.park_position
    world = load(WORLD)
    base = [float(value) for value in world["arm"]["base_position_meters"]]

    band = [float(v) for v in world["arm"]["tool_above_base_meters"]]
    bounds = armmod.ReachBounds(
        float(world["arm"]["reach_min_meters"]),
        float(world["arm"]["reach_max_meters"]),
        (band[0], band[1]),
    )
    assert armmod.reaches((base[0], base[1]), park[0], park[1], bounds)
    lowest, highest = bounds.tool_above_base
    assert lowest <= park[2] - base[2] <= highest


def test_the_park_pose_stands_clear_of_the_sensing_gate() -> None:
    """The park pose stands clear of the sensing gate.

    An arm resting under the detection camera puts itself in the frames the
    tracker reads. The gate images about 0.92 m along travel, so the park pose
    has to sit outside that span and off the belt.
    """
    world = load(WORLD)
    park = shipped().task.park_position
    gate = next(
        entry for entry in world["cameras"] if str(entry["role"]) == "detection"
    )
    gate_x = float(gate["position_meters"][0])
    half_span = 0.92 / 2.0
    assert abs(park[0] - gate_x) > half_span, "the arm parks inside the gate"

    half_width = float(world["belt"]["width_meters"]) / 2.0
    assert abs(park[1]) > half_width, "the arm parks over the belt"


def test_the_park_marker_is_red_and_unlike_any_track_colour() -> None:
    """The park marker is red and unlike any track colour.

    The park pose is the one marker that is not a track, so it has to be
    distinguishable from every track colour the palette can produce.
    """
    from clave.tracker.markers import color_for

    park = shipped().task.park_marker_color
    assert park[0] > 0.5 and park[1] < 0.3 and park[2] < 0.3
    for track_id in range(64):
        assert math.dist(color_for(track_id), park) > 0.2


def test_the_park_pose_is_drawn_where_the_configuration_puts_it() -> None:
    """The park pose is drawn where the configuration puts it.

    Configurable and visible are the same requirement here: the point of
    drawing it is that a reader can check the number without reading the
    file.
    """
    mujoco = pytest.importorskip("mujoco")
    from clave.tracker.markers import draw_park

    settings = shipped()
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="2 2 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        first = renderer.scene.ngeom
        added = draw_park(
            renderer.scene,
            settings.task.park_position,
            settings.task.park_marker_color,
            belt_surface=0.90,
        )
        assert added == 2
        ball = renderer.scene.geoms[first]
        assert tuple(round(float(value), 6) for value in ball.pos[:3]) == tuple(
            round(value, 6) for value in settings.task.park_position
        )
        assert tuple(round(float(c), 3) for c in ball.rgba[:3]) == tuple(
            round(c, 3) for c in settings.task.park_marker_color
        )
    finally:
        renderer.close()


def test_the_park_marker_stops_at_the_scenes_capacity() -> None:
    """AC-MOVE-15: the park marker stops at the scene's capacity."""
    mujoco = pytest.importorskip("mujoco")
    from clave.tracker.markers import draw_park

    settings = shipped()
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="2 2 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        renderer.scene.ngeom = renderer.scene.maxgeom
        assert (
            draw_park(
                renderer.scene,
                settings.task.park_position,
                settings.task.park_marker_color,
                belt_surface=0.90,
            )
            == 0
        )
    finally:
        renderer.close()


def test_the_debug_views_are_named_and_one_is_the_default() -> None:
    """The debug views are named and one is the default."""
    debug = load(ROOT / "configs" / "debug" / "tracker.yml")
    assert set(debug["views"]) >= {"belt", "line"}
    assert debug["default_view"] in debug["views"]


def test_an_unknown_view_names_the_ones_that_exist() -> None:
    """An unknown view names the ones that exist."""
    from clave.sim.overlay import _view
    from clave.sim.report import DebugRunError

    debug = load(ROOT / "configs" / "debug" / "tracker.yml")
    with pytest.raises(DebugRunError, match="belt"):
        _view(debug, "from_the_moon")


def test_the_park_marker_is_visible_from_the_default_view() -> None:
    """The park marker is visible from the default view.

    The point of drawing the park pose is that a person can check it, so the
    default view has to actually contain it. This renders the shipped world
    twice and demands the marker change pixels, which fails if the pose moves
    behind the pedestal or out of frame.
    """
    mujoco = pytest.importorskip("mujoco")
    numpy = pytest.importorskip("numpy")
    from clave.sim.overlay import _view
    from clave.tracker.markers import draw_park
    from clave.world import config as world_config
    from clave.world import scene

    world = load(WORLD)
    settings = shipped()
    view = _view(load(ROOT / "configs" / "debug" / "tracker.yml"), None)
    model, data, _ = scene.build(world, numpy.random.default_rng(0), ROOT)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.azimuth = float(view["azimuth_degrees"])
    camera.elevation = float(view["elevation_degrees"])
    camera.distance = float(view["distance_meters"])
    camera.lookat[:] = [float(value) for value in view["lookat_meters"]]

    renderer = mujoco.Renderer(model, height=270, width=480)
    try:
        renderer.update_scene(data, camera=camera)
        plain = renderer.render().copy()
        renderer.update_scene(data, camera=camera)
        draw_park(
            renderer.scene,
            settings.task.park_position,
            settings.task.park_marker_color,
            belt_surface=float(
                world_config.require(
                    world_config.require(world, "belt"), "surface_height_meters"
                )
            ),
        )
        marked = renderer.render()
    finally:
        renderer.close()
    assert int(numpy.any(plain != marked, axis=2).sum()) > 20
