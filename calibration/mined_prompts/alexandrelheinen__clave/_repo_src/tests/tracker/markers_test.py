"""The grasp pose, drawn in the world rather than onto a frame."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from clave.tracker.belt_frame import Footprint
from clave.tracker.evidence import Role
from clave.tracker.markers import (
    MAX_GRASP_ABOVE_SURFACE_METERS,
    UNORIENTED_PADS,
    GraspMarker,
    color_for,
    draw,
    grasp_plane,
    marker_for,
    markers_for,
)
from clave.tracker.track import WasteObject
from clave.world.config import load
from clave.world.effector import Effector

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"
BELT_SURFACE = 0.90


def effector() -> Effector:
    """The shipped effector, so the test moves when the configuration does."""
    return Effector.load(load(CONFIG))


def record(
    track_id: int = 1,
    center: NDArray[np.float64] | None = None,
    major: float = 0.12,
    minor: float = 0.06,
    yaw: float = 0.0,
    oriented: bool = True,
) -> WasteObject:
    """One settled record, with only the fields a marker reads set."""
    if center is None:
        center = np.asarray((0.2, 0.1, 0.95), dtype=np.float64)
    return WasteObject(
        track_id=track_id,
        observed_at_nanos=0,
        valid_until_nanos=1,
        footprint=Footprint(
            center=center,
            major_extent=major,
            minor_extent=minor,
            yaw=yaw,
            oriented=oriented,
        ),
        height=None,
        material="M-02",
        material_confidence=0.9,
        density=None,
        mass=None,
        codes=(),
        evidence=frozenset({Role.DETECTION}),
        simulated=frozenset(),
    )


def test_the_pads_sit_over_the_records_grasp_point() -> None:
    """AC-MARK-02: the pads sit over the record's grasp point."""
    marker = marker_for(
        record(center=np.asarray((0.35, -0.12, 0.95), dtype=np.float64)),
        effector(),
        BELT_SURFACE,
    )
    left, right = marker.pads
    assert (left[0] + right[0]) / 2.0 == pytest.approx(0.35)
    assert (left[1] + right[1]) / 2.0 == pytest.approx(-0.12)


def test_the_pad_plane_is_the_configured_height_above_the_belt() -> None:
    """The pad plane is the configured height above the belt.

    Nothing on this line estimates a height, so the vertical placement is a
    configured standoff rather than a reading, and the test says which.
    """
    tool = effector()
    marker = marker_for(record(), tool, BELT_SURFACE)
    for pad in marker.pads:
        assert pad[2] == pytest.approx(BELT_SURFACE + tool.grasp_height)


def test_the_jaw_closes_across_the_minor_extent() -> None:
    """AC-MARK-03: the jaw closes across the minor extent."""
    marker = marker_for(record(yaw=0.0), effector(), BELT_SURFACE)
    # The major axis lies along x, so the jaw closes along y.
    assert marker.closing_axis == pytest.approx(math.pi / 2.0)
    left, right = marker.pads
    assert abs(left[1] - right[1]) > abs(left[0] - right[0])


def test_the_jaw_turns_with_the_footprint() -> None:
    """The jaw turns with the footprint."""
    turned = marker_for(record(yaw=math.pi / 2.0), effector(), BELT_SURFACE)
    left, right = turned.pads
    # A major axis along y puts the closing direction along x.
    assert abs(left[0] - right[0]) > abs(left[1] - right[1])


def test_an_unoriented_footprint_gets_no_jaw_axis() -> None:
    """AC-MARK-04: an unoriented footprint gets no jaw axis."""
    marker = marker_for(
        record(major=0.08, minor=0.08, yaw=1.1, oriented=False),
        effector(),
        BELT_SURFACE,
    )
    assert marker.oriented is False
    assert marker.closing_axis is None
    assert marker.pads == ()


def test_the_pads_are_separated_by_the_grasp_width() -> None:
    """AC-MARK-05: the pads are separated by the grasp width."""
    tool = effector()
    marker = marker_for(record(minor=0.05), tool, BELT_SURFACE)
    left, right = marker.pads
    gap = math.dist(left[:2], right[:2]) - tool.pad_thickness
    assert gap == pytest.approx(0.05)


def test_the_flange_stands_a_finger_length_above_the_pads() -> None:
    """AC-MARK-06: the flange stands a finger length above the pads."""
    tool = effector()
    marker = marker_for(record(), tool, BELT_SURFACE)
    assert marker.flange[2] == pytest.approx(
        BELT_SURFACE + tool.grasp_height + tool.finger_length
    )
    assert marker.flange[:2] == pytest.approx((0.2, 0.1))


def test_the_pads_are_the_shape_of_the_effector_the_marker_was_given() -> None:
    """AC-MARK-15: the pads are the shape of the effector it was given.

    The marker carried 12 x 40 x 50 mm pads while the effector was hypothetical:
    a quarter turn from the jaw the model closes and a different size in every
    direction. They come from the effector the caller passes now, and
    `tests/world/gripper_test.py` is what pins that effector to the compiled
    model.
    """
    tool = Effector(
        finger_length=0.1558,
        pad_thickness=0.011,
        pad_depth=0.033,
        pad_height=0.044,
        grasp_height=0.030,
        lowest_below_flange=0.1735,
        open_lowest_below_flange=0.1603,
        jaw_clearance=0.010,
        opening=0.085,
    )
    marker = marker_for(record(minor=0.03), tool, BELT_SURFACE)
    assert marker.pad_size == pytest.approx((0.0055, 0.0165, 0.022))
    left, right = marker.pad_positions_belt
    reach = marker.opening / 2.0 + tool.pad_thickness / 2.0
    assert math.dist(left[:2], right[:2]) == pytest.approx(2.0 * reach)


def test_the_grasp_plane_stands_on_the_open_jaw() -> None:
    """AC-MARK-17: the grasp plane's floor is the open jaw's pinch floor.

    A centre below that floor is lifted to it. A centre above it is taken as
    it stands, and a centre past the height an object on the belt still has
    is clamped there. The shut hang is what the hold rises by, not the floor
    the descent arrives at.
    """
    tool = effector()
    floor = BELT_SURFACE + tool.open_pinch_floor
    assert grasp_plane(BELT_SURFACE, tool, floor - 0.020) == pytest.approx(floor)
    stood = grasp_plane(BELT_SURFACE, tool, floor + 0.010)
    assert stood == pytest.approx(floor + 0.010)
    ceiling = BELT_SURFACE + MAX_GRASP_ABOVE_SURFACE_METERS
    high = grasp_plane(BELT_SURFACE, tool, BELT_SURFACE + 1.0)
    assert high == pytest.approx(ceiling)
    assert floor < BELT_SURFACE + tool.pinch_floor


def test_an_object_wider_than_the_jaw_is_marked_unreachable() -> None:
    """AC-MARK-07: an object wider than the jaw is marked unreachable."""
    tool = effector()
    wide = marker_for(
        record(major=tool.opening + 0.10, minor=tool.opening + 0.05),
        tool,
        BELT_SURFACE,
    )
    assert wide.reachable is False
    assert marker_for(record(), tool, BELT_SURFACE).reachable is True


def test_a_marker_carries_the_tracks_own_colour() -> None:
    """AC-MARK-08: a marker carries the track's own colour."""
    marker = marker_for(record(track_id=7), effector(), BELT_SURFACE)
    assert marker.color == color_for(7)
    assert color_for(7) != color_for(8)


def test_a_colour_is_three_channels_inside_the_unit_range() -> None:
    """A colour is three channels inside the unit range."""
    for track_id in range(12):
        channels = color_for(track_id)
        assert len(channels) == 3
        assert all(0.0 <= channel <= 1.0 for channel in channels)


def test_one_marker_is_produced_per_open_track() -> None:
    """AC-MARK-09: one marker is produced per open track."""
    records = (record(track_id=1), record(track_id=2), record(track_id=5))
    markers = markers_for(records, effector(), BELT_SURFACE)
    assert tuple(marker.track_id for marker in markers) == (1, 2, 5)
    assert markers_for((), effector(), BELT_SURFACE) == ()


def test_drawing_adds_geometry_and_reports_how_much() -> None:
    """AC-MARK-10: drawing adds geometry and reports how much."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        before = renderer.scene.ngeom
        added = draw(renderer.scene, markers_for((record(),), effector(), BELT_SURFACE))
        assert added > 0
        assert renderer.scene.ngeom == before + added
    finally:
        renderer.close()


def test_drawing_stops_at_the_scenes_capacity() -> None:
    """AC-MARK-11: drawing stops at the scene's capacity."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        renderer.scene.ngeom = renderer.scene.maxgeom
        many = markers_for(
            tuple(record(track_id=n) for n in range(50)), effector(), BELT_SURFACE
        )
        assert draw(renderer.scene, many) == 0
        assert renderer.scene.ngeom == renderer.scene.maxgeom
    finally:
        renderer.close()


def test_a_marker_is_geometry_rather_than_a_painted_frame() -> None:
    """AC-MARK-13: a marker is geometry rather than a painted frame.

    The render changes because the scene changed, which is the whole
    distinction between a marker and an overlay. If this ever passes with
    `draw` doing nothing to the scene, the marker has become a post-process.
    """
    mujoco = pytest.importorskip("mujoco")
    numpy = pytest.importorskip("numpy")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="plane" size="2 2 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 120.0, -25.0, 1.5
    camera.lookat[:] = [0.2, 0.1, 0.95]
    renderer = mujoco.Renderer(model, height=120, width=160)
    try:
        renderer.update_scene(data, camera=camera)
        plain = renderer.render().copy()
        renderer.update_scene(data, camera=camera)
        added = draw(renderer.scene, markers_for((record(),), effector(), BELT_SURFACE))
        marked = renderer.render()
    finally:
        renderer.close()
    # Asserted apart, so a failure says which half broke: the marker never
    # entered the scene, or it entered and the renderer drew nothing.
    assert added > 0, "no marker entered the scene"
    assert int(numpy.any(plain != marked, axis=2).sum()) > 0, "the scene rendered flat"


def test_the_marker_type_says_nothing_about_a_sensor() -> None:
    """The marker type says nothing about a sensor.

    A marker is read from the record, which names roles rather than cameras,
    so nothing here may grow a source id.
    """
    fields = set(GraspMarker.__dataclass_fields__)
    assert not {name for name in fields if "source" in name or "camera" in name}


def test_an_unoriented_marker_rings_the_object_rather_than_covering_it() -> None:
    """An unoriented marker rings the object rather than covering it.

    A track with no detection takes the configured unmeasured extent, which
    is the widest the jaw opens. A filled disc that size hides exactly the
    object a reader is checking the marker against, and a single pair of pads
    would claim an axis the record declined to give.
    """
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        first = renderer.scene.ngeom
        added = draw(
            renderer.scene,
            markers_for(
                (record(major=0.18, minor=0.18, oriented=False),),
                effector(),
                BELT_SURFACE,
            ),
        )
        # One shaft and a ring of pads, none of them a disc over the object.
        assert added == 1 + UNORIENTED_PADS
        types = {int(renderer.scene.geoms[first + n].type) for n in range(1, added)}
        assert types == {int(mujoco.mjtGeom.mjGEOM_BOX)}
    finally:
        renderer.close()


def world() -> tuple[Any, Any]:
    """A model with one object and two cameras, standing in for the line."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <light pos="0 0 3"/>
            <geom name="belt" type="plane" size="2 1 .1" pos="0 0 .9"/>
            <body pos="0.2 0.1 0.95">
              <geom name="package" type="box" size=".06 .03 .05"/>
            </body>
            <camera name="gate" pos="0.2 0.1 2.0" euler="0 0 0"/>
            <camera name="watch" pos="1.2 -1.0 1.8" euler="1.0 0 0.9"/>
          </worldbody>
        </mujoco>
        """
    )
    return model, mujoco.MjData(model)


def test_a_marker_never_reaches_the_camera_the_tracker_reads() -> None:
    """AC-MARK-10: a marker never reaches the camera the tracker reads.

    The markers exist so a person can see where a pick is planned. A pixel of
    one landing in the detection render would feed the perception stage an
    object that is not there, so this renders the gate with markers standing
    in the other renderer's scene and demands the segmentation be identical
    to a run where no marker was ever drawn.
    """
    mujoco = pytest.importorskip("mujoco")
    numpy = pytest.importorskip("numpy")
    model, data = world()
    mujoco.mj_forward(model, data)

    gate = mujoco.Renderer(model, height=120, width=160)
    gate.enable_segmentation_rendering()
    watching = mujoco.Renderer(model, height=120, width=160)
    try:
        gate.update_scene(data, camera="gate")
        clean = gate.render().copy()

        watching.update_scene(data, camera="watch")
        assert draw(watching.scene, markers_for((record(),), effector(), 0.9)) > 0
        marked_view = watching.render()

        gate.update_scene(data, camera="gate")
        alongside = gate.render()
    finally:
        gate.close()
        watching.close()

    assert numpy.array_equal(clean, alongside), "a marker reached the gate"
    assert marked_view is not None


def test_drawing_a_marker_adds_nothing_to_the_model() -> None:
    """AC-MARK-10: drawing a marker adds nothing to the model.

    A marker lives in the scene the renderer rebuilds every frame, so it is
    invisible to physics, to contacts, and to every camera that renders
    without it.
    """
    mujoco = pytest.importorskip("mujoco")
    model, data = world()
    mujoco.mj_forward(model, data)
    before = (model.ngeom, model.nbody, data.ncon)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data, camera="watch")
        draw(renderer.scene, markers_for((record(),), effector(), 0.9))
        renderer.render()
    finally:
        renderer.close()
    mujoco.mj_forward(model, data)
    assert (model.ngeom, model.nbody, data.ncon) == before


def test_update_scene_clears_the_markers_of_the_frame_before() -> None:
    """A marker does not survive into the next frame's scene.

    Markers are pushed after `update_scene` and the next call rebuilds the
    scene from the model, so they cannot accumulate and a run that stops
    drawing them stops showing them.
    """
    mujoco = pytest.importorskip("mujoco")
    model, data = world()
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data, camera="watch")
        plain = renderer.scene.ngeom
        draw(renderer.scene, markers_for((record(),), effector(), 0.9))
        assert renderer.scene.ngeom > plain
        renderer.update_scene(data, camera="watch")
        assert renderer.scene.ngeom == plain
    finally:
        renderer.close()


def test_no_track_is_ever_given_red() -> None:
    """No track is ever given red.

    Red belongs to the park pose, which is the one marker in the scene that is
    not a track. The palette is dense, so this holds because a wedge of the
    hue circle is reserved rather than because the first few identities happen
    to miss it.
    """
    from clave.tracker.markers import RESERVED_HUE

    red = (1.0, 0.0, 0.0)
    assert RESERVED_HUE > 0.0
    for track_id in range(512):
        assert math.dist(color_for(track_id), red) > 0.25, track_id
