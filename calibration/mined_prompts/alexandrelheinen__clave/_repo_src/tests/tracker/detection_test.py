"""Turning pixels into a footprint, and throwing away what the render leaks.



The trap this file guards is the render leaking identity. A MuJoCo
segmentation render is keyed by geometry name, that name is `object_<slot>`,
and `clave.world.belt` makes the slot the same integer the world hands out as
`ObjectLabel.object_id`.
An adapter that passed the key through would hand the tracker the answer while
appearing to perceive it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from clave.tracker.adapters.detection import (
    DetectionError,
    detections_from_masks,
    oriented_extent,
    to_belt,
)
from clave.tracker.belt_frame import Footprint, NadirOptics
from clave.tracker.evidence import Detection, Evidence, PixelMask, Role

WIDE = NadirOptics(camera_height=1.942, fovy_degrees=55.65)
CAMERA = np.asarray((-1.0, 0.0, 1.942), dtype=np.float64)
SURFACE = 0.90


def footprint_of(reading: Evidence) -> Footprint:
    """The footprint of a detection reading, narrowed from the payload union."""
    payload = reading.payload
    assert isinstance(payload, Detection)
    return payload.footprint


def height_of(reading: Evidence) -> float | None:
    """The height of a detection reading, narrowed from the payload union."""
    payload = reading.payload
    assert isinstance(payload, Detection)
    return payload.height


def square(row: int, column: int, side: int, size: int = 480) -> PixelMask:
    """A filled square of pixels."""
    return PixelMask(
        width=size,
        height=size,
        runs=tuple((row + offset, column, side) for offset in range(side)),
    )


def test_a_pixel_at_the_image_centre_is_under_the_camera() -> None:
    """A pixel at the image centre is under the camera."""
    x, y = to_belt(320.0, 240.0, CAMERA, WIDE, SURFACE, (640, 480))
    assert x == pytest.approx(CAMERA[0], abs=1e-9)
    assert y == pytest.approx(CAMERA[1], abs=1e-9)


def test_the_column_axis_runs_along_belt_travel() -> None:
    """The column axis runs along belt travel.

    Measured rather than reasoned about: placing an object 0.20 m downstream
    moved its bounds 96 columns at a 640 by 480 render, and nothing the other
    way.
    """
    metres = WIDE.meters_per_pixel(SURFACE, 480)
    x, y = to_belt(320.0 + 96.0, 240.0, CAMERA, WIDE, SURFACE, (640, 480))
    assert x == pytest.approx(CAMERA[0] + 96.0 * metres)
    assert y == pytest.approx(CAMERA[1])


def test_the_row_axis_runs_across_the_belt_and_is_inverted() -> None:
    """The row axis runs across the belt and is inverted.

    Row zero is the top of the image, which is the far side of the belt, so a
    larger row is a smaller `y`. Getting this backwards would mirror every
    footprint about the centreline and nothing would say so.
    """
    metres = WIDE.meters_per_pixel(SURFACE, 480)
    _, near = to_belt(320.0, 240.0 + 96.0, CAMERA, WIDE, SURFACE, (640, 480))
    _, far = to_belt(320.0, 240.0 - 96.0, CAMERA, WIDE, SURFACE, (640, 480))
    assert near == pytest.approx(CAMERA[1] - 96.0 * metres)
    assert far == pytest.approx(CAMERA[1] + 96.0 * metres)
    assert near < far


def test_a_square_mask_has_no_orientation_to_state() -> None:
    """A square mask has no orientation to state.

    A can is circular in plan under a nadir camera, and a confident random yaw
    on a circular footprint is worse than an absent one because the safety
    layer acts on it.
    """
    major, minor, yaw, oriented = oriented_extent(square(100, 100, 40))
    assert major == pytest.approx(minor, rel=0.05)
    assert not oriented
    assert yaw == pytest.approx(0.0)


def test_an_elongated_mask_states_its_major_axis() -> None:
    """An elongated mask states its major axis."""
    mask = PixelMask(
        width=480, height=480, runs=tuple((100 + row, 100, 80) for row in range(10))
    )
    major, minor, yaw, oriented = oriented_extent(mask)
    assert major > minor
    assert oriented
    # The long side runs along columns, which is the image x axis, so the
    # second moment puts the major axis there.
    assert abs(math.sin(yaw)) < 0.2


def test_a_diagonal_mask_reports_a_yaw_between_the_axes() -> None:
    """A diagonal mask reports a yaw between the axes."""
    runs = tuple((100 + step, 100 + step, 6) for step in range(60))
    major, minor, yaw, oriented = oriented_extent(PixelMask(480, 480, runs))
    assert major > minor
    assert oriented
    assert 0.2 < abs(yaw) < math.pi / 2 - 0.2


def test_the_adapter_produces_one_reading_per_mask() -> None:
    """The adapter produces one reading per mask."""
    masks = {"object_3": square(100, 100, 40), "object_7": square(300, 300, 30)}
    readings = detections_from_masks(
        masks,
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE,
        render=(480, 480),
    )
    assert len(readings) == 2
    assert {reading.role for reading in readings} == {Role.DETECTION}
    assert all(reading.source_id == "gate_wide" for reading in readings)


def test_the_adapter_discards_the_key_the_render_was_indexed_by() -> None:
    """The adapter discards the key the render was indexed by.

    The trap. `object_3` is the simulator's `object_id` for that object, so a
    reading carrying it anywhere would be a reading carrying the answer.
    """
    masks = {"object_3": square(100, 100, 40)}
    readings = detections_from_masks(
        masks,
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE,
        render=(480, 480),
    )
    rendered = repr(readings)
    assert "object_3" not in rendered
    assert "object_" not in rendered


def test_no_field_of_a_reading_is_derivable_from_the_simulator_id() -> None:
    """No field of a reading is derivable from the simulator id.

    Stronger than checking a string: two objects whose only difference is the
    key they were rendered under produce identical readings.
    """
    shared = square(100, 100, 40)
    first = detections_from_masks(
        {"object_3": shared},
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE,
        render=(480, 480),
    )
    second = detections_from_masks(
        {"object_19": shared},
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE,
        render=(480, 480),
    )
    assert first == second


def test_the_belt_frame_centre_is_the_same_at_two_render_sizes() -> None:
    """The belt frame centre is the same at two render sizes.

    The resolution-invariance proof. A leaked pixel coordinate, or a scale
    factor fitted to one render and applied to another, fails here. Both are
    mistakes that otherwise surface as a systematically wrong footprint that
    every test agrees with.
    """
    coarse = detections_from_masks(
        {"object_0": square(100, 100, 40, size=480)},
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE,
        render=(480, 480),
    )[0]
    fine = detections_from_masks(
        {"object_0": square(200, 200, 80, size=960)},
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE,
        render=(960, 960),
    )[0]
    assert footprint_of(fine).center[0] == pytest.approx(
        footprint_of(coarse).center[0], abs=1e-3
    )
    assert footprint_of(fine).center[1] == pytest.approx(
        footprint_of(coarse).center[1], abs=1e-3
    )
    assert footprint_of(fine).major_extent == pytest.approx(
        footprint_of(coarse).major_extent, abs=1e-3
    )


def test_a_taller_object_resolves_finer_and_the_adapter_says_so() -> None:
    """A taller object resolves finer and the adapter says so.

    Scaling at the belt plane inflates a tall object's footprint. The same mask
    at a greater height covers less belt, so the footprint has to shrink.
    """
    mask = {"object_0": square(100, 100, 40)}
    at_belt = detections_from_masks(
        mask,
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE,
        render=(480, 480),
        belt_surface=SURFACE,
    )[0]
    tall = detections_from_masks(
        mask,
        source_id="gate_wide",
        observed_at_nanos=1_000,
        camera=CAMERA,
        optics=WIDE,
        surface_height=SURFACE + 0.15,
        render=(480, 480),
        belt_surface=SURFACE,
    )[0]
    assert footprint_of(tall).major_extent < footprint_of(at_belt).major_extent
    assert height_of(tall) == pytest.approx(0.15)


def test_a_camera_that_cannot_see_the_surface_is_refused() -> None:
    """A negative standoff yields a negative footprint."""
    with pytest.raises(DetectionError, match="standoff|above"):
        detections_from_masks(
            {"object_0": square(100, 100, 40)},
            source_id="gate_wide",
            observed_at_nanos=1_000,
            camera=CAMERA,
            optics=WIDE,
            surface_height=2.5,
            render=(480, 480),
        )


def test_nothing_in_the_package_imports_mujoco_at_module_scope() -> None:
    """Nothing in the package imports mujoco at module scope.

    The property worth guarding is that the package imports on a machine with
    no simulator, so the quality gate can read it without a render. Counting
    which files mention MuJoCo is a proxy for that and a worse one: it broke
    the moment a debug driver legitimately grew a simulator dependency, while
    the property itself never moved.

    So this asserts the property. Every module under `clave.tracker` imports,
    and any file naming MuJoCo does so inside a function.
    """
    import ast
    import importlib

    package = Path(__file__).resolve().parents[2] / "src" / "clave" / "tracker"
    for path in sorted(package.rglob("*.py")):
        module = (
            "clave.tracker."
            + path.relative_to(package).with_suffix("").as_posix().replace("/", ".")
        ).removesuffix(".__init__")
        importlib.import_module(module.removesuffix("."))

        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            if not any(name.split(".")[0] in {"mujoco", "cv2"} for name in names):
                continue
            assert node.col_offset > 0, (
                f"{path.name} imports a simulator or vision library at module "
                f"scope, so the package no longer imports without one"
            )


def test_the_adapter_finds_a_real_object_where_the_world_actually_put_it() -> None:
    """The adapter finds a real object where the world actually put it.

    Every other test in this file works from literals, which proves the
    arithmetic and not the axis mapping. This one renders the real gate camera,
    segments it, and asks whether the belt-frame centre the adapter computes is
    where the world actually placed the object.

    It is the test that catches a mirrored row axis, a transposed pair or a
    scale factor taken at the wrong plane, none of which a literal can catch
    because a literal agrees with whatever convention wrote it.
    """
    import os

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    from clave.tracker.adapters.render import segment_masks
    from clave.tracker.sensors import load_sensors, require_role
    from clave.world import config, scene

    root = Path(__file__).resolve().parents[2]
    raw = config.load(root / "configs" / "world" / "sorting_line.yml")
    model, data, _ = scene.build(raw, np.random.default_rng(0), root)
    wide = require_role(load_sensors(raw), Role.DETECTION)
    belt_surface = float(
        config.require(config.require(raw, "belt"), "surface_height_meters", "belt")
    )

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.enable_segmentation_rendering()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_0")
    address = model.jnt_qposadr[model.body_jntadr[body]]

    try:
        for placed_x, placed_y in ((-1.00, 0.00), (-0.80, 0.15), (-0.90, -0.30)):
            data.qpos[address : address + 3] = [
                placed_x,
                placed_y,
                belt_surface + 0.03,
            ]
            data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=wide.source_id)
            masks = segment_masks(model, renderer.render())
            assert masks, f"({placed_x}, {placed_y}) should be in frame"

            reading = detections_from_masks(
                masks,
                source_id=wide.source_id,
                observed_at_nanos=0,
                camera=wide.position,
                optics=wide.optics,
                # This package stands about 0.095 m, and the scale is taken at
                # the surface being imaged rather than at the belt.
                surface_height=belt_surface + 0.095,
                render=(640, 480),
                belt_surface=belt_surface,
            )[0]
            found_x, found_y, _ = footprint_of(reading).center
            assert found_x == pytest.approx(placed_x, abs=0.005)
            assert found_y == pytest.approx(placed_y, abs=0.005)
    finally:
        renderer.close()


def test_a_mask_too_small_to_have_a_size_is_dropped() -> None:
    """AC-TRACK-38: a mask too small to have a size is dropped.

    A sliver at the edge of the frame segments to a mask a pixel across,
    whose projected extent rounds to nothing. A footprint refuses to be
    built from that, and the refusal used to end the run: one pixel, three
    minutes of rollout gone.
    """
    sliver = PixelMask(width=640, height=480, runs=((10, 20, 1),))
    assert (
        detections_from_masks(
            {"1": sliver},
            source_id="gate_wide",
            observed_at_nanos=0,
            camera=np.asarray((0.0, 0.0, 1.942), dtype=np.float64),
            optics=WIDE,
            surface_height=0.95,
            render=(640, 480),
        )
        == ()
    )
