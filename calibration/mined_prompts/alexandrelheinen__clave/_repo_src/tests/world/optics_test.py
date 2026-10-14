"""The field of view a camera's parts produce, and the belt they have to cover.

A camera names a sensor and a lens rather than an angle, because an angle
written directly is a number nobody can check against a catalog. These tests
hold the derivation to the arithmetic every optics table uses, and hold the
shipped line to covering the belt it stands over.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from clave.world.config import WorldConfigError, load, require, require_range
from clave.world.scene import field_of_view, sensor_span_millimeters

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"

SENSORS = {"imx264": {"pixels": [2448, 2048], "pixel_pitch_micrometers": 3.45}}
LENSES = {"eight": {"focal_length_millimeters": 8.0}}


def camera(**overrides: object) -> dict[str, object]:
    """A camera entry, with the shipped parts unless overridden."""
    entry: dict[str, object] = {
        "id": "probe",
        "role": "detection",
        "sensor": "imx264",
        "lens": "eight",
        "long_axis": "across",
        "position_meters": [0.0, 0.0, 2.0],
    }
    entry.update(overrides)
    return entry


def test_the_long_sensor_axis_spans_whichever_belt_axis_it_is_told_to() -> None:
    """The IMX264 is 8.446 mm by 7.066 mm, and which side faces the belt width
    is the cheapest parameter on the page."""
    across, along = sensor_span_millimeters(camera(), SENSORS, LENSES)
    assert across == pytest.approx(2448 * 0.00345)
    assert along == pytest.approx(2048 * 0.00345)

    turned_across, turned_along = sensor_span_millimeters(
        camera(long_axis="along"), SENSORS, LENSES
    )
    assert (turned_across, turned_along) == (along, across)


def test_the_field_of_view_is_derived_from_the_sensor_and_the_lens() -> None:
    """MuJoCo's fovy is vertical, so it is the across-belt angle."""
    expected = 2.0 * math.degrees(math.atan(2448 * 0.00345 / (2.0 * 8.0)))
    assert field_of_view(camera(), SENSORS, LENSES) == pytest.approx(expected)


def test_turning_the_camera_narrows_the_across_belt_angle() -> None:
    """Laying the long axis along travel spends it on the wrong direction."""
    assert field_of_view(camera(long_axis="along"), SENSORS, LENSES) < field_of_view(
        camera(), SENSORS, LENSES
    )


def test_a_camera_naming_a_part_no_catalog_declares_is_refused() -> None:
    """A lens nobody can order is worse than no lens."""
    with pytest.raises(WorldConfigError, match="sensors"):
        field_of_view(camera(sensor="imaginary"), SENSORS, LENSES)
    with pytest.raises(WorldConfigError, match="lenses"):
        field_of_view(camera(lens="imaginary"), SENSORS, LENSES)


def test_a_camera_that_spans_neither_belt_axis_is_refused() -> None:
    """`long_axis` is across or along; there is no third direction."""
    with pytest.raises(WorldConfigError, match="long_axis"):
        field_of_view(camera(long_axis="sideways"), SENSORS, LENSES)


def test_a_lens_of_no_focal_length_forms_no_image() -> None:
    """A zero would divide, and a negative would report a mirror."""
    with pytest.raises(WorldConfigError, match="focal length"):
        field_of_view(camera(), SENSORS, {"eight": {"focal_length_millimeters": 0.0}})


def test_every_configured_camera_covers_the_whole_belt_width() -> None:
    """The gate stands over the belt, so it has to see all of it.

    This is the criterion the shipped line failed before the optics were
    grounded in real parts: `gate_wide` saw 0.800 m of a 1.00 m belt while
    objects spawned out to 0.42 m, so the detection camera could not see every
    object it exists to detect.

    The across-belt extent depends only on `fovy` and the standoff, never on
    the aspect anything renders at, which is what makes this arithmetic worth
    asserting rather than only sweeping.
    """
    raw = load(CONFIG)
    sensors, lenses = require(raw, "sensors"), require(raw, "lenses")
    surface = float(require(require(raw, "belt"), "surface_height_meters", "belt"))
    width = float(require(require(raw, "belt"), "width_meters", "belt"))

    spans = []
    for entry in require(raw, "cameras"):
        standoff = float(entry["position_meters"][2]) - surface
        assert standoff > 0.0, entry["id"]
        fovy = field_of_view(entry, sensors, lenses)
        across = 2.0 * standoff * math.tan(math.radians(fovy) / 2.0)
        spans.append((str(entry["id"]), str(entry["role"]), across, standoff))

    detection = [s for s in spans if s[1] == "detection"]
    assert detection, "the line declares a detection camera"
    for name, _, across, _standoff in detection:
        assert across >= width, f"{name} sees {across:.3f} m of a {width} m belt"


def test_the_code_cameras_tile_the_belt_rather_than_sample_it() -> None:
    """Three narrow cameras have to overlap, not sit edge to edge.

    They were spaced 0.342 m apart while each saw 0.192 m across, which left a
    dead band on each side of the belt where no barcode could be read at all.
    """
    raw = load(CONFIG)
    sensors, lenses = require(raw, "sensors"), require(raw, "lenses")
    surface = float(require(require(raw, "belt"), "surface_height_meters", "belt"))
    half = float(require(require(raw, "belt"), "width_meters", "belt")) / 2.0

    bands = []
    for entry in require(raw, "cameras"):
        if str(entry["role"]) != "code":
            continue
        standoff = float(entry["position_meters"][2]) - surface
        fovy = field_of_view(entry, sensors, lenses)
        reach = standoff * math.tan(math.radians(fovy) / 2.0)
        center = float(entry["position_meters"][1])
        bands.append((center - reach, center + reach))

    assert len(bands) >= 2, "one code camera cannot tile anything"
    bands.sort()
    assert bands[0][0] <= -half, f"the belt edge at {-half} is outside {bands[0]}"
    assert bands[-1][1] >= half, f"the belt edge at {half} is outside {bands[-1]}"
    for lower, upper in zip(bands, bands[1:], strict=False):
        assert lower[1] >= upper[0], f"a dead band between {lower} and {upper}"


def test_the_code_cameras_resolve_a_barcode_module() -> None:
    """An EAN-13 narrow module is about 0.33 mm and decoding wants two pixels
    across it, so 0.165 mm per pixel is the floor rather than a target."""
    raw = load(CONFIG)
    sensors, lenses = require(raw, "sensors"), require(raw, "lenses")
    surface = float(require(require(raw, "belt"), "surface_height_meters", "belt"))

    for entry in require(raw, "cameras"):
        if str(entry["role"]) != "code":
            continue
        standoff = float(entry["position_meters"][2]) - surface
        fovy = field_of_view(entry, sensors, lenses)
        across = 2.0 * standoff * math.tan(math.radians(fovy) / 2.0)
        pixels = max(require(sensors[str(entry["sensor"])], "pixels", "sensors"))
        assert across * 1000.0 / pixels <= 0.165, entry["id"]


def test_the_detection_camera_sees_every_lateral_position_an_object_spawns_at() -> None:
    """Coverage is not enough if the objects sit outside it."""
    raw = load(CONFIG)
    sensors, lenses = require(raw, "sensors"), require(raw, "lenses")
    surface = float(require(require(raw, "belt"), "surface_height_meters", "belt"))
    lateral = require_range(require(raw, "spawn"), "lateral_offset_meters", "spawn")

    wide = next(e for e in require(raw, "cameras") if str(e["role"]) == "detection")
    standoff = float(wide["position_meters"][2]) - surface
    reach = standoff * math.tan(
        math.radians(field_of_view(wide, sensors, lenses)) / 2.0
    )
    center = float(wide["position_meters"][1])
    assert center - reach <= lateral.low
    assert center + reach >= lateral.high
