"""Tests for world configuration loading."""

from pathlib import Path

import pytest

from clave.world.config import WorldConfigError, load, require, require_range

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"


def test_the_shipped_configuration_loads() -> None:
    """The world reads its tunables from a file."""

    raw = load(CONFIG)
    assert "belt" in raw
    assert "objects" in raw


def test_a_missing_key_fails_naming_its_location() -> None:
    """Nothing is substituted for an absent key."""
    with pytest.raises(WorldConfigError, match="belt.speed_meters_per_second"):
        require({}, "speed_meters_per_second", "belt")


def test_an_inverted_range_fails_naming_the_key() -> None:
    """A range whose bounds are the wrong way round is a defect."""
    with pytest.raises(WorldConfigError, match="inverted"):
        require_range({"speed": [0.4, 0.1]}, "speed", "belt")


def test_a_scalar_where_a_range_belongs_is_rejected() -> None:
    """A randomizable quantity must be a range."""
    with pytest.raises(WorldConfigError, match="two-element range"):
        require_range({"speed": 0.2}, "speed", "belt")


def test_every_randomizable_belt_and_spawn_value_is_a_range() -> None:
    """Every randomizable belt and spawn value is a range, in the shipped
    configuration rather than a fixture."""
    raw = load(CONFIG)
    require_range(raw["belt"], "speed_meters_per_second", "belt")
    require_range(raw["spawn"], "spacing_meters", "spawn")
    require_range(raw["spawn"], "lateral_offset_meters", "spawn")
    # Cameras are fixed installations rather than randomized quantities: a
    # sensor that moves between runs cannot be calibrated, so each entry
    # carries scalars and the list itself is what varies between lines.
    assert raw["cameras"], "the shipped line declares at least one sensor"
    for sensor in raw["cameras"]:
        assert {
            "id",
            "role",
            "position_meters",
            "sensor",
            "lens",
            "long_axis",
        } <= set(sensor)
        # A camera names the parts it is built from. An angle written directly
        # is a number nobody can check against a catalog, so `fovy` is derived
        # and must not appear here.
        assert "fovy_degrees" not in sensor, sensor["id"]
        assert sensor["sensor"] in raw["sensors"]
        assert sensor["lens"] in raw["lenses"]


def test_an_object_wider_than_the_gripper_is_refused() -> None:
    """A world that spawns what it cannot grasp measures an impossible task.

    Guards the defect v1.0.1 fixed: every object in the shipped set was wider
    than the gripper's 55.7 mm opening, and nothing said so.
    """
    from clave.world.config import WorldConfigError
    from clave.world.objects import parse

    entry = {
        "name": "boulder",
        "material_class": "M-01",
        "shape": "cylinder",
        "size_a_meters": [0.030, 0.038],
        "size_b_meters": [0.090, 0.120],
        "density_kg_per_m3": [55, 110],
    }
    with pytest.raises(WorldConfigError, match="boulder"):
        parse([entry], 0.045)
    # The same object is accepted when the gripper is wide enough for it.
    assert parse([entry], 0.080)[0].max_grasp_width == pytest.approx(0.076)


def test_every_shipped_object_fits_the_shipped_gripper() -> None:
    """The committed world builds, which means every object clears the bound."""
    from clave.world.objects import parse

    raw = load(CONFIG)
    limit = float(require(require(raw, "arm"), "max_grasp_width_meters"))
    specs = parse(require(raw, "objects"), limit)
    assert specs
    assert all(spec.max_grasp_width <= limit for spec in specs)


def test_a_mesh_object_declares_no_size_and_is_checked_against_the_file() -> None:
    """A number written here could drift from the mesh it describes.

    Guards the YCB objects added at v1.0.3: their width comes from the compiled
    vertices, checked in the scene, rather than from a value in configuration.
    """
    from clave.world.objects import parse

    entry = {
        "name": "scanned",
        "material_class": "M-09",
        "shape": "mesh",
        "mesh": "third_party/ycb_sim/meshes/009_gelatin_box.msh",
        "density_kg_per_m3": [90, 160],
    }
    spec = parse([entry], 0.050)[0]
    assert spec.is_mesh
    assert spec.mesh is not None
    # Zero rather than a guess, so the primitive check cannot pass judgment on
    # a file it has not read.
    assert spec.max_grasp_width == 0.0


def test_a_mesh_object_must_name_its_mesh() -> None:
    """A mesh with no file is a shape nothing can draw."""
    from clave.world.config import WorldConfigError
    from clave.world.objects import parse

    with pytest.raises(WorldConfigError, match="mesh"):
        parse(
            [
                {
                    "name": "scanned",
                    "material_class": "M-09",
                    "shape": "mesh",
                    "density_kg_per_m3": [90, 160],
                }
            ]
        )


def test_every_shipped_mesh_object_is_checked_out() -> None:
    """The meshes the world names have to be on disk for it to build."""
    from clave.world.objects import parse

    raw = load(CONFIG)
    specs = parse(require(raw, "objects"))
    meshes = [spec for spec in specs if spec.is_mesh]
    assert meshes, "the world declares no scanned objects"
    for spec in meshes:
        assert spec.mesh is not None
        assert (ROOT / spec.mesh).is_file(), f"{spec.mesh} is not checked out"
