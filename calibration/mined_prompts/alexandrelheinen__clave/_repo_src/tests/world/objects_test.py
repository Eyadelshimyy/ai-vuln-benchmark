"""Tests for the object set and its material classes."""

from pathlib import Path

import pytest

from clave.taxonomy import BY_ID
from clave.world.config import WorldConfigError, load
from clave.world.objects import ObjectSpec, channels, material_classes, parse

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"


def shipped() -> tuple[ObjectSpec, ...]:
    """Parse the object set the world actually ships with."""

    return parse(load(CONFIG)["objects"])


def test_every_object_carries_a_taxonomy_material_class() -> None:
    """A rollout is labeled by construction."""
    assert all(spec.material_class in BY_ID for spec in shipped())


def test_the_object_set_covers_the_classes_a_real_object_exists_for() -> None:
    """The object set covers the classes a real object exists for.

    The criterion asks for six classes. The set covers four, because every
    object is now a scanned package and nothing made of the missing six fits a
    55.7 mm gripper: a soda can is 66 mm across and no glass container in
    either pinned collection is under 52 mm.

    The assertion is exact rather than a floor, so growing the set past four
    classes fails here and forces the trade to be revisited.
    """
    assert set(material_classes(shipped())) == {"M-02", "M-04", "M-06", "M-09"}


def test_every_object_is_a_scanned_package_rather_than_a_primitive() -> None:
    """A colored cylinder teaches shape, which is what this removed."""
    assert all(spec.is_mesh for spec in shipped())


def test_an_unknown_material_class_is_rejected_naming_the_object() -> None:
    """A mislabeled object would corrupt every rollout."""
    entry = [
        {
            "name": "mystery",
            "material_class": "M-99",
            "shape": "box",
            "size_a_meters": [0.1, 0.2],
            "size_b_meters": [0.1, 0.2],
            "density_kg_per_m3": [10, 20],
        }
    ]
    with pytest.raises(WorldConfigError, match="mystery"):
        parse(entry)


def test_an_unknown_shape_is_rejected_naming_the_object() -> None:
    """Only the two primitives the scene can build are accepted."""
    entry = [
        {
            "name": "blob",
            "material_class": "M-01",
            "shape": "torus",
            "size_a_meters": [0.1, 0.2],
            "size_b_meters": [0.1, 0.2],
            "density_kg_per_m3": [10, 20],
        }
    ]
    with pytest.raises(WorldConfigError, match="blob"):
        parse(entry)


def test_channels_are_deduplicated_and_ordered() -> None:
    """Bin layout must not depend on object declaration order."""
    resolved = channels(shipped())
    assert resolved == tuple(sorted(set(resolved)))
    assert "CH-FIBER" in resolved
