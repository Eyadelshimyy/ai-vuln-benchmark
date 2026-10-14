"""The waste object set.

Every object carries a material class identifier from the taxonomy, so a rollout
recorded in this world is labeled by construction rather than by a labeling pass
afterwards.

Objects are parametric primitives sized from their real counterparts, not
scanned meshes. That gives correct mass, footprint and grasp width for belt
dynamics, and gives nothing at all for appearance. A perception model trained on
these frames alone would learn shape, which is a limitation this project records
rather than hides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clave.taxonomy import BY_ID, channel_of
from clave.world.config import Range, require, require_range


@dataclass(frozen=True)
class ObjectSpec:
    """One kind of waste object that can appear on the belt.

    Attributes:
        name: Identifier used in the scene and in recorded rollouts.
        material_class: Taxonomy identifier of the form `M-NN`.
        shape: `cylinder`, `box`, or `mesh` for a scanned object.
        size: Half-extents range, interpreted per shape. Ignored for a mesh,
            which carries its own size.
        density: Density range in kilograms per cubic meter.
        mesh: Path to the mesh, relative to the repository root, when the shape
            is a mesh.
        texture: Path to its texture, when it has one.
    """

    name: str
    material_class: str
    shape: str
    size: tuple[Range, Range]
    density: Range
    mesh: str | None = None
    texture: str | None = None

    @property
    def is_mesh(self) -> bool:
        """Whether this object is a scanned mesh rather than a primitive."""
        return self.shape == "mesh"

    @property
    def channel(self) -> str:
        """The channel this object routes to by default."""
        return channel_of(self.material_class)

    @property
    def max_grasp_width(self) -> float:
        """The widest the gripper has to open for this object, in meters.

        A primitive is grasped across `size_a`: a cylinder by its diameter and
        a box by its width. The largest value the range can draw is what the
        gripper has to clear, because a randomized world draws it eventually.

        A mesh carries its own size, so this returns zero and the scene checks
        the compiled vertices instead. Declaring a mesh's width here would be a
        number that can drift from the file it describes.
        """
        if self.is_mesh:
            return 0.0
        return 2.0 * self.size[0].high


def parse(
    entries: list[dict[str, Any]], max_grasp_width: float | None = None
) -> tuple[ObjectSpec, ...]:
    """Read the object set from configuration.

    Args:
        entries: The `objects` list from the world configuration.
        max_grasp_width: The widest object the gripper can close on, in meters.
            When given, an object wider than this is refused.

    Returns:
        The parsed object specs.

    Raises:
        WorldConfigError: If a key is missing, a range is invalid, a material
            class is not in the taxonomy, or an object is wider than the
            gripper can open. An unknown class names the offending object,
            because a mislabeled object silently corrupts every rollout
            recorded from this world.
    """
    from clave.world.config import WorldConfigError

    specs: list[ObjectSpec] = []
    for index, entry in enumerate(entries):
        where = f"objects[{index}]"
        name = require(entry, "name", where)
        material_class = require(entry, "material_class", f"objects.{name}")
        if material_class not in BY_ID:
            raise WorldConfigError(
                f"object {name!r} declares material class {material_class!r}, "
                f"which is not in the taxonomy"
            )
        shape = require(entry, "shape", f"objects.{name}")
        if shape not in ("cylinder", "box", "mesh"):
            raise WorldConfigError(
                f"object {name!r} declares shape {shape!r}; "
                f"expected 'cylinder', 'box' or 'mesh'"
            )
        if shape == "mesh":
            zero = Range(0.0, 0.0)
            size = (zero, zero)
            mesh: str | None = str(require(entry, "mesh", f"objects.{name}"))
            texture: str | None = entry.get("texture")
        else:
            size = (
                require_range(entry, "size_a_meters", f"objects.{name}"),
                require_range(entry, "size_b_meters", f"objects.{name}"),
            )
            mesh = texture = None
        specs.append(
            ObjectSpec(
                name=name,
                material_class=material_class,
                shape=shape,
                size=size,
                density=require_range(entry, "density_kg_per_m3", f"objects.{name}"),
                mesh=mesh,
                texture=None if texture is None else str(texture),
            )
        )
    if max_grasp_width is not None:
        _check_graspable(specs, max_grasp_width)
    return tuple(specs)


def _check_graspable(specs: list[ObjectSpec], limit: float) -> None:
    """Refuse an object the gripper cannot close on.

    A world that spawns objects wider than its own gripper simulates picking
    that could never happen, and every number recorded from it describes a task
    the manipulator was never able to perform. The check is here rather than in
    a review comment because the sizes are randomized and a reviewer reads the
    range rather than the draw.

    Args:
        specs: The parsed objects.
        limit: The widest object the gripper can close on, in meters.

    Raises:
        WorldConfigError: Naming the object, its width and the limit.
    """
    from clave.world.config import WorldConfigError

    for spec in specs:
        if spec.max_grasp_width > limit:
            raise WorldConfigError(
                f"object {spec.name!r} can be drawn {spec.max_grasp_width * 1000:.1f} "
                f"mm wide, and the gripper opens {limit * 1000:.1f} mm. A world "
                f"that spawns objects it cannot grasp measures a task the arm "
                f"cannot perform."
            )


def channels(specs: tuple[ObjectSpec, ...]) -> tuple[str, ...]:
    """List the distinct channels an object set needs, in a stable order.

    Args:
        specs: The object set.

    Returns:
        Channel identifiers, deduplicated and sorted, so the bin layout does not
        depend on the order objects happen to be declared in.
    """
    return tuple(sorted({spec.channel for spec in specs}))


def material_classes(specs: tuple[ObjectSpec, ...]) -> tuple[str, ...]:
    """List the distinct material classes an object set covers."""
    return tuple(sorted({spec.material_class for spec in specs}))
