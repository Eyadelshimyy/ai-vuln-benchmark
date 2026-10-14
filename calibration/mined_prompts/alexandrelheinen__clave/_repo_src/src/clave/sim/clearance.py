"""Jaw clearance measured from the compiled gripper.

The debug run reports how close the jaw came to the belt. The measurement
reads the collision geoms bolted to the flange, not the flange pose, because
the pads hang below the site the arm is commanded to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class JawState:
    """Where the gripper's own geometry stands relative to the belt.

    Attributes:
        lowest_z: The lowest world height of any collision geometry bolted to
            the flange, in meters. This is the jaw rather than the pose: the
            pose is the flange, and the jaw hangs below it.
        clearance: That height less the belt surface, in meters. Negative means
            the geometry is inside the belt.
        contact: Whether any of that geometry is touching the belt.
        tilt_degrees: How far the tool's own axis stands from the belt normal,
            in degrees.
    """

    lowest_z: float
    clearance: float
    contact: bool
    tilt_degrees: float


def _jaw_collision_geoms(model: Any, arm: Any) -> tuple[int, ...]:
    """Return every collision geom bolted to the flange.

    Found by walking down the body tree from the body the pinch site stands on
    rather than by matching names, so a vendored gripper that renames its parts
    still reports the geometry that would hit the belt.

    Args:
        model: The compiled model.
        arm: The arm indices.

    Returns:
        The geom ids, in model order.
    """
    base = int(model.site_bodyid[arm.pinch_site])
    children: dict[int, list[int]] = {}
    for body in range(model.nbody):
        children.setdefault(int(model.body_parentid[body]), []).append(body)
    subtree: set[int] = set()
    stack = [base]
    while stack:
        body = stack.pop()
        subtree.add(body)
        stack.extend(children.get(body, ()))
    return tuple(
        geom
        for geom in range(model.ngeom)
        if int(model.geom_bodyid[geom]) in subtree and model.geom_contype[geom] != 0
    )


_MESH_VERTS: dict[tuple[int, int], NDArray[np.float64]] = {}
"""Mesh vertices copied once per compiled model, keyed by model id and mesh id.

Used when a marker or resting height needs the mesh itself. Jaw clearance no
longer walks these vertices (`AC-PERF-03`).
"""


def _mesh_vertices(model: Any, mesh: int) -> NDArray[np.float64]:
    """Return one mesh's vertices, copied out of the model the first time.

    Args:
        model: The compiled model.
        mesh: The mesh id.

    Returns:
        The vertices, shape `(count, 3)`, in the mesh frame.
    """
    key = (id(model), mesh)
    cached = _MESH_VERTS.get(key)
    if cached is None:
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        cached = np.array(model.mesh_vert[start : start + count], dtype=np.float64)
        _MESH_VERTS[key] = cached
    return cached


def _aabb_support_down(rotation: Any, centre: float, size: Any) -> float:
    """Return the lowest world height of a local AABB under a rotation.

    The support of an axis-aligned box along world-down is each half-extent
    times how much of that local axis points down. Written out rather than as
    a matrix product because this runs once per geom per physics tick.

    Args:
        rotation: The geom's 3x3 world rotation.
        centre: The geom origin's world height, in meters.
        size: Local half-extents along x, y, and z.

    Returns:
        The lowest world height of that box, in meters.
    """
    return centre - (
        abs(float(rotation[2, 0])) * float(size[0])
        + abs(float(rotation[2, 1])) * float(size[1])
        + abs(float(rotation[2, 2])) * float(size[2])
    )


def _lowest_world_z(model: Any, data: Any, geom: int) -> float:
    """Return the lowest height of one collision geom's own volume.

    Exact for the primitive shapes a gripper pad is made of. Mesh geoms use
    MuJoCo's local AABB (`geom_size` half-extents) under the same support
    formula as a box (`AC-PERF-03`): that bound never sits above the true
    mesh, so a clearance report never overstates how close the jaw came, and
    it stays O(1) where walking tens of thousands of vertices per tick was
    most of the debug-run wall.

    Args:
        model: The compiled model.
        data: Its state, with forward kinematics current.
        geom: The geom to measure.

    Returns:
        The lowest world height of that geom, in meters.
    """
    import mujoco

    rotation = data.geom_xmat[geom].reshape(3, 3)
    centre = float(data.geom_xpos[geom][2])
    size = model.geom_size[geom]
    kind = int(model.geom_type[geom])
    if kind in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH)):
        return _aabb_support_down(rotation, centre, size)
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return centre - float(size[0])
    if kind in (
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    ):
        return centre - abs(float(rotation[2, 2])) * float(size[1]) - float(size[0])
    return centre - float(max(size))


def _jaw_state(
    mujoco: Any,
    model: Any,
    data: Any,
    arm: Any,
    jaw_geoms: tuple[int, ...],
    belt_geom: int,
    surface: float,
) -> JawState:
    """Read the jaw's clearance, its contact with the belt and its tilt.

    Args:
        mujoco: The imported MuJoCo module.
        model: The compiled model.
        data: Its state, with forward kinematics current.
        arm: The arm indices.
        jaw_geoms: The collision geoms bolted to the flange.
        belt_geom: The belt's geom id, or -1 when the world names none.
        surface: Belt surface height, in world frame meters.

    Returns:
        The state.
    """

    lowest = min(
        (_lowest_world_z(model, data, geom) for geom in jaw_geoms),
        default=float("inf"),
    )
    jaw = set(jaw_geoms)
    touching = False
    for index in range(data.ncon):
        pair = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        # Either order: MuJoCo reports the pair in whichever order its broad
        # phase produced, and a test that assumed the jaw came first missed
        # every contact the run was making.
        if belt_geom in pair and pair & jaw:
            touching = True
            break
    axis = data.site_xmat[arm.tool_site].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(axis[2])))))
    return JawState(
        lowest_z=lowest,
        clearance=lowest - surface,
        contact=touching,
        tilt_degrees=tilt,
    )
