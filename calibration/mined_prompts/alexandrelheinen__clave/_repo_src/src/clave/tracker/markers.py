"""The grasp pose, as geometry in the world.

A marker is what the tracker believes, standing in the scene where it believes
it. One per open track, coloured by the track's identity, showing where a jaw
would close, how wide it would open, and which way it would be turned.

This is geometry and not an overlay, and the difference is the whole reason
the module exists. An overlay is painted onto a frame after the renderer is
done with it, so it argues for something the simulator never drew. A marker is
pushed into the scene before the render, so MuJoCo shades it, occludes it
behind the arm, and moves it with the camera exactly as it does every other
solid. `agents/claude.md` forbids the first and permits the second.

Two parts of the pose are derived and one is assumed. The horizontal position,
the closing axis and the opening all come from the record's footprint, which
came from pixels. The height does not: the gate looks straight down, no sensor
on this line estimates depth, and a `WasteObject` here carries no height at
all. So the pad plane is a configured standoff read from
[clave.world.effector.Effector], and this docstring is where that is admitted
rather than in a comment nobody reaches.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.taxonomy import REJECT_CHANNEL, channel_of
from clave.tracker.track import WasteObject
from clave.world.effector import Effector

SHAFT_RADIUS = 0.004
"""How thick the approach shaft is drawn, in meters.

A marker is drawn to scale wherever a dimension means something. This one does
not: no shaft exists on any effector, and it is here to show the approach
direction and to tie the pads to the flange. It is deliberately thin enough to
read as an annotation rather than as a part.
"""

UNORIENTED_PADS = 4
"""How many pads ring a footprint with no axis.

A circular footprint has no minor axis, so there is no one direction to turn a
jaw to. Ringing the object says the jaw could close along any diameter, which
is the claim the record actually supports. A single pair at the yaw that fell
out of the arithmetic would show a decision the tracker did not make, and a
filled disc of the same radius would hide the object a reader is checking the
marker against.
"""

UNREACHABLE_ALPHA = 0.35
"""How far a marker fades when the jaw cannot open that wide.

Fading rather than hiding: an object the effector cannot take is exactly the
one a reader wants to see, and a marker that vanishes looks like a tracker
that lost the object.
"""

OPAQUE = 1.0
"""Alpha for a pose the effector could take."""

PARK_POST_RADIUS = 0.010
"""How thick the post under the park pose is drawn, in meters."""

PARK_BALL_RADIUS = 0.030
"""How large the ball at the park pose is drawn, in meters."""


@dataclass(frozen=True)
class GraspMarker:
    """Where the effector would go for one track.

    Attributes:
        track_id: The identity this marker belongs to.
        valid_until_nanos: When belt travel invalidates this pose, copied from
            the record. A pose and the window it holds for travel together
            here for the same reason they travel together in the published
            decision: a pose with no expiry is a pose somebody will use late.
        pinch_position_belt: Where the pads would close, in belt frame meters.
        flange_position_world: Where the face the tool bolts to would sit.
        pad_positions_belt: The pad centers in belt frame, or an empty tuple when
            the footprint has no axis to turn a jaw to.
        pad_size: One pad's half-extents, in its own frame: along the closing
            direction, across it, and vertical.
        closing_yaw_belt: Rotation of the closing direction about the belt normal,
            in radians, or None when the footprint is not oriented.
        opening: How far the jaw would have to open, in meters.
        oriented: Whether `closing_yaw_belt` means anything.
        reachable: Whether the effector opens that wide at all.
        extent: The footprint's larger horizontal side, which sizes the disc
            drawn when there is no axis.
        color: The track's color, as red, green and blue in the unit range.
        channel: Where the object routes to, resolved from its material
            class. Carried here because the arm has to know which chute to
            release over, and the alternative is the task layer resolving a
            taxonomy identifier it has no other reason to hold.
        velocity_world: How the belt is carrying it, or None when nothing
            measured it. `AC-MOVE-46` predicts travel and drift with it.
        yaw_rate_belt: How fast it is turning about the belt normal, in radians
            per second, or None when nothing measured it. A grasp pose is a yaw
            claim made once per capture and the jaws close up to half a second
            later, so a tool sent to the claimed yaw arrives wherever the
            object has turned to since: the caller turns the tool to where the
            object *will* be, the same way it aims at where the object will be.
    """

    track_id: int
    valid_until_nanos: int
    pinch_position_belt: NDArray[np.float64]
    flange_position_world: NDArray[np.float64]
    pad_positions_belt: tuple[NDArray[np.float64], ...]
    pad_size: NDArray[np.float64]
    closing_yaw_belt: float | None
    opening: float
    oriented: bool
    reachable: bool
    extent: float
    color: tuple[float, float, float]
    channel: str = REJECT_CHANNEL
    velocity_world: NDArray[np.float64] | None = None
    yaw_rate_belt: float | None = None

    def __init__(
        self,
        track_id: int,
        valid_until_nanos: int,
        pinch_position_belt: NDArray[np.float64] | None = None,
        flange_position_world: NDArray[np.float64] | None = None,
        pad_positions_belt: tuple[NDArray[np.float64], ...] | None = None,
        pad_size: NDArray[np.float64] | None = None,
        closing_yaw_belt: float | None = None,
        opening: float = 0.0,
        oriented: bool = False,
        reachable: bool = False,
        extent: float = 0.0,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        channel: str = REJECT_CHANNEL,
        velocity_world: NDArray[np.float64] | None = None,
        yaw_rate_belt: float | None = None,
        *,
        grasp: NDArray[np.float64] | None = None,
        flange: NDArray[np.float64] | None = None,
        pads: tuple[NDArray[np.float64], ...] | None = None,
        closing_axis: float | None = None,
    ) -> None:
        p_grasp = grasp if grasp is not None else pinch_position_belt
        if p_grasp is None:
            raise TypeError("GraspMarker requires pinch_position_belt or grasp")
        p_flange = flange if flange is not None else flange_position_world
        if p_flange is None:
            raise TypeError("GraspMarker requires flange_position_world or flange")
        p_pads = (
            pads
            if pads is not None
            else (() if pad_positions_belt is None else pad_positions_belt)
        )
        axis = closing_axis if closing_axis is not None else closing_yaw_belt
        size = pad_size if pad_size is not None else np.zeros(3, dtype=np.float64)

        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "valid_until_nanos", valid_until_nanos)
        object.__setattr__(
            self, "pinch_position_belt", np.asarray(p_grasp, dtype=np.float64)
        )
        object.__setattr__(
            self, "flange_position_world", np.asarray(p_flange, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "pad_positions_belt",
            tuple(np.asarray(pad, dtype=np.float64) for pad in p_pads),
        )
        object.__setattr__(self, "pad_size", np.asarray(size, dtype=np.float64))
        object.__setattr__(self, "closing_yaw_belt", axis)
        object.__setattr__(self, "opening", opening)
        object.__setattr__(self, "oriented", oriented)
        object.__setattr__(self, "reachable", reachable)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(
            self,
            "velocity_world",
            None
            if velocity_world is None
            else np.asarray(velocity_world, dtype=np.float64),
        )
        object.__setattr__(self, "yaw_rate_belt", yaw_rate_belt)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraspMarker):
            return NotImplemented
        if self.velocity_world is None and other.velocity_world is None:
            velocity_match = True
        elif self.velocity_world is None or other.velocity_world is None:
            velocity_match = False
        else:
            velocity_match = bool(
                np.allclose(
                    self.velocity_world, other.velocity_world, rtol=0.0, atol=1e-12
                )
            )
        return (
            self.track_id == other.track_id
            and self.valid_until_nanos == other.valid_until_nanos
            and bool(
                np.allclose(
                    self.pinch_position_belt,
                    other.pinch_position_belt,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            and bool(
                np.allclose(
                    self.flange_position_world,
                    other.flange_position_world,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            and len(self.pad_positions_belt) == len(other.pad_positions_belt)
            and all(
                bool(np.allclose(left, right, rtol=0.0, atol=1e-12))
                for left, right in zip(
                    self.pad_positions_belt, other.pad_positions_belt, strict=True
                )
            )
            and bool(np.allclose(self.pad_size, other.pad_size, rtol=0.0, atol=1e-12))
            and self.closing_yaw_belt == other.closing_yaw_belt
            and self.opening == other.opening
            and self.oriented == other.oriented
            and self.reachable == other.reachable
            and self.extent == other.extent
            and self.color == other.color
            and self.channel == other.channel
            and velocity_match
            and self.yaw_rate_belt == other.yaw_rate_belt
        )

    @property
    def grasp(self) -> NDArray[np.float64]:
        """Backwards compatibility alias for pinch_position_belt."""
        return self.pinch_position_belt

    @property
    def flange(self) -> NDArray[np.float64]:
        """Backwards compatibility alias for flange_position_world."""
        return self.flange_position_world

    @property
    def closing_axis(self) -> float | None:
        """Backwards compatibility alias for closing_yaw_belt."""
        return self.closing_yaw_belt

    @property
    def pads(self) -> tuple[NDArray[np.float64], ...]:
        """Backwards compatibility alias for pad_positions_belt."""
        return self.pad_positions_belt


RESERVED_HUE = 0.05
"""The wedge at each end of the hue circle no track is given.

Red belongs to the park pose, which is the one marker in the scene that is not
a track, and a reader has to be able to tell it apart at a glance. The golden
ratio sequence is dense, so no amount of spreading keeps a track away from red
on its own: track 0 landed on pure red, and with enough tracks something lands
arbitrarily close to any hue. Reserving the wedge is what makes the guarantee
hold for every identity rather than for the first few.
"""


def color_for(track_id: int) -> tuple[float, float, float]:
    """Return a stable color for one track, as red, green and blue.

    Derived from the identity rather than assigned in order, so a track keeps
    its color when another retires, and a marker and a record correspond
    without a reader counting.

    Args:
        track_id: The track's identity.

    Returns:
        The color, each channel in the unit range, and never red.
    """
    # The golden ratio spreads successive integers around the hue circle, so
    # neighbouring track ids never come out as neighbouring colors. The span
    # is then squeezed off both ends, leaving red to the park pose.
    spread = (track_id * 0.618033988749895) % 1.0
    hue = RESERVED_HUE + spread * (1.0 - 2.0 * RESERVED_HUE)
    return colorsys.hsv_to_rgb(hue, 0.85, 1.0)


MAX_GRASP_ABOVE_SURFACE_METERS = 0.150
"""How high above the belt surface a grasp plane is still taken at face value.

An object the world laid on the belt rests within this of the surface, and the
same bound is what decides a body is on the belt at all. Past it the record is
about something else -- an object that fell, or one the arm is already carrying
-- and the plane is clamped rather than followed.
"""


def grasp_plane(surface: float, effector: Effector, centre_z: float) -> float:
    """Return the pinch plane a marker may stand at for an object's centre.

    The clamp's floor is the clearance the open jaw keeps plus how far its
    lowest collision geometry hangs below the pinch point, plus the millimetre
    the rise needs so it leads the hang. The shut jaw hangs further, and the
    hold climbs that difference while the fingers close, so the clearance at
    the end of the close is the clearance the arrival granted. Quoting the
    floor at the shut hang instead holds a flat parcel 13 mm above its centre
    for the whole close, and the pads meet it at the top.

    Args:
        surface: Belt surface height, in world frame meters.
        effector: The jaw whose clearance this is.
        centre_z: The object's centre height, in world frame meters.

    Returns:
        The pinch plane, in world frame meters.
    """
    return min(
        max(centre_z, surface + effector.open_pinch_floor),
        surface + MAX_GRASP_ABOVE_SURFACE_METERS,
    )


def marker_for(
    record: WasteObject,
    effector: Effector,
    belt_surface_height_world: float | None = None,
    *,
    belt_surface: float | None = None,
) -> GraspMarker:
    """Return where the effector would go for one record.

    Args:
        record: What the tracker settled.
        effector: The effector the pick geometry assumes.
        belt_surface_height_world: Height of the belt surface, in meters.
        belt_surface: Legacy keyword alias for belt_surface_height_world.

    Returns:
        The marker. Its horizontal position, closing axis and opening come
        from the record; its height comes from `effector`, because nothing on
        this line measures one.
    """
    surface = (
        belt_surface_height_world
        if belt_surface_height_world is not None
        else (0.0 if belt_surface is None else belt_surface)
    )
    x, y, _ = record.grasp_point
    pad_z = surface + effector.grasp_height
    opening = record.grasp_width
    oriented = record.footprint.oriented

    pads: tuple[NDArray[np.float64], ...] = ()
    axis: float | None = None
    if oriented:
        axis = record.grasp_axis
        # The pads face each other across the object, so their centers stand
        # half a pad beyond the opening on either side.
        reach = opening / 2.0 + effector.pad_thickness / 2.0
        step_x, step_y = math.cos(axis) * reach, math.sin(axis) * reach
        pads = (
            np.asarray((x - step_x, y - step_y, pad_z), dtype=np.float64),
            np.asarray((x + step_x, y + step_y, pad_z), dtype=np.float64),
        )

    return GraspMarker(
        track_id=record.track_id,
        valid_until_nanos=record.valid_until_nanos,
        pinch_position_belt=np.asarray((x, y, pad_z), dtype=np.float64),
        flange_position_world=np.asarray(
            (x, y, pad_z + effector.finger_length), dtype=np.float64
        ),
        pad_positions_belt=pads,
        pad_size=np.asarray(
            (
                effector.pad_thickness / 2.0,
                effector.pad_depth / 2.0,
                effector.pad_height / 2.0,
            ),
            dtype=np.float64,
        ),
        closing_yaw_belt=axis,
        opening=opening,
        oriented=oriented,
        reachable=opening <= effector.opening,
        channel=_channel_of(record.material),
        extent=record.footprint.major_extent,
        color=color_for(record.track_id),
    )


def markers_for(
    records: tuple[WasteObject, ...],
    effector: Effector,
    belt_surface_height_world: float | None = None,
    *,
    belt_surface: float | None = None,
) -> tuple[GraspMarker, ...]:
    """Return one marker per record, in the order given.

    Args:
        records: What the tracker settled at one instant.
        effector: The effector the pick geometry assumes.
        belt_surface_height_world: Height of the belt surface, in meters.
        belt_surface: Legacy keyword alias for belt_surface_height_world.

    Returns:
        The markers. A track that retired is absent from `records` and
        therefore absent here, which is how a marker stops being drawn.
    """
    surface = (
        belt_surface_height_world
        if belt_surface_height_world is not None
        else (0.0 if belt_surface is None else belt_surface)
    )
    return tuple(marker_for(record, effector, surface) for record in records)


def _box_corners(size: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the eight corners of a box centred on its geom frame.

    Args:
        size: Half-extents, the three numbers MuJoCo stores for a box.

    Returns:
        The corners, shape `(8, 3)`, in the geom frame.
    """
    signs = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    return signs * size


def _horizontal_footprint(
    local: NDArray[np.float64], rotation: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Return how wide an object is in the belt plane, and which way it faces.

    The jaw closes horizontally. The width it has to clear and the axis it has
    to line up with are the short and long directions of the object after the
    body's rotation, projected onto the belt. The body's Euler yaw is that axis
    only while the object lies flat on the axes it was scanned in; a parcel on
    its side has its long direction in what used to be its height.

    Args:
        local: Points in the geom frame, one row each.
        rotation: The geom's rotation, mapping that frame into the world.

    Returns:
        The full width along the short horizontal axis, the full width along
        the long one, both in meters, and the yaw of the short axis in radians.
        The yaw is an axis, so the opposite direction is the same answer.
    """
    flat = (local @ rotation.T)[:, :2]
    centered = flat - flat.mean(axis=0)
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    minor = vectors[:, 0]
    major = vectors[:, 1]
    along_short = centered @ minor
    along_long = centered @ major
    return (
        float(along_short.max() - along_short.min()),
        float(along_long.max() - along_long.min()),
        math.atan2(float(minor[1]), float(minor[0])),
    )


def ground_truth_markers(
    model: Any,
    data: Any,
    active_objects: list[Any],
    plan: Any,
    effector: Effector,
    belt_surface_height_world: float,
    at_nanos: int,
    window_exit: float,
    belt_speed: float,
) -> tuple[GraspMarker, ...]:
    """Compute grasp markers directly from MuJoCo ground-truth physics.

    One marker per active object on the belt, derived directly from the
    simulator body position, orientation quaternion, and geom extents rather
    than from camera segmentation and tracker estimation.

    Args:
        model: Compiled MuJoCo model.
        data: Current MuJoCo simulation state.
        active_objects: Objects currently managed by the conveyor.
        plan: Resolved scene layout carrying belt geometry.
        effector: Effector parameters sizing pads, fingers, and grasp height.
        belt_surface_height_world: Surface height of the belt in world frame.
        at_nanos: Current simulated timestamp in nanoseconds.
        window_exit: X-coordinate in belt frame where reachability closes.
        belt_speed: Conveyor belt speed in meters per second.

    Returns:
        One marker per active object resting on the belt within the reachable
        window, in pool slot order. A marker's identity is the object's spawn
        serial rather than its pool slot, because a slot is reused as soon as
        the object in it leaves the belt: a consumer that remembered the slot
        would be remembering a place, and would refuse to serve the second
        object to occupy it.
    """
    import mujoco

    surface = belt_surface_height_world

    markers: list[GraspMarker] = []
    for item in active_objects:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
        if body < 0:
            continue

        # The centre of mass, not the body origin. A scanned mesh keeps its
        # origin wherever the scan left it, and on this set that is tens of
        # millimetres from the mass. The free joint's linear velocity follows
        # the origin, so a marker built on the origin aims the jaw at a point
        # that whips around the parcel as it turns. `xipos` is the mass, and
        # `cvel`'s linear half is the velocity of that mass.
        com = data.xipos[body]
        x, y, z = float(com[0]), float(com[1]), float(com[2])

        # Exclude objects not resting on the driven belt region.
        # Height tolerance of 0.15 m accommodates packaging resting on the belt
        # while excluding dropped objects or items lifted by the gripper.
        on_belt = (
            abs(x) <= plan.belt.length / 2.0
            and abs(y) <= plan.belt.width / 2.0
            and surface - 0.05 < z <= surface + 0.15
        )
        if not on_belt or x > window_exit:
            continue

        # Body orientation quaternion: data.xquat[body] is [w, x, y, z]
        quat = data.xquat[body]
        w, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        geom_id = model.body_geomadr[body]
        geom_type = model.geom_type[geom_id] if geom_id >= 0 else None
        # The geom's own rotation, which for these bodies is the body's: used
        # for the mesh extents, for the height a mesh spans, and for the one
        # projection that turns a body-frame angular velocity into a rate about
        # the belt normal. One matrix, so the three cannot disagree.
        rotation = data.geom_xmat[geom_id].reshape(3, 3)

        oriented = True
        closing_axis: float | None = yaw

        pad_z = surface + effector.grasp_height
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r = float(model.geom_size[geom_id][0])
            opening = 2.0 * r
            extent = 2.0 * r
            oriented = False
            closing_axis = None
            center_z = float(data.geom_xpos[geom_id][2])
            pad_z = grasp_plane(surface, effector, center_z)
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            opening, extent, closing_axis = _horizontal_footprint(
                _box_corners(np.asarray(model.geom_size[geom_id], dtype=np.float64)),
                rotation,
            )
            center_z = float(data.geom_xpos[geom_id][2])
            pad_z = grasp_plane(surface, effector, center_z)
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            if mesh_id >= 0:
                start = model.mesh_vertadr[mesh_id]
                count = model.mesh_vertnum[mesh_id]
                verts = np.asarray(model.mesh_vert[start : start + count])
                opening, extent, closing_axis = _horizontal_footprint(verts, rotation)
                world_z = (rotation[2, :] @ verts.T) + float(data.geom_xpos[geom_id][2])
                center_z = float((world_z.min() + world_z.max()) / 2.0)
                pad_z = grasp_plane(surface, effector, center_z)
            else:
                opening = effector.opening * 0.5
                extent = opening
        else:
            opening = effector.opening * 0.5
            extent = opening

        flange_z = pad_z + effector.finger_length

        if closing_axis is not None:
            closing_axis = (closing_axis + math.pi) % (2.0 * math.pi) - math.pi

        pads: tuple[NDArray[np.float64], ...] = ()
        if oriented and closing_axis is not None:
            reach = opening / 2.0 + effector.pad_thickness / 2.0
            step_x = math.cos(closing_axis) * reach
            step_y = math.sin(closing_axis) * reach
            pads = (
                np.asarray((x - step_x, y - step_y, pad_z), dtype=np.float64),
                np.asarray((x + step_x, y + step_y, pad_z), dtype=np.float64),
            )

        obj_velocity: NDArray[np.float64] | None = None
        yaw_rate: float | None = None
        # `cvel` is the com-based spatial velocity, angular then linear, in the
        # world frame. The linear half belongs to the same point `xipos` names,
        # which is the point the marker stands on. The free joint's own `qvel`
        # is the origin's velocity instead, and the two disagree by the cross
        # product of the spin and the origin's offset from the mass.
        if data.cvel is not None and len(data.cvel) > body:
            spatial = data.cvel[body]
            obj_velocity = np.asarray(
                (float(spatial[3]), float(spatial[4]), float(spatial[5])),
                dtype=np.float64,
            )
            yaw_rate = float(spatial[2])

        remaining = max(0.0, window_exit - x)
        effective_speed = (
            float(obj_velocity[0])
            if obj_velocity is not None and obj_velocity[0] > 0.01
            else belt_speed
        )
        expires = (
            at_nanos + int(remaining / effective_speed * 1_000_000_000)
            if effective_speed > 0.0
            else at_nanos
        )

        markers.append(
            GraspMarker(
                track_id=item.serial,
                valid_until_nanos=expires,
                pinch_position_belt=np.asarray((x, y, pad_z), dtype=np.float64),
                flange_position_world=np.asarray((x, y, flange_z), dtype=np.float64),
                pad_positions_belt=pads,
                pad_size=np.asarray(
                    (
                        effector.pad_thickness / 2.0,
                        effector.pad_depth / 2.0,
                        effector.pad_height / 2.0,
                    ),
                    dtype=np.float64,
                ),
                closing_yaw_belt=closing_axis,
                opening=opening,
                oriented=oriented,
                reachable=opening <= effector.opening,
                channel=item.channel,
                extent=extent,
                color=color_for(item.serial),
                velocity_world=obj_velocity,
                yaw_rate_belt=yaw_rate,
            )
        )

    return tuple(markers)


def draw(scene: Any, markers: tuple[GraspMarker, ...]) -> int:
    """Add every marker to a scene, and report how many geoms that took.

    The scene is the renderer's own, already populated by `update_scene`. What
    this adds lives until the next `update_scene` overwrites it, so a marker
    is visible to whoever renders next and to nobody else: not to physics, not
    to a camera the tracker reads, and not to a run that never calls this.

    Args:
        scene: The `mjvScene` to add to.
        markers: What to draw.

    Returns:
        How many geoms were added, which is fewer than asked for when the
        scene runs out of room.
    """
    import mujoco
    import numpy as np

    added = 0
    for marker in markers:
        alpha = OPAQUE if marker.reachable else UNREACHABLE_ALPHA
        rgba = np.array([*marker.color, alpha], dtype=np.float32)
        for geom_type, size, position, yaw in _solids(marker):
            if scene.ngeom >= scene.maxgeom:
                return added
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                geom_type,
                np.asarray(size, dtype=np.float64),
                np.asarray(position, dtype=np.float64),
                _spin(yaw),
                rgba,
            )
            scene.ngeom += 1
            added += 1
    return added


def draw_park(
    scene: Any,
    position_world: NDArray[np.float64] | None = None,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    belt_surface_height_world: float | None = None,
    *,
    position: NDArray[np.float64] | None = None,
    belt_surface: float | None = None,
) -> int:
    """Add the park pose to a scene, and report how many geoms that took.

    A ball where the flange rests and a post down to belt height, so the pose
    reads as a place in the world rather than as a dot floating in it. Neither
    shape is a jaw: the arm grasps nothing at park, and drawing pads there
    would say it was about to.

    Args:
        scene: The `mjvScene` to add to.
        position_world: Where the flange rests in world coordinates.
        color: What to draw it in, as red, green and blue in the unit range.
        belt_surface_height_world: Height of the belt surface, in meters, which is where
            the post stops.
        position: Legacy keyword alias for position_world.
        belt_surface: Legacy keyword alias for belt_surface_height_world.

    Returns:
        How many geoms were added, which is fewer than asked for when the
        scene runs out of room.
    """
    import mujoco
    import numpy as np

    pos = position_world if position_world is not None else position
    if pos is None:
        raise TypeError("draw_park requires position_world or position")
    pos = np.asarray(pos, dtype=np.float64)
    surface = (
        belt_surface_height_world
        if belt_surface_height_world is not None
        else (0.0 if belt_surface is None else belt_surface)
    )

    rgba = np.array([*color, OPAQUE], dtype=np.float32)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    half_post = max((z - surface) / 2.0, PARK_BALL_RADIUS)
    solids: tuple[tuple[int, NDArray[np.float64], NDArray[np.float64]], ...] = (
        (
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            np.asarray((PARK_BALL_RADIUS, 0.0, 0.0), dtype=np.float64),
            pos,
        ),
        (
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            np.asarray((PARK_POST_RADIUS, half_post, 0.0), dtype=np.float64),
            np.asarray((x, y, z - half_post), dtype=np.float64),
        ),
    )
    added = 0
    for geom_type, size, place in solids:
        if scene.ngeom >= scene.maxgeom:
            return added
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            size,
            place,
            _spin(0.0),
            rgba,
        )
        scene.ngeom += 1
        added += 1
    return added


def _solids(
    marker: GraspMarker,
) -> tuple[tuple[int, NDArray[np.float64], NDArray[np.float64], float], ...]:
    """Return every solid one marker is made of.

    A jaw is two pads and the shaft that carries them. A pose with no axis
    gets the same shaft and a ring of pads instead of a pair, so the position
    the tracker is sure of is drawn without inventing the angle it is not.

    Args:
        marker: The pose to draw.

    Returns:
        Tuples of MuJoCo geom type, half-size, position, and rotation about
        the belt normal.
    """
    import mujoco

    x, y, grasp_z = (
        float(marker.grasp[0]),
        float(marker.grasp[1]),
        float(marker.grasp[2]),
    )
    half_shaft = (float(marker.flange[2]) - grasp_z) / 2.0
    solids: list[tuple[int, NDArray[np.float64], NDArray[np.float64], float]] = [
        (
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            np.asarray((SHAFT_RADIUS, half_shaft, 0.0), dtype=np.float64),
            np.asarray((x, y, grasp_z + half_shaft), dtype=np.float64),
            0.0,
        )
    ]
    for angle, pad in _pad_places(marker):
        solids.append((int(mujoco.mjtGeom.mjGEOM_BOX), marker.pad_size, pad, angle))
    return tuple(solids)


def _pad_places(marker: GraspMarker) -> tuple[tuple[float, NDArray[np.float64]], ...]:
    """Return where each pad stands and which way it faces.

    Args:
        marker: The pose to draw.

    Returns:
        Pairs of rotation about the belt normal and pad center.
    """
    if marker.oriented and marker.closing_axis is not None:
        return tuple((marker.closing_axis, pad) for pad in marker.pads)

    grasp_z = float(marker.grasp[2])
    reach = marker.opening / 2.0 + float(marker.pad_size[0])
    import numpy as np

    # A ring of pads round the grasp, one expression rather than one per pad:
    # the angles are `linspace` over a full turn and the centers are the grasp
    # plus the radius on each axis.
    angles = np.linspace(0.0, 2.0 * np.pi, UNORIENTED_PADS, endpoint=False)
    centers = np.stack(
        (
            float(marker.grasp[0]) + np.cos(angles) * reach,
            float(marker.grasp[1]) + np.sin(angles) * reach,
            np.full(UNORIENTED_PADS, grasp_z),
        ),
        axis=1,
    )
    return tuple(
        (float(angle), np.asarray((x, y, z), dtype=np.float64))
        for angle, (x, y, z) in zip(angles, centers, strict=True)
    )


def _spin(yaw: float) -> Any:
    """Return the rotation matrix for a turn about the belt normal.

    Args:
        yaw: The rotation, in radians.

    Returns:
        The matrix, flattened row major, as MuJoCo expects it.
    """
    import numpy as np

    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.array([cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _channel_of(material: str) -> str:
    """Return the channel a material class routes to.

    Args:
        material: A taxonomy identifier, or `reject` when the tracker could
            not settle on one.

    Returns:
        The channel. Anything outside the taxonomy goes to the reject
        channel rather than raising: a track the tracker cannot classify is
        an ordinary outcome and the line has somewhere to put it.
    """
    try:
        return channel_of(material)
    except KeyError:
        return REJECT_CHANNEL
