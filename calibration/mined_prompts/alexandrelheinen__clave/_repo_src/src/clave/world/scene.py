"""Assemble the simulated sorting line.

The scene is built programmatically rather than written as a static MJCF file.
Bin count follows the taxonomy, object placement follows a seed, and belt
geometry follows configuration, none of which a fixed XML can express.

The manipulator is attached from a pinned submodule, which is FRET's convention.
CLAVE does not vendor meshes.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.world import arm
from clave.world.config import WorldConfigError, require, require_range
from clave.world.objects import ObjectSpec, channels, parse

LOGGER = logging.getLogger(__name__)

FLANGE_SITE = "attachment_site"
"""The face the gripper bolts to, as the arm model names it before prefixing.

`clave.world.arm` knows it as `arm_attachment_site`, which is the same site
after the scene's own prefix. Every trusted reach bound was swept against it
and the gripper hangs off it rather than displacing it, so adopting an
effector does not move the workspace.
"""

GRIPPER_MODEL = Path("third_party") / "mujoco_menagerie_robotiq_2f85" / "2f85.xml"
GRIPPER_ACTUATOR_NAME = "fingers_actuator"
"""The gripper's own actuator, before the arm's prefix is applied to it.

Named here rather than read from `clave.world.arm`, which knows the actuator
under the prefix this scene attaches it with: the limit has to be set on the
vendored spec, which is the one place the unprefixed name exists.
"""
"""A Robotiq 2F-85 parallel jaw, from the same Menagerie commit as the arm.

Adopted against what a municipal packaging line would install, which is
suction, and the reason is the simulator rather than the process:
`docs/research/sorting-outputs-and-effectors.md` records that Menagerie
carries no suction model and MuJoCo no vacuum primitive, so a suction pick
would be a weld appearing when the tool is near enough. A jaw closing is
contact physics the solver works out.
"""

ARM_MODEL = Path("third_party") / "mujoco_menagerie_ur10e" / "ur10e.xml"
"""A Universal Robots UR10e, adopted from MuJoCo Menagerie rather than authored
here. A validated model derived from manufacturer CAD outranked the SCARA
this replaced, and the one model is copied into third_party/ instead of
pinning a 2.3 GB collection to obtain 35 MB of it.
"""

PARKED_Z = 0.05
"""Height at which pooled objects wait before they are spawned.

They rest on the floor well to the side of the belt rather than hanging beneath
it. A MuJoCo plane collides from above only, so a body parked below the floor
falls forever and eventually drives the solver to a NaN. v0.6.0 worked around
that by having the conveyor pin every parked slot on each step, which left the
scene stable only while a conveyor happened to be stepping it: building the
world and stepping it directly, as the arm controller does, reproduced the
instability. Parking above the floor makes the scene stable on its own.
"""

PARKED_X = 3.0
"""How far to the side pooled objects wait, clear of the belt and the camera."""

OFFSCREEN_WIDTH = 1920
"""Floor on the widest frame any renderer attached to this model can produce.

A sensor declaring more pixels than this raises it, because a camera that
cannot be rendered at its own resolution is a camera whose stated resolution
means nothing. `configs/world/sorting_line.yml` declares 2448 by 2048, and the
1920 default silently capped it.
"""

OFFSCREEN_HEIGHT = 1080
"""Tallest frame any renderer attached to this model can produce."""

DEFAULT_DEMO_PRESENTATION: dict[str, Any] = {
    "quality": {
        "shadowsize": 2048,
        "offsamples": 4,
    },
    "headlight": {
        "ambient": [0.26, 0.27, 0.29],
        "diffuse": [0.08, 0.08, 0.10],
        "specular": [0.28, 0.28, 0.28],
    },
    "background": {
        "top": [0.22, 0.23, 0.26],
        "bottom": [0.40, 0.38, 0.36],
    },
    "lights": [
        {
            "position_meters": [1.40, -2.10, 1.15],
            "direction": [-0.55, 0.72, -0.30],
            "directional": True,
            "diffuse": [0.82, 0.80, 0.74],
            "specular": [0.12, 0.12, 0.12],
            "cast_shadow": True,
        },
        {
            "position_meters": [-1.70, 1.90, 1.05],
            "direction": [0.68, -0.66, -0.28],
            "directional": True,
            "diffuse": [0.48, 0.50, 0.54],
            "specular": [0.06, 0.06, 0.06],
            "cast_shadow": False,
        },
        {
            "position_meters": [2.20, -0.40, 0.95],
            "direction": [-0.80, 0.20, -0.40],
            "directional": True,
            "diffuse": [0.28, 0.24, 0.18],
            "specular": [0.03, 0.03, 0.03],
            "cast_shadow": False,
        },
    ],
}


@dataclass(frozen=True)
class BeltGeometry:
    """The belt's extent and how fast it runs.

    Attributes:
        length: Belt length along the travel axis, in meters.
        width: Belt width, in meters.
        surface_height: Height of the belt surface above the floor, in meters.
        speed: Belt speed in meters per second, resolved from its range.
        height_tolerance: Vertical allowance for deciding an object still
            rests on the belt, in meters.
    """

    length: float
    width: float
    surface_height: float
    speed: float
    height_tolerance: float


@dataclass(frozen=True)
class TakeawayDrive:
    """How sorted objects are driven once they leave a chute.

    Attributes:
        speed: Drive speed along the take-away, in meters per second.
        height_min: Lowest height that still counts as on a take-away.
        height_max: Highest height that still counts as on a take-away.
        mouth_half_width: Lateral acceptance around a chute mouth, in meters.
        past_mouth: How far past the mouth along the take-away still counts.
        toward_mouth: How far back toward the mouth still counts, in meters.
    """

    speed: float
    height_min: float
    height_max: float
    mouth_half_width: float
    past_mouth: float
    toward_mouth: float


@dataclass(frozen=True)
class SceneLayout:
    """Everything the belt drive and the reachability report need.

    Attributes:
        belt: Belt geometry and speed.
        arm_base: Where the arm's base column stands, in meters. This is the
            top of its pedestal, and the reach test is radial about it.
        reach_min: Inner radius of the annulus the arm is trusted over.
        reach_max: Outer radius of that annulus, in meters.
        tool_above_base: Lowest and highest the flange is trusted at, relative
            to the arm's own base rather than to the belt, because that is what
            the sweep measured.
        pedestal: Footprint of the support the arm stands on, in meters.
        channels: Channel identifiers, one chute each.
        chutes: Where each channel's chute mouth stands, in world meters.
            Computed here rather than where the geometry is built, so the
            arm releasing over an opening and the opening itself come from
            one arithmetic.
        takeaway: How the take-away conveyor drives sorted objects.
        objects: The object set the pool draws from.
        pool_size: How many object bodies exist.
        timestep: Simulation timestep in seconds.
        dressed: Whether the warehouse scene dressing was found and used. It
            is reported rather than assumed, because the assets are generated
            from a submodule and a clone that has not run the importer gets a
            plain floor. Nothing the simulation measures depends on it.
    """

    belt: BeltGeometry
    arm_base: NDArray[np.float64]
    reach_min: float
    reach_max: float
    tool_above_base: tuple[float, float]
    pedestal: tuple[float, float]
    channels: tuple[str, ...]
    objects: tuple[ObjectSpec, ...]
    pool_size: int
    timestep: float
    takeaway: TakeawayDrive
    chutes: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    dressed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "arm_base", np.asarray(self.arm_base, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "chutes",
            {
                channel: np.asarray(mouth, dtype=np.float64)
                for channel, mouth in self.chutes.items()
            },
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SceneLayout):
            return NotImplemented
        if self.chutes.keys() != other.chutes.keys():
            return False
        chutes_match = all(
            bool(np.allclose(self.chutes[key], other.chutes[key], rtol=0.0, atol=1e-12))
            for key in self.chutes
        )
        return (
            self.belt == other.belt
            and bool(np.allclose(self.arm_base, other.arm_base, rtol=0.0, atol=1e-12))
            and self.reach_min == other.reach_min
            and self.reach_max == other.reach_max
            and self.tool_above_base == other.tool_above_base
            and self.pedestal == other.pedestal
            and self.channels == other.channels
            and self.objects == other.objects
            and self.pool_size == other.pool_size
            and self.timestep == other.timestep
            and self.takeaway == other.takeaway
            and chutes_match
            and self.dressed == other.dressed
        )


def _arm_spec(root: Path) -> Any:
    """Load the manipulator.

    Args:
        root: Repository root.

    Returns:
        The manipulator spec.

    Raises:
        FileNotFoundError: If the model is absent, which means a broken
            checkout rather than a missing submodule: this file is committed.
    """

    import mujoco

    path = root / ARM_MODEL
    if not path.is_file():
        raise FileNotFoundError(
            f"{ARM_MODEL} is missing. It is committed to this repository "
            f"rather than generated, so a checkout that lacks it is broken."
        )
    return mujoco.MjSpec.from_file(str(path))


def _gripper_spec(root: Path, closing_torque: float) -> Any:
    """Load the gripper model, with the force its fingers may apply.

    The vendored model ships ±5 N·m on its actuator, which is around a hundred
    newtons at the pads and four orders of magnitude more than the ten gram
    parcels this world carries. Left as it ships, closing on one pushes it
    through the belt surface and takes the jaw down with it: measured, 27 mm of
    the object below the belt and the jaw's own collision geometry 1.4 mm
    inside it. The vendored file is a pinned submodule and is not edited, so
    the limit is applied here, to the spec, by the value the world
    configuration asks for.

    Args:
        root: Repository root.
        closing_torque: The torque the fingers may apply, in newton meters.

    Returns:
        The parsed spec.

    Raises:
        FileNotFoundError: If the vendored copy is absent.
    """
    import mujoco

    path = root / GRIPPER_MODEL
    if not path.is_file():
        raise FileNotFoundError(
            f"{GRIPPER_MODEL} is missing. It is committed to this repository "
            f"rather than generated, so a checkout that lacks it is broken."
        )
    spec = mujoco.MjSpec.from_file(str(path))
    for actuator in spec.actuators:
        if actuator.name == GRIPPER_ACTUATOR_NAME:
            actuator.forcerange[:] = (-closing_torque, closing_torque)
    return spec


def _armed(root: Path, closing_torque: float) -> Any:
    """Return the arm with the gripper bolted to its flange.

    The gripper goes onto the arm before the arm goes into the world, so the
    attachment point is the flange the arm itself declares rather than a
    frame this scene computes. That is also why the reach bounds do not
    move: they were swept against `attachment_site`, and the gripper hangs
    off it rather than displacing it.

    Args:
        root: Repository root.
        closing_torque: The torque the gripper's fingers may apply, in newton
            meters, which the reach sweep does not depend on.

    Returns:
        The combined spec.
    """
    arm = _arm_spec(root)
    with warnings.catch_warnings():
        # The gripper declares its own impratio and cone; the scene has
        # already settled both, so the conflict notice carries no
        # information.
        warnings.simplefilter("ignore")
        arm.attach(
            _gripper_spec(root, closing_torque),
            prefix="grip_",
            site=arm.site(FLANGE_SITE),
        )
    return arm


def layout(raw: dict[str, Any], rng: np.random.Generator) -> SceneLayout:
    """Resolve configuration and randomization ranges into one layout.

    Args:
        raw: The parsed world configuration.
        rng: Generator used to resolve ranges.

    Returns:
        The resolved layout.

    Raises:
        WorldConfigError: If a required key is missing, a range is invalid, or
            an object can be drawn wider than the gripper opens.
    """
    physics = require(raw, "physics")
    belt_cfg = require(raw, "belt")
    arm_cfg = require(raw, "arm")
    spawn_cfg = require(raw, "spawn")
    specs = parse(
        require(raw, "objects"),
        float(require(arm_cfg, "max_grasp_width_meters", "arm")),
    )
    LOGGER.debug(
        "world layout: %d object templates, jaw limit %.3f m, seed state initialized",
        len(specs),
        float(require(arm_cfg, "max_grasp_width_meters", "arm")),
    )

    belt = BeltGeometry(
        length=float(require(belt_cfg, "length_meters", "belt")),
        width=float(require(belt_cfg, "width_meters", "belt")),
        surface_height=float(require(belt_cfg, "surface_height_meters", "belt")),
        speed=require_range(belt_cfg, "speed_meters_per_second", "belt").sample(rng),
        height_tolerance=float(require(belt_cfg, "height_tolerance_meters", "belt")),
    )
    takeaway_cfg = require(require(raw, "chutes"), "takeaway", "chutes")
    takeaway = TakeawayDrive(
        speed=float(
            require(takeaway_cfg, "speed_meters_per_second", "chutes.takeaway")
        ),
        height_min=float(require(takeaway_cfg, "height_min_meters", "chutes.takeaway")),
        height_max=float(require(takeaway_cfg, "height_max_meters", "chutes.takeaway")),
        mouth_half_width=float(
            require(takeaway_cfg, "mouth_half_width_meters", "chutes.takeaway")
        ),
        past_mouth=float(require(takeaway_cfg, "past_mouth_meters", "chutes.takeaway")),
        toward_mouth=float(
            require(takeaway_cfg, "toward_mouth_meters", "chutes.takeaway")
        ),
    )
    base = require(arm_cfg, "base_position_meters", "arm")
    band = require(arm_cfg, "tool_above_base_meters", "arm")
    pedestal = require(arm_cfg, "pedestal_footprint_meters", "arm")
    return SceneLayout(
        belt=belt,
        arm_base=np.asarray(
            (float(base[0]), float(base[1]), float(base[2])), dtype=np.float64
        ),
        reach_min=float(require(arm_cfg, "reach_min_meters", "arm")),
        reach_max=float(require(arm_cfg, "reach_max_meters", "arm")),
        tool_above_base=(float(band[0]), float(band[1])),
        pedestal=(float(pedestal[0]), float(pedestal[1])),
        channels=channels(specs),
        chutes=_chute_mouths(
            require(raw, "chutes"), channels(specs), belt.surface_height
        ),
        takeaway=takeaway,
        objects=specs,
        pool_size=int(require(spawn_cfg, "pool_size", "spawn")),
        timestep=float(require(physics, "timestep_seconds", "physics")),
    )


def _add_chutes(
    mujoco: Any, world: Any, plan: SceneLayout, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Build one funnel per channel, each over its own take-away conveyor.

    The funnel is four slanted walls from a mouth at belt height down to a
    throat, rather than a hole in a plate. An object released near the edge
    of a hole is stranded on its lip; the same object released over a funnel
    slides in, which is the difference between a chute and a target the arm
    has to hit exactly.

    What sits below the throat is plant. It has no collision and nothing
    measures it, and it is here so a channel visibly goes somewhere.

    Args:
        mujoco: The imported module.
        world: The worldbody the geometry attaches to.
        plan: The resolved scene layout, for the channels and the arm.
        cfg: The `chutes` configuration block.

    Returns:
        Channel identifier to its mouth geom, so a presentation scenario can
        recolor the openings per channel.

    Raises:
        WorldConfigError: If a key is absent, or if any mouth falls outside
            the region the arm is trusted over, or if any mouth overlaps the
            pedestal in the belt travel axis. A chute the arm cannot reach is
            a channel that can never be served, and catching that at load
            beats counting it as a fault at run time.
    """
    mouth = [float(v) for v in require(cfg, "mouth_meters", "chutes")]
    throat = [float(v) for v in require(cfg, "throat_meters", "chutes")]
    fall = float(require(cfg, "throat_height_meters", "chutes"))
    takeaway = require(cfg, "takeaway", "chutes")
    takeaway_surface_height_world = float(
        require(takeaway, "surface_height_meters", "chutes.takeaway")
    )
    away_wide = float(require(takeaway, "width_meters", "chutes.takeaway"))
    away_long = float(require(takeaway, "length_meters", "chutes.takeaway"))

    belt_surface_height_world = plan.belt.surface_height
    mouths: dict[str, Any] = {}
    for channel in plan.channels:
        x, offset, _ = plan.chutes[channel]
        _refuse_unreachable(plan, channel, x, offset, belt_surface_height_world)
        _refuse_over_pedestal(plan, channel, x, mouth[0])
        mouths[channel] = _add_funnel(
            mujoco,
            world,
            channel,
            center_world=(x, offset),
            mouth=mouth,
            throat=throat,
            belt_surface_height_world=belt_surface_height_world,
            fall=fall,
        )
        # The take-away runs out from under the throat, away from the line, so
        # it clears the belt and the pedestal both.
        takeaway_y = offset + throat[1] / 2.0 - away_long / 2.0
        world.add_geom(
            name=f"takeaway_{channel}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[away_wide / 2.0, away_long / 2.0, 0.02],
            pos=[x, takeaway_y, takeaway_surface_height_world - 0.02],
            rgba=[0.22, 0.22, 0.26, 1.0],
            contype=1,
            conaffinity=1,
        )
        guide_h = 0.04
        for sign in (-1.0, 1.0):
            world.add_geom(
                name=f"takeaway_{channel}_guide_{'lo' if sign < 0 else 'hi'}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.005, away_long / 2.0, guide_h / 2.0],
                pos=[
                    x + sign * (away_wide / 2.0 + 0.005),
                    takeaway_y,
                    takeaway_surface_height_world + guide_h / 2.0,
                ],
                rgba=[0.30, 0.30, 0.34, 1.0],
            )
        world.add_geom(
            name=f"takeaway_{channel}_end",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[away_wide / 2.0 + 0.01, 0.005, guide_h / 2.0],
            pos=[
                x,
                offset + throat[1] / 2.0 - away_long - 0.005,
                takeaway_surface_height_world + guide_h / 2.0,
            ],
            rgba=[0.30, 0.30, 0.34, 1.0],
        )
    return mouths


def _add_funnel(
    mujoco: Any,
    world: Any,
    channel: str,
    center_world: tuple[float, float] | None = None,
    mouth: list[float] | None = None,
    throat: list[float] | None = None,
    belt_surface_height_world: float | None = None,
    fall: float = 0.0,
    *,
    centre: tuple[float, float] | None = None,
    top: float | None = None,
) -> Any:
    """Build one funnel and return the geom that marks its mouth.

    Args:
        mujoco: The imported module.
        world: The worldbody the geometry attaches to.
        channel: What the funnel is for, used to name its geoms.
        center_world: Where the mouth stands in world frame, along belt travel
            and across it.
        mouth: Mouth size, along travel and across it.
        throat: Throat size, the same way round.
        belt_surface_height_world: Height of the mouth, which is the belt
            surface.
        fall: How far the throat sits below the mouth.
        centre: Legacy keyword alias for center_world.
        top: Legacy keyword alias for belt_surface_height_world.

    Returns:
        A thin frame geom at the mouth, which is what a presentation
        scenario recolors and what a reader sees from above.
    """
    c = center_world if center_world is not None else centre
    if c is None:
        raise TypeError("_add_funnel requires center_world or centre")
    t = (
        belt_surface_height_world
        if belt_surface_height_world is not None
        else (0.0 if top is None else top)
    )
    m = mouth if mouth is not None else [0.0, 0.0]
    th = throat if throat is not None else [0.0, 0.0]

    x, y = c
    half_fall = fall / 2.0
    for axis, (wide_at_mouth, wide_at_throat) in enumerate(
        ((m[0], th[0]), (m[1], th[1]))
    ):
        for sign in (-1.0, 1.0):
            # Each wall spans mouth to throat, so its tilt is the difference
            # in half widths over the fall. A wall is a slab rotated about the
            # other horizontal axis.
            run = (wide_at_mouth - wide_at_throat) / 2.0
            lean = math.atan2(run, fall)
            length = math.hypot(run, fall) / 2.0
            mid = sign * (wide_at_mouth + wide_at_throat) / 4.0
            across = (m[1] if axis == 0 else m[0]) / 2.0
            pos = [x, y, t - half_fall]
            pos[axis] += mid
            # Rotate about the axis the wall does not span, leaning inward.
            angle = sign * lean if axis == 0 else -sign * lean
            quat = (
                [math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0]
                if axis == 0
                else [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]
            )
            world.add_geom(
                name=f"chute_{channel}_{'xy'[axis]}{'lo' if sign < 0 else 'hi'}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(
                    [0.006, across, length] if axis == 0 else [across, 0.006, length]
                ),
                pos=pos,
                quat=quat,
                rgba=[0.30, 0.30, 0.34, 1.0],
            )
    return world.add_geom(
        name=f"chute_{channel}_mouth",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[m[0] / 2.0, m[1] / 2.0, 0.004],
        pos=[x, y, t + 0.004],
        rgba=[0.15, 0.45, 0.65, 0.0],
        contype=0,
        conaffinity=0,
    )


def _refuse_unreachable(
    plan: SceneLayout, channel: str, x: float, y: float, top: float
) -> None:
    """Refuse a chute the arm cannot reach, naming it.

    Args:
        plan: The resolved scene layout.
        channel: What the chute is for.
        x: Where its mouth stands along belt travel.
        y: Where it stands across the belt.
        top: The mouth's height.

    Raises:
        WorldConfigError: If the mouth lies outside the annulus or outside
            the vertical band the arm is trusted over.
    """
    radius = math.hypot(x - plan.arm_base[0], y - plan.arm_base[1])
    if not plan.reach_min <= radius <= plan.reach_max:
        raise WorldConfigError(
            f"chute {channel!r} stands {radius:.3f} m from the arm base, outside "
            f"the {plan.reach_min:.2f} m to {plan.reach_max:.2f} m annulus, so "
            f"that channel could never be served"
        )
    above = top - plan.arm_base[2]
    low, high = plan.tool_above_base
    if not low <= above <= high:
        raise WorldConfigError(
            f"chute {channel!r} opens {above:+.3f} m from the arm base, outside "
            f"the trusted {low:+.2f} m to {high:+.2f} m band"
        )


def _refuse_over_pedestal(
    plan: SceneLayout, channel: str, x: float, mouth_length: float
) -> None:
    """Refuse a chute whose take-away would run through the arm's pedestal.

    Args:
        plan: The resolved scene layout.
        channel: What the chute is for.
        x: Where its mouth stands along belt travel.
        mouth_length: How far the mouth spans along belt travel.

    Raises:
        WorldConfigError: If the mouth overlaps the pedestal footprint in the
            belt travel axis.
    """
    reach = plan.pedestal[0] / 2.0 + mouth_length / 2.0
    if abs(x - plan.arm_base[0]) < reach:
        raise WorldConfigError(
            f"chute {channel!r} stands at x={x:+.3f} m, within {reach:.3f} m of "
            f"the pedestal at x={plan.arm_base[0]:+.3f} m, so its take-away "
            f"conveyor would run through the structure carrying the arm"
        )


def _chute_mouths(
    cfg: dict[str, Any], names: tuple[str, ...], surface: float
) -> dict[str, NDArray[np.float64]]:
    """Return where each channel's chute mouth stands.

    Args:
        cfg: The `chutes` configuration block.
        names: The channels the routing policy resolves to.
        surface: Height of the belt surface, which the mouths sit at.

    Returns:
        Channel identifier to the centre of its mouth.

    Raises:
        WorldConfigError: If a key is absent.
    """
    offset = float(require(cfg, "offset_from_belt_meters", "chutes"))
    first = float(require(cfg, "first_position_meters", "chutes"))
    spacing = float(require(cfg, "spacing_meters", "chutes"))
    return {
        channel: np.asarray(
            (first + index * spacing, offset, surface), dtype=np.float64
        )
        for index, channel in enumerate(names)
    }


WAREHOUSE_ASSETS = Path("assets") / "warehouse"
"""Where `scripts/import_scene_assets.py` writes the warehouse props."""

CONVEYOR_MODULE = Path("assets") / "conveyor" / "module.obj"
"""Where the same script writes the conveyor module."""

MODULE_SOURCE_METERS = (0.500, 0.504, 0.502)
"""The module as published, before the world scales it to its own belt.

Recorded here so the scale factors are derived from a measurement rather than
from a number somebody typed. `assets/conveyor/IMPORTED.md` carries the same
figures beside the digest they came from.
"""


def _light_for_presentation(
    mujoco: Any, spec: Any, world: Any, presentation: dict[str, Any]
) -> None:
    """Light the scene for a photograph rather than for a dataset.

    Every value comes from the still's own configuration. Nothing here is read
    when a caller does not ask for it, so a dataset recorded from this world is
    lit by the single overhead light above and by nothing else.

    The modelling follows the FRET project-page stills: headlight diffuse is
    kept low so the floor does not blow out, and the form comes from grazing
    directional fills rather than a top-down spot.

    Args:
        mujoco: The imported module.
        spec: The model spec, which owns the skybox and the headlight.
        world: The worldbody the extra lights attach to.
        presentation: The `lighting` section of a still scenario.
    """
    headlight = presentation.get("headlight")
    if headlight:
        for channel in ("ambient", "diffuse", "specular"):
            value = headlight.get(channel)
            if value is not None:
                setattr(spec.visual.headlight, channel, [float(v) for v in value])

    quality = presentation.get("quality")
    if quality:
        if "shadowsize" in quality:
            spec.visual.quality.shadowsize = int(quality["shadowsize"])
        if "offsamples" in quality:
            spec.visual.quality.offsamples = int(quality["offsamples"])

    background = presentation.get("background")
    if background:
        spec.add_texture(
            name="presentation_sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            width=512,
            height=512,
            rgb1=[float(v) for v in require(background, "top", "background")],
            rgb2=[float(v) for v in require(background, "bottom", "background")],
        )

    for index, light in enumerate(presentation.get("lights", [])):
        kwargs: dict[str, Any] = {
            "name": f"presentation_light_{index}",
            "pos": [float(v) for v in require(light, "position_meters", "lights")],
            "dir": [float(v) for v in require(light, "direction", "lights")],
            "diffuse": [float(v) for v in require(light, "diffuse", "lights")],
            "specular": [float(v) for v in light.get("specular", [0.1, 0.1, 0.1])],
            "castshadow": bool(light.get("cast_shadow", True)),
        }
        if light.get("directional"):
            kwargs["type"] = int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        ambient = light.get("ambient")
        if ambient is not None:
            kwargs["ambient"] = [float(v) for v in ambient]
        world.add_light(**kwargs)


def _finish_presentation_lights(mujoco: Any, model: Any) -> None:
    """Leave only the presentation lights active on a compiled model.

    The shared world carries a top-down key and the vendored arm carries a
    tracking spotlight. Both suit a dataset render and flatten a project-page
    still, so a presentation build turns them off after compile
    (`AC-STILL-02`).

    Args:
        mujoco: The imported module.
        model: The compiled model, modified in place.
    """
    for index in range(model.nlight):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, index)
        if name is not None and name.startswith("presentation_"):
            continue
        model.light_active[index] = 0


def _annotation_color(annotations: dict[str, Any], channel: str) -> list[float] | None:
    """The color a channel is drawn in, or None when the still names no colors.

    Args:
        annotations: The `annotations` section of a still scenario.
        channel: The channel identifier to look up.

    Returns:
        A four-component color, or None.

    Raises:
        KeyError: Through `require`, if colors are declared and this channel is
            not among them. A figure that silently draws one class in the
            default color teaches a reader the wrong legend.
    """
    colors = annotations.get("channel_colors")
    if not colors:
        return None
    rgb = [float(v) for v in require(colors, channel, "annotations.channel_colors")]
    return [rgb[0], rgb[1], rgb[2], 1.0]


def _annotate_for_presentation(
    mujoco: Any,
    world: Any,
    plan: SceneLayout,
    annotations: dict[str, Any],
    mouths: dict[str, Any],
    markers: list[tuple[Any, str]],
) -> None:
    """Stand the explanatory geometry in the scene, for a figure.

    This is what keeps a published figure a render rather than a diagram drawn
    over one. The ring, the window edges and the class markers are geometry the
    renderer sees, and every quantity they express is read from the resolved
    layout or from the same reachability sweep the safety layer consults, never
    from the scenario file.

    Nothing here introduces a body, and every shape carries no mass and no
    contact, so a world built with annotations has the bodies, coordinates and
    degrees of freedom of one built without them, and an annotated object falls
    exactly as an unannotated one does.

    Args:
        mujoco: The imported module.
        world: The worldbody the ring and the edges attach to.
        plan: The resolved scene layout.
        annotations: The `annotations` section of a still scenario.
        mouths: Channel identifier to its chute mouth, for recoloring.
        markers: Each object body paired with the channel it routes to.
    """
    # Imported here rather than at module scope: the reachability sweep lives
    # in the conveyor module, which reads this module's layout type, so naming
    # it at the top would close a cycle.
    from clave.world.belt import reach_report

    inert = {"density": 0.0, "contype": 0, "conaffinity": 0}

    if annotations.get("color_chutes"):
        for channel, geom in mouths.items():
            color = _annotation_color(annotations, channel)
            if color is not None:
                geom.rgba = color

    ring = annotations.get("reach_ring")
    if ring:
        rgba = [float(v) for v in require(ring, "rgba", "annotations.reach_ring")]
        thickness = float(require(ring, "thickness_meters", "annotations.reach_ring"))
        count = int(require(ring, "segment_count", "annotations.reach_ring"))
        # The radii are the layout's, so a ring cannot outlive the
        # workspace it describes.
        for label, radius in (("inner", plan.reach_min), ("outer", plan.reach_max)):
            for index in range(count):
                angle = 2.0 * math.pi * index / count
                world.add_geom(
                    name=f"annotation_ring_{label}_{index}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[radius * math.pi / count, thickness / 2.0, thickness / 2.0],
                    pos=[
                        plan.arm_base[0] + radius * math.cos(angle),
                        plan.arm_base[1] + radius * math.sin(angle),
                        plan.belt.surface_height + thickness,
                    ],
                    quat=[math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)],
                    rgba=rgba,
                    **inert,
                )

    edges = annotations.get("window_edges")
    if edges:
        rgba = [float(v) for v in require(edges, "rgba", "annotations.window_edges")]
        thickness = float(
            require(edges, "thickness_meters", "annotations.window_edges")
        )
        height = float(require(edges, "height_meters", "annotations.window_edges"))
        # The sweep places these, so the figure and the safety layer
        # cannot disagree about where an object enters and leaves reach.
        for label, position in zip(
            ("open", "close"), reach_report(plan).window_edges, strict=True
        ):
            world.add_geom(
                name=f"annotation_window_{label}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[thickness / 2.0, plan.belt.width / 2.0, height / 2.0],
                pos=[position, 0.0, plan.belt.surface_height + height / 2.0],
                rgba=rgba,
                **inert,
            )

    tags = annotations.get("class_markers")
    if tags:
        radius = float(require(tags, "radius_meters", "annotations.class_markers"))
        above = float(require(tags, "height_above_meters", "annotations.class_markers"))
        # The color is the object's own channel rather than a value
        # the scenario assigns per slot, so the legend cannot drift from the
        # object set.
        for body, channel in markers:
            color = _annotation_color(annotations, channel)
            if color is None:
                continue
            body.add_geom(
                name=f"{body.name}_marker",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[radius, 0.0, 0.0],
                pos=[0.0, 0.0, above],
                rgba=color,
                **inert,
            )


def _dress(mujoco: Any, spec: Any, world: Any, raw: dict[str, Any], root: Path) -> bool:
    """Lay the floor and stand the scene dressing on it.

    The assets are converted from a pinned submodule and are not committed, so
    a clone that has not run the importer gets the plain floor instead. Nothing
    here touches the belt, the arm or the objects: every prop is static geometry
    standing clear of both, and the overhead camera the models see looks
    straight down at the belt rather than at any of it.

    Args:
        mujoco: The imported module.
        spec: The model spec, which owns textures and meshes.
        world: The worldbody being assembled.
        raw: The parsed world configuration.
        root: Repository root.

    Returns:
        Whether the warehouse assets were found and used.
    """
    assets = root / WAREHOUSE_ASSETS
    ground = assets / "textures" / "ground.png"
    if not ground.is_file():
        world.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[5.0, 5.0, 0.1],
            pos=[0.0, 0.0, 0.0],
        )
        return False

    spec.add_texture(
        name="warehouse_ground",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(ground),
    )
    spec.add_material(
        name="warehouse_ground",
        textures=["", "warehouse_ground"],
        texrepeat=[6.0, 6.0],
    )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        pos=[0.0, 0.0, 0.0],
        material="warehouse_ground",
    )

    for index, prop in enumerate(require(raw, "environment").get("props", [])):
        name = str(require(prop, "mesh", "environment.props"))
        path = assets / "meshes" / name
        if not path.is_file():
            continue
        mesh_name = f"prop_{index}"
        spec.add_mesh(name=mesh_name, file=str(path))
        position = [float(v) for v in require(prop, "position_meters", name)]
        yaw = math.radians(float(require(prop, "yaw_degrees", name)))
        world.add_geom(
            name=mesh_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            pos=position,
            quat=[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
            # Visual only. A prop that collided would be one more thing the
            # solver has to resolve every step for no gain, since none of them
            # is inside the workspace.
            contype=0,
            conaffinity=0,
        )
    return True


def _check_mesh_objects_fit(
    mujoco: Any, model: Any, plan: SceneLayout, limit: float
) -> None:
    """Refuse a scanned object the gripper cannot close on.

    A primitive declares its size, so `clave.world.objects` checks it at load.
    A mesh carries its size in the file, so the check happens here, against the
    vertices MuJoCo compiled. Both ask the same question.

    Both of the two narrowest extents must fit the opening. A single thin side
    is not enough: a flat object with one thin edge and two broad sides still
    cannot fit between the fingers.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        plan: The resolved layout, naming which objects are meshes.
        limit: The widest object the gripper can close on, in meters.

    Raises:
        WorldConfigError: Naming the object, its width and the limit.
    """
    from clave.world.config import WorldConfigError

    for spec in plan.objects:
        if not spec.is_mesh:
            continue
        mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, f"asset_{spec.name}")
        if mesh < 0:
            continue
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count]
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        narrowest = sorted(float(extent) for extent in extents)[:2]
        if not _two_narrowest_sides_fit(extents, limit):
            raise WorldConfigError(
                f"object {spec.name!r} has two narrowest sides of "
                f"{narrowest[0] * 1000:.1f} mm and {narrowest[1] * 1000:.1f} mm, "
                f"but the gripper opens "
                f"{limit * 1000:.1f} mm. A world that spawns objects it cannot "
                f"grasp measures a task the arm cannot perform."
            )


def _two_narrowest_sides_fit(extents: Any, limit: float) -> bool:
    """Return whether both narrowest object sides are below the jaw opening."""
    narrowest = sorted(float(extent) for extent in extents)[:2]
    return len(narrowest) == 2 and max(narrowest) < limit


def _add_conveyor_modules(
    mujoco: Any, spec: Any, world: Any, plan: SceneLayout, modules: int, root: Path
) -> bool:
    """Draw the belt as a row of conveyor modules.

    The module is published at 0.500 by 0.504 by 0.502 m with its belt surface
    on top. The shipped belt is 3.00 by 0.50 m with its surface at 0.90 m, so
    each axis is scaled independently to fit: the length so that `modules` of
    them span the belt, the width to the belt's width, and the height so the
    module's own surface lands on the belt surface. Scaling from the published
    mesh keeps the geometry this project has measured.

    The modules carry no collision geometry. An object still rests on the box
    the belt has always been, so nothing here moves a trajectory.

    Args:
        mujoco: The imported module.
        spec: The model spec, which owns meshes.
        world: The worldbody being assembled.
        plan: The resolved layout.
        modules: How many modules span the belt.
        root: Repository root.

    Returns:
        Whether the modules were drawn. They are not when the asset has not
        been imported, and the belt then keeps the box and the legs it had.

    Raises:
        WorldConfigError: If the module count is not positive.
    """
    from clave.world.config import WorldConfigError

    if modules <= 0:
        raise WorldConfigError(f"belt.modules is {modules}; it has to be positive")
    path = root / CONVEYOR_MODULE
    if not path.is_file():
        return False

    length = plan.belt.length / modules
    scale = [
        length / MODULE_SOURCE_METERS[0],
        plan.belt.width / MODULE_SOURCE_METERS[1],
        plan.belt.surface_height / MODULE_SOURCE_METERS[2],
    ]
    spec.add_mesh(name="conveyor_module", file=str(path), scale=scale)
    first = -plan.belt.length / 2.0 + length / 2.0
    for index in range(modules):
        world.add_geom(
            name=f"conveyor_module_{index}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="conveyor_module",
            pos=[first + index * length, 0.0, plan.belt.surface_height],
            contype=0,
            conaffinity=0,
        )
    return True


def _add_supports(
    mujoco: Any,
    world: Any,
    plan: SceneLayout,
    structure: dict[str, Any],
    belt_legs: bool = True,
) -> None:
    """Stand the belt and the manipulator on something.

    Everything here is static geometry under equipment that was already resting
    on nothing, so no trajectory changes: the belt surface keeps its height, the
    arm keeps its base, and the legs sit below both. It exists because a scene
    shown to an audience should read as equipment rather than as boxes floating
    over a floor.

    The supports are placed clear of the effector's workspace. The pedestal is
    under the arm base and the legs are under the belt frame, neither of which
    the arm reaches into.

    Args:
        mujoco: The imported module.
        world: The worldbody being assembled.
        plan: The resolved layout, carrying the belt and the arm base.
        structure: The `structure` section of the configuration.
        belt_legs: Whether to stand the belt on a frame and legs. False when
            the conveyor modules are drawn, since they carry their own.
    """
    leg = float(require(structure, "leg_side_meters", "structure"))
    inset = float(require(structure, "leg_inset_meters", "structure"))
    frame_depth = float(require(structure, "frame_depth_meters", "structure"))
    steel = [0.32, 0.33, 0.36, 1.0]

    if belt_legs:
        _add_belt_legs(mujoco, world, plan, leg, inset, frame_depth, steel)

    # The pedestal the manipulator is bolted to. The arm stands beside the belt
    # rather than hanging over it: a six-axis arm inverted above a plane is
    # near-singular pointing straight down, and a sweep found 3 of 27 sample
    # points reachable that way.
    #
    # A parallelepiped is the right amount of detail here. It is a machine frame
    # and nothing measures it, but an arm floating at working height describes no
    # installation anybody could build, so it has to stand on something.
    width, depth = plan.pedestal
    base_x, base_y, base_z = plan.arm_base
    # The arm sweeps an annulus from `reach_min` outward, so a pedestal whose
    # half-diagonal stays inside that radius cannot be struck by the arm it
    # carries.
    world.add_geom(
        name="arm_pedestal",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[width / 2.0, depth / 2.0, base_z / 2.0],
        pos=[base_x, base_y, base_z / 2.0],
        rgba=steel,
    )
    # A foot, so it reads as bolted down rather than balanced.
    world.add_geom(
        name="arm_pedestal_foot",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[width * 0.75, depth * 0.75, 0.008],
        pos=[base_x, base_y, 0.008],
        rgba=[0.22, 0.23, 0.26, 1.0],
    )


def _add_belt_legs(
    mujoco: Any,
    world: Any,
    plan: SceneLayout,
    leg: float,
    inset: float,
    frame_depth: float,
    steel: list[float],
) -> None:
    """Stand the belt on a frame and four uprights.

    Only for a world without the conveyor modules, which carry their own.

    Args:
        mujoco: The imported module.
        world: The worldbody being assembled.
        plan: The resolved layout.
        leg: Half the side of an upright, in meters.
        inset: How far the uprights sit inside the corners, in meters.
        frame_depth: Depth of the frame under the surface, in meters.
        steel: The color everything structural takes.
    """
    frame_top = plan.belt.surface_height - 0.04
    frame_half = frame_depth / 2.0
    world.add_geom(
        name="belt_frame",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[plan.belt.length / 2.0, plan.belt.width / 2.0, frame_half],
        pos=[0.0, 0.0, frame_top - frame_half],
        rgba=steel,
    )

    # Four uprights, inset from the corners as a bench conveyor's are.
    leg_top = frame_top - frame_depth
    for x_sign in (-1.0, 1.0):
        for y_sign in (-1.0, 1.0):
            side = "front" if x_sign < 0 else "back"
            hand = "left" if y_sign < 0 else "right"
            world.add_geom(
                name=f"belt_leg_{side}_{hand}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[leg, leg, leg_top / 2.0],
                pos=[
                    x_sign * (plan.belt.length / 2.0 - inset),
                    y_sign * (plan.belt.width / 2.0 - inset),
                    leg_top / 2.0,
                ],
                rgba=steel,
            )


ARM_GAIN_SCALE = 10.0
"""How much stiffer the arm's position actuators are made than as shipped.

The vendored model's own README says its actuator values "have not been
carefully tuned", and as shipped the arm droops 2.4 degrees at the shoulder
under its own weight, which puts the tool 31 mm off a commanded target. Scaling
the proportional and derivative terms together brings that to under 3 mm, which
is loose beside the 0.05 mm a real UR10e repeats to and tight enough that a
tracking error is not mistaken for a reach failure.

This is applied to the compiled model rather than edited into
`third_party/mujoco_menagerie_ur10e/`, because that directory is a verbatim
copy and stays one.
"""


def _stiffen_arm_actuators(mujoco: Any, model: Any) -> None:
    """Raise the arm's position gains on the compiled model.

    Args:
        mujoco: The imported module.
        model: The compiled model, modified in place.
    """
    for name in arm.ARM_ACTUATORS:
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator < 0:
            continue
        model.actuator_gainprm[actuator][0] *= ARM_GAIN_SCALE
        # The position term and its damping move together, so the response
        # stays critically damped rather than ringing.
        model.actuator_biasprm[actuator][1] *= ARM_GAIN_SCALE
        model.actuator_biasprm[actuator][2] *= math.sqrt(ARM_GAIN_SCALE)


def sensor_span_millimeters(
    camera: dict[str, Any], sensors: dict[str, Any], lenses: dict[str, Any]
) -> tuple[float, float]:
    """Return how much sensor lies across the belt and along it, in millimeters.

    A camera declares the parts it is built from rather than an angle, so the
    geometry is read off a catalog entry a buyer could order. `long_axis` says
    which belt axis the sensor's longer side spans, because a sensor laid out
    along belt travel spends its long side on a direction the object crosses
    anyway and its short side on the width it has to cover.

    Args:
        camera: One entry from the `cameras` list.
        sensors: The `sensors` catalog.
        lenses: The `lenses` catalog.

    Returns:
        The sensor extent across the belt and along it, in millimeters.

    Raises:
        ConfigError: If a key is absent, or the camera names a part the
            catalogs do not declare, or `long_axis` is neither `across` nor
            `along`.
    """
    name = str(require(camera, "sensor", "cameras"))
    if name not in sensors:
        raise WorldConfigError(
            f"camera {camera.get('id')!r} names sensor {name!r}, "
            f"which `sensors` does not declare"
        )
    pixels = require(sensors[name], "pixels", f"sensors.{name}")
    pitch = float(require(sensors[name], "pixel_pitch_micrometers", f"sensors.{name}"))
    sides = sorted(
        (int(pixels[0]) * pitch / 1000.0, int(pixels[1]) * pitch / 1000.0), reverse=True
    )
    long_axis = str(require(camera, "long_axis", "cameras"))
    if long_axis == "across":
        return sides[0], sides[1]
    if long_axis == "along":
        return sides[1], sides[0]
    raise WorldConfigError(
        f"camera {camera.get('id')!r} has long_axis {long_axis!r}; "
        f"it spans the belt either 'across' or 'along'"
    )


def field_of_view(
    camera: dict[str, Any], sensors: dict[str, Any], lenses: dict[str, Any]
) -> float:
    """Return the vertical field of view a camera's parts produce, in degrees.

    MuJoCo's `fovy` is a vertical field of view, so it sets the image height
    axis, and for a camera looking straight down that axis is the one across
    the belt. Deriving it from the sensor and the lens rather than configuring
    it is what keeps the angle from drifting away from hardware anybody could
    buy.

    Args:
        camera: One entry from the `cameras` list.
        sensors: The `sensors` catalog.
        lenses: The `lenses` catalog.

    Returns:
        The vertical field of view in degrees.

    Raises:
        ConfigError: If a key is absent, or the camera names a lens the catalog
            does not declare, or the focal length is not positive.
    """
    across, _ = sensor_span_millimeters(camera, sensors, lenses)
    name = str(require(camera, "lens", "cameras"))
    if name not in lenses:
        raise WorldConfigError(
            f"camera {camera.get('id')!r} names lens {name!r}, "
            f"which `lenses` does not declare"
        )
    focal = float(require(lenses[name], "focal_length_millimeters", f"lenses.{name}"))
    if focal <= 0.0:
        raise WorldConfigError(
            f"lens {name!r} has focal length {focal!r}, which forms no image"
        )
    return 2.0 * math.degrees(math.atan(across / (2.0 * focal)))


def along_travel_meters(
    camera: dict[str, Any],
    sensors: dict[str, Any],
    lenses: dict[str, Any],
    standoff_meters: float,
) -> float:
    """Return the belt length a nadir camera sees along travel.

    The vertical field of view is the across-belt axis when the long sensor
    side lies across the belt. The length an object crosses is the other axis.

    Args:
        camera: One entry from the `cameras` list.
        sensors: The `sensors` catalog.
        lenses: The `lenses` catalog.
        standoff_meters: Distance from the lens to the belt plane.

    Returns:
        The along-travel footprint in meters.

    Raises:
        WorldConfigError: If the standoff or the focal length cannot form an image.
    """
    if not math.isfinite(standoff_meters) or standoff_meters <= 0.0:
        raise WorldConfigError(
            f"camera {camera.get('id')!r} has standoff {standoff_meters!r}, "
            f"which does not meet the belt"
        )
    _across, along = sensor_span_millimeters(camera, sensors, lenses)
    name = str(require(camera, "lens", "cameras"))
    if name not in lenses:
        raise WorldConfigError(
            f"camera {camera.get('id')!r} names lens {name!r}, "
            f"which `lenses` does not declare"
        )
    focal = float(require(lenses[name], "focal_length_millimeters", f"lenses.{name}"))
    if focal <= 0.0:
        raise WorldConfigError(
            f"lens {name!r} has focal length {focal!r}, which forms no image"
        )
    return standoff_meters * (along / focal)


def detection_along_travel_meters(raw: dict[str, Any]) -> float:
    """Return the along-travel footprint of the first detection camera.

    Args:
        raw: A parsed world configuration.

    Returns:
        The footprint in meters.

    Raises:
        WorldConfigError: If the line has no detection camera, or its geometry
            is incomplete.
    """
    belt = require(raw, "belt")
    surface = float(require(belt, "surface_height_meters", "belt"))
    chosen: dict[str, Any] | None = None
    for camera in require(raw, "cameras"):
        if isinstance(camera, dict) and camera.get("role") == "detection":
            chosen = camera
            break
    if chosen is None:
        raise WorldConfigError("the world has no camera with role 'detection'")
    position = require(chosen, "position_meters", "cameras")
    standoff = float(position[2]) - surface
    return along_travel_meters(
        chosen, require(raw, "sensors"), require(raw, "lenses"), standoff
    )


def build(
    raw: dict[str, Any],
    rng: np.random.Generator,
    root: Path,
    presentation: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
) -> tuple[Any, Any, SceneLayout]:
    """Assemble the model.

    Args:
        raw: The parsed world configuration.
        rng: Generator used to resolve ranges and size pooled objects.
        root: Repository root, used to find the submodule.
        presentation: Extra lights, a headlight setting and a background, for a
            still that has to be legible on a web page. Default `None` adds
            nothing, so the world every dataset, training run, validation pass
            and benchmark consumes is the one this argument does not touch.
            Lighting changes what a render looks like and nothing a body does,
            but the frames a model trains on are renders, which is why this is
            an argument here rather than a key in the world configuration.
        annotations: Explanatory geometry for a published figure: the reach
            ring, the window edges, and a marker over each object in the color
            of the channel it routes to. It reaches the scene the same way
            lighting does, and default `None` adds nothing, so no dataset,
            training run or benchmark can see a marker.

    Returns:
        The compiled model, its data, and the resolved layout.

    Raises:
        FileNotFoundError: If the manipulator submodule is not checked out.
        WorldConfigError: If configuration is incomplete.
    """
    import mujoco

    LOGGER.debug("loading MuJoCo scene assets from %s", root)
    plan = layout(raw, rng)
    chutes_cfg = require(raw, "chutes")
    spawn_cfg = require(raw, "spawn")
    camera_cfg = require(raw, "cameras")
    # The force the jaw closes with is a world parameter rather than a control
    # one: it is a property of the effector, and the model has to carry it for
    # a command to respect it. Read here and applied to the vendored gripper
    # spec, which is why the number is not in the control configuration.
    closing_torque = float(
        require(require(raw, "effector"), "closing_torque_newton_meters", "effector")
    )
    if not closing_torque > 0.0:
        raise WorldConfigError(
            f"effector.closing_torque_newton_meters is {closing_torque}, and a jaw "
            f"that may not close holds nothing"
        )
    closing_torque = float(
        require(require(raw, "effector"), "closing_torque_newton_meters", "effector")
    )
    if not closing_torque > 0.0:
        raise WorldConfigError(
            f"effector.closing_torque_newton_meters is {closing_torque}, and a jaw "
            f"that may not close holds nothing"
        )
    mujoco_spec = mujoco.MjSpec()
    LOGGER.debug(
        "initializing MuJoCo world: timestep %.6f s, pool size %d, channels %s",
        plan.timestep,
        plan.pool_size,
        plan.channels,
    )
    mujoco_spec.option.timestep = plan.timestep
    # Adopt the manipulator's contact settings rather than the defaults, since
    # its grasp behavior was tuned with them.
    mujoco_spec.option.impratio = 10.0
    mujoco_spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # The manipulator ships with implicitfast, which is what its authors tuned
    # its damping and armature against. Adopting it rather than overriding it
    # keeps the arm behaving as the model it came from, and silences the attach
    # conflict that otherwise reports the scene quietly winning that argument.
    mujoco_spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # The offscreen framebuffer bounds what any renderer attached to this model
    # can produce, and MuJoCo defaults it to 640 by 480. That is enough for the
    # 320 by 240 frames the models see and not enough for a demonstration
    # video. It is a ceiling on rendering rather than a property of the world,
    # so changing it changes no trajectory and no measurement.
    #
    # It is raised to whatever the declared sensors need. A camera that cannot
    # be rendered at its own resolution has a resolution that means nothing,
    # and the 1920 floor silently capped the 2448 the shipped sensor declares.
    offscreen = max(
        (
            max(int(value) for value in require(entry, "pixels", "sensors"))
            for entry in require(raw, "sensors").values()
        ),
        default=0,
    )
    mujoco_spec.visual.global_.offwidth = max(OFFSCREEN_WIDTH, offscreen)
    mujoco_spec.visual.global_.offheight = max(OFFSCREEN_HEIGHT, offscreen)

    world = mujoco_spec.worldbody
    # The shared overhead key stays for every dataset and benchmark. A still
    # supplies its own lights and must not fight this one for the floor.
    if presentation:
        _light_for_presentation(mujoco, mujoco_spec, world, presentation)
    else:
        world.add_light(pos=[0.0, 0.0, 2.0], dir=[0.0, 0.0, -1.0])

    plan = replace(plan, dressed=_dress(mujoco, mujoco_spec, world, raw, root))

    modules = _add_conveyor_modules(
        mujoco,
        mujoco_spec,
        world,
        plan,
        int(require(require(raw, "belt"), "modules", "belt")),
        root,
    )

    # The belt an object rests on. When the conveyor modules are drawn they are
    # what a viewer sees, so this keeps its collision and stops being visible:
    # two belts in the same place is one belt and one artifact.
    half = [plan.belt.length / 2.0, plan.belt.width / 2.0, 0.02]
    world.add_geom(
        name="belt",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=[0.0, 0.0, plan.belt.surface_height - half[2]],
        rgba=[0.25, 0.25, 0.28, 0.0 if modules else 1.0],
    )

    # The modules carry their own frame and legs, so the plain belt supports
    # are only drawn for a world that has not imported them. The arm stands on
    # its own pedestal either way.
    _add_supports(mujoco, world, plan, require(raw, "structure"), belt_legs=not modules)

    # Side guides. A real sorting line has them, and without them a cylinder
    # that lands and tips simply rolls off the belt, which showed up as objects
    # recorded at lateral positions outside the belt's own width.
    rail = float(require(require(raw, "belt"), "guide_height_meters", "belt"))
    for sign in (-1.0, 1.0):
        world.add_geom(
            name=f"belt_guide_{'left' if sign < 0 else 'right'}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half[0], 0.01, rail / 2.0],
            pos=[0.0, sign * (half[1] + 0.01), plan.belt.surface_height + rail / 2.0],
            rgba=[0.30, 0.30, 0.34, 1.0],
        )

    mouths = _add_chutes(mujoco, world, plan, chutes_cfg)

    drop = require_range(spawn_cfg, "drop_height_meters", "spawn")
    for spec in plan.objects:
        if not spec.is_mesh or spec.mesh is None:
            continue
        # One mesh asset per object kind, shared by every pool slot that draws
        # it, because a mesh is bytes on disk rather than a per-slot size.
        spec_mesh = root / spec.mesh
        if not spec_mesh.is_file():
            raise FileNotFoundError(
                f"object {spec.name!r} names the mesh {spec.mesh}, which is not "
                "on disk. Run: python scripts/import_scene_assets.py --objects"
            )
        mujoco_spec.add_mesh(name=f"asset_{spec.name}", file=str(spec_mesh))
        if spec.texture is not None and (root / spec.texture).is_file():
            mujoco_spec.add_texture(
                name=f"asset_{spec.name}",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=str(root / spec.texture),
            )
            mujoco_spec.add_material(
                name=f"asset_{spec.name}", textures=["", f"asset_{spec.name}"]
            )

    markers: list[tuple[Any, str]] = []
    for index in range(plan.pool_size):
        template = plan.objects[index % len(plan.objects)]
        body = world.add_body(
            name=f"object_{index}",
            pos=[PARKED_X + 0.3 * index, PARKED_X, PARKED_Z],
        )
        markers.append((body, template.channel))
        body.add_freejoint()
        size_a = template.size[0].sample(rng)
        size_b = template.size[1].sample(rng)
        density = template.density.sample(rng)
        if template.is_mesh:
            body.add_geom(
                name=f"object_{index}_geom",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"asset_{template.name}",
                density=density,
                material=(
                    f"asset_{template.name}"
                    if template.texture is not None
                    and (root / template.texture).is_file()
                    else ""
                ),
            )
        elif template.shape == "cylinder":
            body.add_geom(
                name=f"object_{index}_geom",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[size_a, size_b, 0.0],
                density=density,
                rgba=[0.8, 0.6, 0.2, 1.0],
            )
        else:
            body.add_geom(
                name=f"object_{index}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[size_a, size_a, size_b],
                density=density,
                rgba=[0.7, 0.5, 0.3, 1.0],
            )

    # Every sensor in the configuration becomes a camera named by its id, so a
    # renderer can ask for one without knowing its index. Nothing asks for one
    # by name: the tracker resolves a camera by its role, and a test refuses a
    # camera id written as a literal anywhere under src/clave. A MuJoCo
    # camera already looks along its own negative z, so an identity orientation
    # points it straight down at the belt. Rotating it by 180 degrees about x,
    # which looks correct at a glance, aims it at the sky and renders black.
    for sensor in camera_cfg:
        position = require(sensor, "position_meters", "cameras")
        world.add_camera(
            name=str(require(sensor, "id", "cameras")),
            pos=[float(position[0]), float(position[1]), float(position[2])],
            fovy=field_of_view(sensor, require(raw, "sensors"), require(raw, "lenses")),
        )

    if annotations:
        _annotate_for_presentation(mujoco, world, plan, annotations, mouths, markers)

    frame = world.add_frame()
    # The arm stands on its pedestal, so the attachment point is the pedestal's
    # top face.
    frame.pos = list(plan.arm_base)
    with warnings.catch_warnings():
        # The manipulator declares its own impratio, cone and integrator; this
        # scene adopted all three above, so the conflict notice carries no
        # information.
        warnings.simplefilter("ignore")
        mujoco_spec.attach(_armed(root, closing_torque), prefix="arm_", frame=frame)

    model = mujoco_spec.compile()
    if presentation:
        _finish_presentation_lights(mujoco, model)
    _stiffen_arm_actuators(mujoco, model)
    _check_mesh_objects_fit(
        mujoco,
        model,
        plan,
        float(require(require(raw, "arm"), "max_grasp_width_meters", "arm")),
    )
    data = mujoco.MjData(model)
    _ = drop
    return model, data, plan
