"""Capturing a still of the sorting world.

A still is a frame of a real rollout, chosen by when it happens rather than by
arranging anything: the world runs from a seed with the scripted expert driving
the arm at approach height from the park pose, and the renderer is asked for a
frame at a stated instant. Nothing is drawn on it, composited into it or
corrected afterwards. The encoder writes the pixels the renderer produced.

The lighting and the camera come from the still's own file and are handed to
the world as an argument, so `configs/world/sorting_line.yml` is the file every
dataset, training run and benchmark reads and this changes none of it.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from clave.errors import ClaveError

ENCODER = "ffmpeg"
"""Writes the PNG. No imaging library enters the wheel for one frame."""


ARM_GAIN = 1.0
"""How much of each solved step to command.

One, meaning the solved angles are commanded outright. The arm's own position
actuators already implement a proportional-derivative loop, so interpolating the
setpoint on top of them adds a second lag, and a target riding a belt at 0.31 m/s
exposes it: at 0.35 the tool settled 1.13 m behind the object it was following,
against 35 mm when the setpoint is commanded directly.

The previous arm hid this. It was light and its targets were nearly static at the
scale its workspace covered, so a fractional setpoint looked like smoothing
rather than lag.
"""


class StillError(ClaveError):
    """A still scenario is missing a key or cannot be rendered."""


@dataclass(frozen=True)
class StillCamera:
    """One point of view on the scene.

    Attributes:
        name: Used in the output file name.
        azimuth: Degrees around the scene, measured as MuJoCo measures it.
        elevation: Degrees above the horizon, negative looking down.
        distance: Meters from the point it looks at.
        lookat: What it points at, in world meters.
        fovy: Vertical field of view in degrees.
    """

    name: str
    azimuth: float
    elevation: float
    distance: float
    lookat: NDArray[np.float64]
    fovy: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lookat", np.asarray(self.lookat, dtype=np.float64))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StillCamera):
            return NotImplemented
        return (
            self.name == other.name
            and self.azimuth == other.azimuth
            and self.elevation == other.elevation
            and self.distance == other.distance
            and bool(np.allclose(self.lookat, other.lookat, rtol=0.0, atol=1e-12))
            and self.fovy == other.fovy
        )


@dataclass(frozen=True)
class StillExpect:
    """What the capture instant must show on the belt.

    Attributes:
        packages: How many active packages the frame must carry.
        classes: How many distinct material classes those packages cover.
    """

    packages: int
    classes: int


@dataclass(frozen=True)
class StillScenario:
    """What one still captures, and how it is lit.

    Attributes:
        name: The scenario's name.
        description: What the frame is meant to show.
        seed: Seed the rollout runs from.
        capture_at_seconds: Simulated instant the frame is taken at.
        width: Frame width in pixels.
        height: Frame height in pixels.
        lighting: Extra lights, headlight and background, passed to the world
            rather than written into it.
        annotations: Explanatory geometry the scene stands up for a figure,
            passed the same way and empty by default.
        cameras: The points of view to render.
        expect: How many packages and classes the capture instant must show.
    """

    name: str
    description: str
    seed: int
    capture_at_seconds: float
    width: int
    height: int
    lighting: dict[str, Any]
    annotations: dict[str, Any]
    cameras: tuple[StillCamera, ...]
    expect: StillExpect

    @classmethod
    def load(cls, path: Path) -> StillScenario:
        """Read a still scenario from YAML.

        Args:
            path: The scenario file.

        Returns:
            The scenario.

        Raises:
            StillError: If a key is absent or no camera is declared.
        """
        raw = yaml.safe_load(path.read_text())
        still = _require(raw, "still")
        cameras = _require(raw, "cameras")
        if not isinstance(cameras, list) or not cameras:
            raise StillError("the scenario declares no camera")
        expect_raw = _require(still, "expect", "still")
        return cls(
            name=str(_require(still, "name", "still")),
            description=str(_require(still, "description", "still")).strip(),
            seed=int(_require(still, "seed", "still")),
            capture_at_seconds=float(_require(still, "capture_at_seconds", "still")),
            width=int(_require(still, "width", "still")),
            height=int(_require(still, "height", "still")),
            lighting=dict(raw.get("lighting") or {}),
            annotations=dict(raw.get("annotations") or {}),
            cameras=tuple(_camera(entry) for entry in cameras),
            expect=StillExpect(
                packages=int(_require(expect_raw, "packages", "still.expect")),
                classes=int(_require(expect_raw, "classes", "still.expect")),
            ),
        )


def _camera(entry: Any) -> StillCamera:
    """Read one camera."""
    name = str(_require(entry, "name", "cameras"))
    lookat = [float(v) for v in _require(entry, "lookat_meters", name)]
    if len(lookat) != 3:
        raise StillError(f"camera {name!r}: lookat_meters needs three numbers")
    return StillCamera(
        name=name,
        azimuth=float(_require(entry, "azimuth_degrees", name)),
        elevation=float(_require(entry, "elevation_degrees", name)),
        distance=float(_require(entry, "distance_meters", name)),
        lookat=np.asarray((lookat[0], lookat[1], lookat[2]), dtype=np.float64),
        fovy=float(_require(entry, "fovy_degrees", name)),
    )


def _require(mapping: Any, key: str, path: str = "") -> Any:
    """Read a key, failing with its location when it is absent.

    Args:
        mapping: The mapping to read from.
        key: The key required.
        path: Name of the parent, used in the error message.

    Returns:
        The value.

    Raises:
        StillError: If the key is absent.
    """
    if not isinstance(mapping, dict) or key not in mapping:
        where = f"{path}.{key}" if path else key
        raise StillError(f"required still key {where!r} is missing")
    return mapping[key]


def write_png(frame: Any, path: Path, width: int, height: int) -> None:
    """Write one rendered frame as a PNG.

    The frame goes to the encoder as raw `rgb24` and comes back as sRGB with no
    alpha, which is what the card plate expects. Nothing is drawn on it.

    Args:
        frame: The rendered frame, height by width by three bytes.
        path: Where the PNG goes.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Raises:
        StillError: If the encoder is absent or refuses the frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                ENCODER,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-i",
                "-",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                str(path),
            ],
            input=np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as error:
        raise StillError(
            f"{ENCODER} is not installed, so no still can be written"
        ) from error
    except subprocess.CalledProcessError as error:
        raise StillError(f"{ENCODER} refused the frame: {error.stderr!r}") from error


def _simulate(
    root: Path, scenario: StillScenario
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Run the world to the capture instant with the arm posed, not colliding.

    Args:
        root: Repository root.
        scenario: What to advance.

    Returns:
        The MuJoCo module, model, data, plan, conveyor and arm indices.

    Raises:
        StillError: If the belt never enters the arm's reach.
    """
    import os

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco

    from clave.control.settings import ControlSettings
    from clave.data.examples import ObjectLabel
    from clave.data.expert import decide
    from clave.world import arm as armmod
    from clave.world import belt, config, scene

    raw = config.load(root / "configs" / "world" / "sorting_line.yml")
    control = ControlSettings.load(
        config.load(root / "configs" / "runtime" / "control.yml")
    )
    rng = np.random.default_rng(scenario.seed)
    model, data, plan = scene.build(
        raw,
        rng,
        root,
        presentation=scenario.lighting,
        annotations=scenario.annotations,
    )
    spawn = config.require(raw, "spawn")
    conveyor = belt.Conveyor(
        plan,
        rng,
        config.require_range(spawn, "spacing_meters", "spawn"),
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(config.require(spawn, "entry_margin_meters", "spawn")),
    )
    indices = armmod.locate(
        model,
        armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
    )
    exit_coordinate = belt.window_exit(plan)
    if exit_coordinate is None:
        raise StillError(
            "the belt never enters the arm's reach, so no still can be captured"
        )
    _park_the_arm(mujoco, model, data, indices, control.task.park_position)
    _let_the_arm_pass_through(mujoco, model)
    approach_z = float(plan.belt.surface_height) + float(control.task.approach_height)

    for _ in range(int(scenario.capture_at_seconds / plan.timestep)):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
        labels = []
        for item in conveyor.active:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
            address = model.jnt_qposadr[model.body_jntadr[body]]
            position = np.asarray(data.qpos[address : address + 3], dtype=np.float64)
            labels.append(
                ObjectLabel(
                    object_id=item.index,
                    material_class=item.material_class,
                    channel=item.channel,
                    position=position,
                    in_reachable_window=belt.within_reach(position, plan),
                )
            )
        chosen = decide(tuple(labels), exit_coordinate)
        if chosen is None:
            continue
        target = np.asarray(
            (float(chosen.position[0]), float(chosen.position[1]), approach_z),
            dtype=np.float64,
        )
        if armmod.reachable(indices, target):
            armmod.step_toward(model, data, indices, target, gain=ARM_GAIN)
    return mujoco, model, data, plan, conveyor, indices


def capture(root: Path, scenario: StillScenario, out: Path) -> list[Path]:
    """Run the rollout to its instant and render every camera.

    The arm starts at the configured park pose and the scripted expert drives
    the flange at approach height over the reachable package nearest the
    window exit (`AC-STILL-05`). Chasing the object's centre of mass buried
    the tool under the belt and left it beside the line; the approach height
    is what puts a figure's arm over the packages a reader expects to see.

    Args:
        root: Repository root.
        scenario: What to capture.
        out: Directory the PNGs go in.

    Returns:
        The files written, in camera order.

    Raises:
        StillError: If the capture instant does not match the scenario's
            package and class claim (`AC-STILL-04`), or the flange is not
            serving the belt (`AC-STILL-05`).
    """
    mujoco, model, data, plan, conveyor, indices = _simulate(root, scenario)
    packages = len(conveyor.active)
    classes = len({item.material_class for item in conveyor.active})
    if packages != scenario.expect.packages or classes != scenario.expect.classes:
        raise StillError(
            f"capture at {scenario.capture_at_seconds:.2f} s on seed "
            f"{scenario.seed} shows {packages} packages and {classes} classes, "
            f"but still.expect asks for {scenario.expect.packages} packages and "
            f"{scenario.expect.classes} classes"
        )
    _refuse_if_arm_misses_the_belt(mujoco, model, data, indices, plan, conveyor)

    renderer = mujoco.Renderer(model, height=scenario.height, width=scenario.width)
    written: list[Path] = []
    for camera in scenario.cameras:
        view = mujoco.MjvCamera()
        view.azimuth = camera.azimuth
        view.elevation = camera.elevation
        view.distance = camera.distance
        view.lookat[:] = camera.lookat
        model.vis.global_.fovy = camera.fovy
        renderer.update_scene(data, camera=view)
        path = out / f"{scenario.name}-{camera.name}.png"
        write_png(renderer.render(), path, scenario.width, scenario.height)
        written.append(path)
    return written


def _let_the_arm_pass_through(mujoco: Any, model: Any) -> None:
    """Keep the posed arm from sweeping packages off the belt.

    The still commands the flange through the packages it is photographing.
    Contact resolution is not the same on every machine, so a colliding arm
    ejects a different number of them and the belt claim stops matching the
    capture. The arm is scenery here: it has to be seen over a package, not
    to move one.

    Args:
        mujoco: The imported module.
        model: The compiled model, modified in place.
    """
    for geom in range(model.ngeom):
        # The link meshes ship without a geom name. The body name is what
        # marks them as the arm; a named pad is the other case.
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
        )
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        belongs = (body is not None and body.startswith("arm_")) or (
            name is not None and name.startswith("arm_")
        )
        if belongs:
            model.geom_contype[geom] = 0
            model.geom_conaffinity[geom] = 0


def _park_the_arm(
    mujoco: Any,
    model: Any,
    data: Any,
    indices: Any,
    park: NDArray[np.float64],
) -> None:
    """Solve the park pose and write it into the simulation.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state, modified in place.
        indices: The arm indices.
        park: Where the arm rests.

    Raises:
        StillError: If the park pose does not solve.
    """
    from clave.world import arm as armmod

    armmod._TRACKING.pop((id(model), id(data)), None)
    try:
        angles = armmod.solve(model, data, indices, np.asarray(park, dtype=float))
    except armmod.ReachError as error:
        raise StillError(f"the park pose does not solve: {error}") from error
    for slot, joint in enumerate(indices.joint_ids):
        data.qpos[model.jnt_qposadr[joint]] = angles[slot]
        data.ctrl[indices.actuator_ids[slot]] = angles[slot]
    mujoco.mj_forward(model, data)


# How close the flange may sit from a reachable package in the belt plane.
# Packages on this line are a few centimetres across; half a package away is
# still clearly serving that object rather than standing beside the line.
_BELT_SERVICE_RADIUS_METERS = 0.08


def _refuse_if_arm_misses_the_belt(
    mujoco: Any,
    model: Any,
    data: Any,
    indices: Any,
    plan: Any,
    conveyor: Any,
) -> None:
    """Refuse a capture whose flange is not serving a reachable package.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state, with forward kinematics current.
        indices: The arm indices.
        plan: The resolved layout.
        conveyor: The belt with its active packages.

    Raises:
        StillError: If the flange is below the belt, outside the annulus, or
            farther than [_BELT_SERVICE_RADIUS_METERS] from every reachable
            package in the belt plane (`AC-STILL-05`).
    """
    from clave.world import arm as armmod
    from clave.world import belt

    flange = armmod.end_effector_position(data, indices)
    if float(flange[2]) < float(plan.belt.surface_height):
        raise StillError(
            f"flange sits at z={float(flange[2]):.3f} m, below the belt surface "
            f"at {float(plan.belt.surface_height):.3f} m"
        )
    if not armmod.reachable(indices, flange):
        raise StillError(
            f"flange at {np.round(flange, 3).tolist()} is outside the trusted reach"
        )
    nearest = math.inf
    for item in conveyor.active:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
        place = np.asarray(data.xpos[body], dtype=np.float64)
        if not belt.within_reach(place, plan):
            continue
        gap = math.hypot(float(flange[0] - place[0]), float(flange[1] - place[1]))
        nearest = min(nearest, gap)
    if math.isinf(nearest):
        # Nothing reachable to serve; the arm may stay parked.
        return
    if nearest > _BELT_SERVICE_RADIUS_METERS:
        raise StillError(
            f"flange is {nearest:.3f} m from the nearest reachable package in "
            f"the belt plane, past the {_BELT_SERVICE_RADIUS_METERS:.2f} m "
            f"service radius"
        )
