"""Recording rollouts from the simulated world.

Stepping the world and capturing frames is mechanical. What matters is that
every captured frame carries the labels the world already holds, so the dataset
is labeled by construction rather than by a later pass.

Rendering runs offscreen on CPU through OSMesa, because this machine has no
usable accelerator. That costs roughly 52 milliseconds a frame at 320 by 240.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.data.examples import CameraCapture, Example, ObjectLabel, Origin, Rollout
from clave.data.expert import decide
from clave.errors import ClaveError
from clave.tracker.evidence import Role
from clave.tracker.sensors import load_sensors, of_role, require_role
from clave.world import arm as armmod
from clave.world import belt, config, scene
from clave.world.config import Range

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


class RecordingError(ClaveError):
    """A world cannot produce the demonstrations a recording needs."""


def _ensure_software_rendering() -> None:
    """Select the CPU rendering backend when none was chosen.

    Setting this after MuJoCo has initialized its GL context has no effect, so
    it happens before the first import inside this module's functions.
    """
    os.environ.setdefault("MUJOCO_GL", "osmesa")


def _boxes_from_segmentation(
    model: Any, segmentation: Any
) -> dict[str, tuple[int, int, int, int]]:
    """Derive exact pixel bounds per object from a segmentation render.

    MuJoCo renders geometry ids per pixel, so the bounds are ground truth rather
    than a projection estimate, and an object absent from the render is simply
    absent from the result.

    Args:
        model: The compiled model, used to map geometry ids to names.
        segmentation: The segmentation buffer, height by width by two.

    Returns:
        Object body name to pixel bounds as (x_min, y_min, x_max, y_max).
    """
    import mujoco
    import numpy as np

    geom_ids = segmentation[:, :, 0]
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for geom_id in np.unique(geom_ids):
        if geom_id < 0 or geom_id >= model.ngeom:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        if name is None or not name.startswith("object_"):
            continue
        rows, columns = np.where(geom_ids == geom_id)
        boxes[name.removesuffix("_geom")] = (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        )
    return boxes


def _instance_map(
    model: Any,
    segmentation: Any,
    serial_by_body: dict[str, int],
) -> NDArray[np.uint16]:
    """Paint each object pixel with that object's spawn serial.

    MuJoCo's segmentation buffer carries geometry ids. The serial is what a
    later frame can match, because the pool slot those geometry names are
    built from gets reused.
    """
    import mujoco

    geom_ids = segmentation[:, :, 0]
    canvas = np.zeros(geom_ids.shape[:2], dtype=np.uint16)
    for geom_id in np.unique(geom_ids):
        if geom_id < 0 or geom_id >= model.ngeom:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        if name is None or not name.startswith("object_"):
            continue
        serial = serial_by_body.get(name.removesuffix("_geom"))
        if serial is None:
            continue
        if serial > np.iinfo(np.uint16).max:
            raise RecordingError(
                f"spawn serial {serial} does not fit in the instance map"
            )
        rows, columns = np.where(geom_ids == geom_id)
        canvas[rows, columns] = np.uint16(serial)
    return canvas


def _labels_for(
    model: Any,
    data: Any,
    conveyor: belt.Conveyor,
    boxes: dict[str, tuple[int, int, int, int]],
) -> tuple[ObjectLabel, ...]:
    """Read every active object's label straight from the world."""
    import mujoco

    labels: list[ObjectLabel] = []
    for item in conveyor.active:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
        address = model.jnt_qposadr[model.body_jntadr[body]]
        velocity = model.jnt_dofadr[model.body_jntadr[body]]
        position = np.asarray(data.qpos[address : address + 3], dtype=np.float64)
        pose = data.qpos[address + 3 : address + 7]
        speed = data.qvel[velocity : velocity + 3]
        spin = data.qvel[velocity + 3 : velocity + 6]
        labels.append(
            ObjectLabel(
                object_id=item.serial,
                material_class=item.material_class,
                channel=item.channel,
                position=position,
                in_reachable_window=belt.within_reach(position, conveyor.plan),
                bbox=boxes.get(item.name),
                orientation=(
                    float(pose[0]),
                    float(pose[1]),
                    float(pose[2]),
                    float(pose[3]),
                ),
                linear_velocity=(float(speed[0]), float(speed[1]), float(speed[2])),
                angular_velocity=(float(spin[0]), float(spin[1]), float(spin[2])),
                object_name=item.object_name,
            )
        )
    return tuple(labels)


def record(
    root: Path,
    seed: int,
    seconds: float,
    capture_interval_seconds: float,
    height: int,
    width: int,
    rollout_id: str,
    *,
    belt_speed: float | None = None,
    spacing_meters: float | None = None,
    drive_arm: bool = True,
    cameras: tuple[str, ...] | None = None,
) -> Rollout:
    """Run the world once and capture labeled frames.

    Args:
        root: Repository root, used to find the configuration and the submodule.
        seed: Seed controlling belt speed, placement and spawn timing.
        seconds: Simulated seconds to run.
        capture_interval_seconds: Simulated seconds between captures.
        height: Frame height in pixels.
        width: Frame width in pixels.
        rollout_id: Identity for this rollout within a dataset.
        belt_speed: Meters per second to hold, instead of the speed the world
            draws. None leaves the draw in place.
        spacing_meters: Metres of belt between releases. None leaves the
            world's range in place, so each gap is drawn.
        drive_arm: Whether to move the arm toward the scripted expert. A corpus
            records the stream with the arm parked.
        cameras: Detection camera ids to store. None stores the first detection
            camera only, which is what `record-dataset` has always done.

    Returns:
        The rollout, with one example per captured frame.
    """
    _ensure_software_rendering()
    import mujoco

    raw = config.load(root / "configs" / "world" / "sorting_line.yml")
    config_digest = _config_digest(root)
    rng = np.random.default_rng(seed)
    model, data, plan = scene.build(raw, rng, root)
    spawn = config.require(raw, "spawn")
    spacing = (
        Range(spacing_meters, spacing_meters)
        if spacing_meters is not None
        else config.require_range(spawn, "spacing_meters", "spawn")
    )
    conveyor = belt.Conveyor(
        plan,
        rng,
        spacing,
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(config.require(spawn, "entry_margin_meters", "spawn")),
    )
    if belt_speed is not None:
        conveyor.speed = belt_speed
    # The swept downstream edge, not half the window's length. Those agree only
    # while the arm stands at the belt centre, and the expert replayed below
    # ranks objects by their distance to this coordinate.
    sensors = load_sensors(raw)
    detection = of_role(sensors, Role.DETECTION)
    camera_ids: tuple[str, ...]
    if cameras is None:
        camera_ids = (require_role(sensors, Role.DETECTION).source_id,)
        store_all = False
    else:
        known = {sensor.source_id for sensor in detection}
        missing = [camera for camera in cameras if camera not in known]
        if missing:
            raise RecordingError(
                f"cameras {missing} are not detection cameras; the line has "
                f"{sorted(known)}"
            )
        camera_ids = cameras
        store_all = True
    exit_coordinate = belt.window_exit(conveyor.plan)
    if exit_coordinate is None:
        raise RecordingError(
            "the belt never enters the arm's reach, so no demonstration can be "
            "recorded from this world"
        )

    indices = armmod.locate(
        model,
        armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
    )
    renderer = mujoco.Renderer(model, height=height, width=width)
    segmenter = mujoco.Renderer(model, height=height, width=width)
    segmenter.enable_segmentation_rendering()
    try:
        return _capture_rollout(
            model,
            data,
            conveyor,
            indices,
            renderer,
            segmenter,
            seconds=seconds,
            capture_interval_seconds=capture_interval_seconds,
            seed=seed,
            config_digest=config_digest,
            camera_ids=camera_ids,
            store_all=store_all,
            drive_arm=drive_arm,
            exit_coordinate=exit_coordinate,
            rollout_id=rollout_id,
            spacing_meters=spacing_meters,
            timestep=plan.timestep,
        )
    finally:
        renderer.close()
        segmenter.close()


def _capture_rollout(
    model: Any,
    data: Any,
    conveyor: Any,
    indices: Any,
    renderer: Any,
    segmenter: Any,
    *,
    seconds: float,
    capture_interval_seconds: float,
    seed: int,
    config_digest: str,
    camera_ids: tuple[str, ...],
    store_all: bool,
    drive_arm: bool,
    exit_coordinate: float,
    rollout_id: str,
    spacing_meters: float | None,
    timestep: float,
) -> Rollout:
    """Step one rollout and return its labeled frames."""
    import mujoco

    examples: list[Example] = []
    next_capture = 0.0
    for _ in range(int(seconds / timestep)):
        mujoco.mj_step(model, data)
        conveyor.step(model, data)
        # A corpus leaves the arm parked. Driving it here would record the
        # expert's motion, which this dataset is not.
        if drive_arm:
            labels_now = _labels_for(model, data, conveyor, {})
            chosen = decide(labels_now, exit_coordinate)
            if chosen is not None:
                armmod.step_toward(
                    model, data, indices, np.array(chosen.position), gain=ARM_GAIN
                )
        if data.time < next_capture:
            continue
        serials = {item.name: item.serial for item in conveyor.active}
        captures: list[CameraCapture] = []
        for camera_id in camera_ids:
            renderer.update_scene(data, camera=camera_id)
            segmenter.update_scene(data, camera=camera_id)
            segmentation = segmenter.render()
            boxes = _boxes_from_segmentation(model, segmentation)
            captures.append(
                CameraCapture(
                    camera_id=camera_id,
                    frame=renderer.render().astype(np.uint8),
                    labels=_labels_for(model, data, conveyor, boxes),
                    instance_ids=_instance_map(model, segmentation, serials),
                )
            )
        primary = captures[0]
        examples.append(
            Example(
                frame=primary.frame,
                labels=primary.labels,
                simulated_time=float(data.time),
                seed=seed,
                config_digest=config_digest,
                origin=Origin.SIMULATED,
                arm_joints=tuple(
                    float(angle)
                    for angle in armmod.joint_positions(model, data, indices)
                ),
                captures=tuple(captures) if store_all else (),
            )
        )
        next_capture = data.time + capture_interval_seconds
    return Rollout(
        rollout_id=rollout_id,
        seed=seed,
        examples=tuple(examples),
        belt_speed=conveyor.running,
        spacing_meters=spacing_meters,
    )


def _config_digest(root: Path) -> str:
    """Digest the world configuration that produced a rollout.

    Two datasets recorded under different world configurations are different
    datasets even at the same seed, so the digest travels with every example.
    """
    from clave.corpus.artifacts import digest_of

    return digest_of(root / "configs" / "world" / "sorting_line.yml")
