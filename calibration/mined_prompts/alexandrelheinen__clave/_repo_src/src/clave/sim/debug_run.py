"""Driving the tracker over a rollout and showing what it decided.

This is a diagnostic, not a demonstration. It drives `clave.tracker` directly
rather than through `clave.runtime.loop`, because the tracker is deliberately
not wired into the loop yet.

What a reader sees is the world, with a grasp marker standing where the
tracker believes each object is and turned the way a jaw would have to close
on it. The markers are scene geometry rather than paint on a frame, so MuJoCo
shades and occludes them like any other solid, and nothing touches the image
after the renderer is finished with it.

The sensing cameras and the watching camera never mix. The tracker segments
every camera carrying the detection role, because those are the sensors the
line actually has, and it is handed nothing else. The human watches a view
named in `configs/debug/tracker.yml`, because a grasp pose seen from
straight above has no approach to read. That file carries two views and the
run takes one by name: one camera cannot both read a 60 mm jaw and hold the
park pose in frame.

`--camera-video` writes a second file of the primary detection camera, with a
box on each object a segmentation of that same frame covered. That render
stays off unless it is asked for (`AC-CAM-01`).

**The report separates active motion from grasping.** An arm can reach the pose it
was sent to perfectly and still hold nothing, which is what happens when the
pose is not where the object is, so the run reports the distance to the
commanded pose, the distance from the jaw to the nearest object at the
instant it shuts, and how far each visit actually lifted anything. Only the
last of those is a grasp, and reading the first as one is how a broken pick
looks solved.

A window opens when a display is available and the run writes its artifacts
either way, because a machine with no display is the ordinary case for this
project and refusing to run on one would make the view useless exactly where
it is needed most.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
        return iterable


import numpy as np
from numpy.typing import NDArray

from clave.control.motion import Command, Reference, toward
from clave.control.pick import JAW_OPEN
from clave.control.selection import Selector, pickable_in_belt
from clave.control.servo import follow
from clave.control.settings import ControlSettings, Phase
from clave.control.story import AnomalyWatch, StoryLog
from clave.control.task import TaskMachine
from clave.control.trajectory import distance, norm, zeros
from clave.sim.clearance import (
    JawState,
    _jaw_collision_geoms,
    _jaw_state,
    _mesh_vertices,
)
from clave.sim.overlay import (
    _camera_recorder,
    _flange,
    _lookat,
    _markers_on,
    _painted,
    _pinch,
    _recorder,
    _trajectory_on,
    _view,
    _without_shadows,
    _write_beliefs,
    _write_camera_frame,
)
from clave.sim.report import (
    GRASPED_METERS,
    DebugRunError,
    DebugRunReport,
    _report_document,
    _run_metadata,
    report_lines,
)
from clave.taxonomy import channel_of
from clave.tracker.adapters.detection import detections_from_masks
from clave.tracker.adapters.render import segment_masks
from clave.tracker.association import SimulatorIdentity
from clave.tracker.codes import load_catalog, resolver_for
from clave.tracker.evidence import Evidence, GroundTruth, Role
from clave.tracker.fusion import FusionSettings
from clave.tracker.intake import Deployment, Intake
from clave.tracker.markers import (
    ground_truth_markers,
    markers_for,
)
from clave.tracker.sensors import load_sensors, of_role, require_role
from clave.tracker.track import Tracker
from clave.world import arm as armmod
from clave.world import belt, config, scene
from clave.world.effector import Effector
from clave.world.feed import FeedSettings, RateController

LOGGER = logging.getLogger(__name__)

NANOS_PER_SECOND = 1_000_000_000
"""Nanoseconds in a second, for the instants an observation carries."""

_VISITING = frozenset({Phase.TRACK, Phase.DESCEND, Phase.RETREAT})
"""The phases that are about an object rather than about going home."""


class _TelemetryWriter:
    """Write fixed-width MuJoCo telemetry columns for PlotJuggler."""

    def __init__(self, path: Path, model: Any, pool_size: int, rate: float) -> None:
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._period = 1.0 / rate
        self.next_sample = 0.0
        self._pool_size = pool_size
        self._headers = [
            "time",
            "belt_speed",
            "gripper_command",
            "flange_x",
            "flange_y",
            "flange_z",
            "pinch_x",
            "pinch_y",
            "pinch_z",
            "phase_code",
            "task_active",
            "commanded_x",
            "commanded_y",
            "commanded_z",
            "commanded_yaw",
            "jaw_lowest_z",
            "jaw_clearance",
            "belt_contact",
            "tool_tilt_degrees",
        ]
        self._headers.extend(f"qpos_{index}" for index in range(model.nq))
        self._headers.extend(f"qvel_{index}" for index in range(model.nv))
        self._headers.extend(f"ctrl_{index}" for index in range(model.nu))
        for index in range(pool_size):
            self._headers.extend(
                (f"object_{index}_x", f"object_{index}_y", f"object_{index}_z")
            )
        self._writer.writerow(self._headers)

    def write(
        self,
        model: Any,
        data: Any,
        indices: Any,
        conveyor: Any,
        belt_speed: float,
        phase: Phase,
        active: bool,
        jaw: JawState,
        command: Any = None,
    ) -> None:
        """Write the current state when the requested sample period is due.

        Written after the tick's command has been decided rather than before it,
        so the phase and the commanded pose in a row are the ones that tick
        commanded. Written before, the row carried the previous tick's phase,
        which is a millisecond of nothing on a 500 Hz tick and a whole leg on a
        coarse one: a run whose `DESCEND` column spanned 0.79 s against a
        0.40 s leg is what that looks like from outside.
        """
        if float(data.time) + 1e-12 < self.next_sample:
            return
        flange = _flange(indices, data)
        pinch = _pinch(indices, data)
        if command is None:
            commanded: list[object] = ["", "", "", ""]
        else:
            commanded = [*command.position, "" if command.yaw is None else command.yaw]
        row: list[object] = [
            float(data.time),
            belt_speed,
            float(data.ctrl[indices.gripper_actuator]),
            *flange,
            *pinch,
            list(Phase).index(phase),
            int(active),
            *commanded,
            jaw.lowest_z,
            jaw.clearance,
            int(jaw.contact),
            jaw.tilt_degrees,
            *[float(value) for value in data.qpos],
            *[float(value) for value in data.qvel],
            *[float(value) for value in data.ctrl],
        ]
        positions = {item.index: item.name for item in conveyor.active}
        for index in range(self._pool_size):
            name = positions.get(index)
            if name is None:
                row.extend(("", "", ""))
                continue
            body = mujoco_body_id(model, name)
            address = model.jnt_qposadr[model.body_jntadr[body]]
            row.extend(float(value) for value in data.qpos[address : address + 3])
        self._writer.writerow(row)
        self._file.flush()
        self.next_sample += self._period

    def close(self) -> None:
        """Flush and close the telemetry file."""
        self._file.close()


def _progress(steps: int, enabled: bool) -> Any:
    """Return the physics-step iterator, with a bar only when one was asked for.

    The bar and a debug log share one stream. ``--no-progress`` is how an
    operator reads the narrative without the bar redrawing over it.

    Args:
        steps: How many physics steps the run takes.
        enabled: Whether to draw the bar.

    Returns:
        A tqdm bar, or a plain range when the bar is off. A machine with no
        tqdm installed gets the plain range either way.
    """
    if not enabled:
        return range(steps)
    return tqdm(range(steps), desc="Simulating", unit="step")


def mujoco_body_id(model: Any, name: str) -> int:
    """Return a body id without importing MuJoCo at module import time."""
    import mujoco

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def run(
    root: Path,
    out: Path,
    seconds: float = 14.0,
    seed: int = 0,
    capture_interval: float = 0.5,
    render: tuple[int, int] = (640, 480),
    window: bool = True,
    video: bool = False,
    camera_video: bool = False,
    frames: bool | None = None,
    fps: int | None = None,
    view_name: str | None = None,
    telemetry_path: Path | None = None,
    telemetry_rate: float = 100.0,
    trajectory_seconds: float = 2.0,
    ground_truth_tracker: bool | None = None,
    belt_speed: float | None = None,
    progress: bool = True,
    render_mode: str = "lite",
) -> DebugRunReport:
    """Drive the tracker over one rollout, annotating every capture.

    Args:
        root: Repository root.
        out: Directory the annotated frames go to. Its own, never the one the
            published figures use.
        seconds: Simulated seconds to run.
        seed: Seed controlling belt speed, placement and spawn timing.
        capture_interval: Simulated seconds between captures.
        render: Frame size as `(width, height)` in pixels.
        window: Open a live window when a display is available.
        video: Encode the rendered frames into a playable file beside them.
            A machine with no `ffmpeg` runs anyway and says none was written.
        camera_video: Encode the primary detection camera, with a box on each
            object a segmentation of that same frame covered (`AC-CAM-01`).
            Off unless asked, because the render is the expensive part. A
            machine with no `ffmpeg` runs anyway and says none was written.
        frames: Write annotated PNG captures. When None, defaults to on if a
            window or video was asked for, and off for a headless run with
            neither (`AC-PERF-02`).
        fps: Playback rate of that file, or None for the rate the debug
            configuration names.
        view_name: Which view in the debug configuration to film from, or
            None for the one that file names as its default.
        telemetry_path: CSV output path, or None to disable telemetry.
        telemetry_rate: Telemetry samples per simulated second.
        trajectory_seconds: Future portion of the active plan to draw.
        ground_truth_tracker: Feed downstream selection and planning from
            MuJoCo ground-truth physics instead of tracker estimates, or None
            to read the debug configuration.
        belt_speed: Override belt speed in meters per second, or None to use
            the speed sampled by the scene layout.
        progress: Draw the tqdm bar. Off when the operator wants the debug
            narrative, which shares the terminal with the bar.

    Returns:
        The report.

    Raises:
        DebugRunError: If the world declares no detection camera.
    """
    write_frames = bool(window or video) if frames is None else frames
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    render_mode = render_mode.lower()
    if render_mode not in ("lite", "demo", "realistic"):
        raise DebugRunError(
            f"unknown render_mode {render_mode!r}. Options: 'lite', 'demo', 'realistic'"
        )

    LOGGER.debug(
        "initializing tracker debug run: root=%s out=%s seconds=%.3f seed=%d "
        "capture_interval=%.3f ground_truth_tracker=%s frames=%s render_mode=%s",
        root,
        out,
        seconds,
        seed,
        capture_interval,
        ground_truth_tracker,
        write_frames,
        render_mode,
    )
    import cv2
    import mujoco

    from clave.tracker.sensors import SensorError

    world_path = root / "configs" / "world" / "sorting_line.yml"
    control_path = root / "configs" / "runtime" / "control.yml"
    debug_path = root / "configs" / "debug" / "tracker.yml"
    fusion_path = root / "configs" / "perception" / "fusion.yml"
    packaging_path = root / "configs" / "perception" / "packaging.yml"
    LOGGER.debug(
        "loading simulation configs: world=%s control=%s debug=%s fusion=%s "
        "packaging=%s",
        world_path,
        control_path,
        debug_path,
        fusion_path,
        packaging_path,
    )
    raw = config.load(world_path)
    sensors = load_sensors(raw)
    try:
        # Every detection camera, not the first one. A line with a single
        # camera over the sensing gate can only ever hand the arm an estimate
        # that has been dead reckoned since the object left that gate, and
        # what the belt model does not predict is exactly what a jaw closing
        # on 8.7 mm of side clearance cannot absorb.
        detection_spec = require_role(sensors, Role.DETECTION)
        detection_camera = detection_spec.source_id
        detecting = of_role(sensors, Role.DETECTION)
    except SensorError as error:
        raise DebugRunError(str(error)) from error

    surface = float(
        config.require(config.require(raw, "belt"), "surface_height_meters", "belt")
    )
    grasp = float(
        config.require(config.require(raw, "arm"), "max_grasp_width_meters", "arm")
    )
    effector = Effector.load(raw)
    control = ControlSettings.load(config.load(control_path))
    debug = config.load(debug_path)
    view = _view(debug, view_name)
    use_ground_truth = (
        ground_truth_tracker
        if ground_truth_tracker is not None
        else bool(debug.get("ground_truth_tracker", False))
    )

    presentation = (
        scene.DEFAULT_DEMO_PRESENTATION
        if render_mode in ("demo", "realistic")
        else None
    )
    model, data, plan = scene.build(
        raw, np.random.default_rng(seed), root, presentation=presentation
    )
    if belt_speed is not None:
        plan = replace(plan, belt=replace(plan.belt, speed=belt_speed))
    spawn = config.require(raw, "spawn")
    conveyor = belt.Conveyor(
        plan,
        np.random.default_rng(seed),
        config.require_range(spawn, "spacing_meters", "spawn"),
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(config.require(spawn, "entry_margin_meters", "spawn")),
    )
    # The belt speed the layout drew is where the loop starts, not where it
    # stays. Its range becomes the drive's limits rather than a distribution.
    drive = config.require_range(
        config.require(raw, "belt"), "speed_meters_per_second", "belt"
    )
    feeding = RateController(
        settings=FeedSettings.load(raw),
        lowest=min(drive.low, plan.belt.speed),
        highest=max(drive.high, plan.belt.speed),
        speed=plan.belt.speed,
    )

    tracker = Tracker(
        intake=Intake(Deployment.SIMULATED, sensors),
        associator=SimulatorIdentity(),
        settings=FusionSettings.load(fusion_path, world_path),
        belt_speed=plan.belt.speed,
        window_exit=belt.window_exit(plan) or 1.034,
        # Seeded from the layout and updated every capture below, because the
        # feed controller moves the belt and a prediction made with the speed
        # the run drew is wrong by however far the controller has trimmed it.
        unmeasured_extent=grasp,
        resolve=resolver_for(load_catalog(packaging_path)),
    )

    LOGGER.debug(
        "resolved simulation parameters: belt_speed=%.3f m/s surface=%.3f m "
        "jaw_opening=%.3f m render=%dx%d view=%s ground_truth_tracker=%s",
        plan.belt.speed,
        surface,
        grasp,
        render[0],
        render[1],
        view_name or "default",
        use_ground_truth,
    )

    width, height = render
    masks = mujoco.Renderer(model, height=height, width=width)
    masks.enable_segmentation_rendering()

    opened, reason = _open_window(cv2, window)
    # Watching RGB is only for PNG dumps, live window, or video (`AC-PERF-01`).
    needs_rgb = write_frames or opened or video
    watching = (
        mujoco.Renderer(
            model,
            height=int(config.require(view, "height", "view")),
            width=int(config.require(view, "width", "view")),
        )
        if needs_rgb
        else None
    )
    eye = mujoco.MjvCamera()
    eye.azimuth = float(config.require(view, "azimuth_degrees", "view"))
    eye.elevation = float(config.require(view, "elevation_degrees", "view"))
    eye.distance = float(config.require(view, "distance_meters", "view"))
    eye.lookat[:] = _lookat(view)

    out.mkdir(parents=True, exist_ok=True)
    tracker_cam = mujoco.Renderer(model, height=180, width=240) if opened else None
    start_wall = time.perf_counter()

    indices = armmod.locate(
        model,
        armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
    )
    jaw_geoms = _jaw_collision_geoms(model, indices)
    belt_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "belt"))
    telemetry = None
    if telemetry_path is not None:
        if telemetry_rate <= 0.0:
            raise DebugRunError(f"telemetry rate {telemetry_rate} must be above zero")
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry = _TelemetryWriter(
            telemetry_path,
            model,
            plan.pool_size,
            telemetry_rate,
        )
        LOGGER.info(
            "telemetry enabled: path=%s rate=%.3f Hz",
            telemetry_path,
            telemetry_rate,
        )
    else:
        LOGGER.info("telemetry disabled")

    metadata_path = out / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            _run_metadata(
                root,
                {
                    "world": world_path,
                    "control": control_path,
                    "debug": debug_path,
                    "fusion": fusion_path,
                    "packaging": packaging_path,
                },
                {
                    "seconds": seconds,
                    "seed": seed,
                    "capture_interval_seconds": capture_interval,
                    "render": list(render),
                    "view": view_name or str(debug.get("default_view", "")),
                    "video": video,
                    "fps": fps,
                    "frames": write_frames,
                    "telemetry_rate": telemetry_rate,
                    "trajectory_seconds": trajectory_seconds,
                    "ground_truth_tracker": use_ground_truth,
                    "belt_speed_override": belt_speed,
                    "window": window,
                    "progress": progress,
                    "render_mode": render_mode,
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    LOGGER.info("run metadata written: %s", metadata_path)

    if trajectory_seconds < 0.0:
        raise DebugRunError(
            f"trajectory horizon {trajectory_seconds} cannot be negative"
        )
    # Start the arm parked. The model's own initial configuration leaves the
    # flange below the trusted vertical band, so the first pose the controller
    # commands is refused for a reason that has nothing to do with the pose: a
    # line starts with its arm at rest, and so does this.
    _park_the_arm(mujoco, model, data, indices, control.task.park_position)

    base_xy = (float(indices.base_position[0]), float(indices.base_position[1]))

    def keep_inside(pose: NDArray[np.float64]) -> NDArray[np.float64]:
        """Push a pose back into the region the arm is trusted over.

        All three axes. Correcting only the horizontal ones leaves a
        reference reseeded from a flange that has overshot the vertical band
        still outside it, and every pose after that is refused.

        Args:
            pose: The pose.

        Returns:
            The pose, unchanged where it was already inside.
        """
        x, y = armmod.project_into_reach(
            base_xy, float(pose[0]), float(pose[1]), indices.reach
        )
        return np.asarray(
            (
                x,
                y,
                armmod.project_into_band(
                    float(indices.base_position[2]), float(pose[2]), indices.reach
                ),
            ),
            dtype=np.float64,
        )

    def admits(pose: NDArray[np.float64]) -> bool:
        """Whether the arm is trusted at a pose.

        Args:
            pose: The pose.

        Returns:
            Whether it lies inside the region the safety layer enforces.
            Selection and planning ask the same question of the same
            function, so the queue cannot offer what a plan would refuse.
        """
        return bool(armmod.reachable(indices, np.asarray(pose, dtype=float)))

    selector = Selector(
        control.selection,
        admits,
        pickable_in_belt(plan.belt.length, plan.belt.width),
    )
    story = StoryLog()
    watch = AnomalyWatch(story)
    task = TaskMachine(
        control.task,
        control.calibration,
        belt_surface=surface,
        belt_speed=plan.belt.speed,
        motion=control.motion,
        admits=admits,
        chutes=plan.chutes,
        belt_width=plan.belt.width,
        effector=effector,
        story=story,
    )
    goal = None
    refusal: str | None = None
    # How near the flange ever got to what a phase asked for. Without
    # interception the arm trails a marker the belt is carrying, and this is
    # the figure that sizes the interception rather than a guess at it.
    reorders: dict[str, int] = {"appeared": 0, "retired": 0, "anchor": 0}
    head_churn = 0
    previous_head: int | None = None
    closest = float("inf")
    clearance = float("inf")
    contacts = 0
    tilt = 0.0
    # The jaw's own figures, over every tick rather than every capture: a
    # contact with the belt lasts milliseconds and the capture cadence is 0.5 s.
    # How far each visit raised the object it went for, against where that
    # object was resting when the plan was made. A visit can be perfect and
    # this still read zero, which is the whole point of measuring it apart.
    lifts: list[float] = []
    # The arm's own vertical motion, integrated at the physics rate rather than
    # sampled at the capture rate: a grasp that slips unloads the arm mid-lift
    # and that is an acceleration lasting ticks, where every pose the plan asks
    # for is smooth. Peaks are per visit, so a run reports a figure per visit.
    lurches: list[float] = []
    climbs: list[float] = []
    lurch, climb = 0.0, 0.0
    last_flange_z = _flange(indices, data)[2]
    last_rise = 0.0
    # And how the tool was turned against the object it closed on, which only a
    # ground-truth run can measure.
    yaw_errors: list[float] = []
    counted = 0
    # And how near the jaw came to any object at all when it shut, which is
    # the figure that separates a visit tick that missed from a grasp that let go.
    gaps: list[float] = []
    resting: dict[str, float] = {}
    # A place is counted once per object, the first capture its centre is
    # seen below a mouth. Counting every capture after that would report a
    # figure that grows while the object sits there.
    placed: dict[str, int] = {}
    seen_down: dict[str, str] = {}
    misrouted = 0
    opening = [
        float(v)
        for v in config.require(config.require(raw, "chutes"), "mouth_meters", "chutes")
    ]
    mouth = (opening[0], opening[1])
    # Which channel each pool slot's object belongs in, learned as objects
    # spawn rather than read once from an empty belt. Ground truth, and used
    # only to count a misroute: nothing the controller reads comes from here.
    routes: dict[int, str] = {}
    # And how near it got to where the head actually was at that instant.
    # Without interception the commanded pose is as old as the decision
    # interval, so the gap between these two figures is the staleness the
    # belt imposes and is what an interception has to close.
    closest_live = float("inf")
    # Motion integrates its own output, so the reference is seeded once and
    # fed back afterwards. Seeding it from the flange every tick would make it
    # chase the arm instead of leading it.
    motion = Reference(position=_flange(indices, data), speed=0.0)
    # And which way it is going, which a plan needs so its first arc begins
    # where the motion already is instead of asking for a step in velocity.
    moving: NDArray[np.float64] = zeros()
    # The furthest any joint may be commanded to move in one tick. Near a
    # wrist singularity the damped solve still asks for a large joint motion
    # to buy a small Cartesian one, and this is what keeps the command
    # inside what the actuators turn at.
    joint_step = control.servo.max_joint_speed * plan.timestep

    standing: tuple[Any, ...] = ()
    captures = written = drawn = geoms = 0
    believed = out / "records.txt"
    believed.write_text("")
    movie = config.require(debug, "video")
    movie_interval = float(config.require(movie, "interval_seconds", "video"))
    playback = fps or int(config.require(movie, "frames_per_second"))
    recorder = _recorder(view, out, playback, movie_interval) if video else None
    # The detection-camera file is a second render. It stays unbuilt unless
    # asked for, which is what keeps a headless run from paying for it
    # (`AC-CAM-01`).
    camera_recorder = None
    camera_rgb = None
    camera_seg = None
    if camera_video:
        camera_recorder = _camera_recorder(out, width, height, playback)
        if camera_recorder is not None:
            camera_rgb = mujoco.Renderer(model, height=height, width=width)
            camera_seg = mujoco.Renderer(model, height=height, width=width)
            camera_seg.enable_segmentation_rendering()
    next_capture = 0.0
    # Ground truth is the object's own pose, so the arm can read it far faster
    # than the cameras settle. A descent lasts 0.4 s and the capture is 0.5 s,
    # which means a visit steered only at the capture closes on a prediction
    # made before the descent began. Fifty milliseconds keeps that prediction
    # inside a few millimetres at the lateral speeds measured on this belt.
    next_ground_truth = 0.0
    ground_truth_steer = 0.05
    next_frame = 0.0
    telemetry_phase = Phase.STANDBY
    # The acceleration watch is a second difference. The first sample has no
    # previous rise, and comparing it with a seed of zero is not a spike.
    accel_ready = False

    def _record_place(name: str, channel: str) -> None:
        """Count one object crossing a mouth, and say so in the narrative.

        Both the physics tick and the capture look for the crossing. The first
        one to see a body records it; the second finds it already seen. The
        story is emitted here, on that first sighting, because a body can fall
        through a mouth between captures.

        Args:
            name: The body name.
            channel: The chute it crossed.
        """
        nonlocal misrouted
        if name in seen_down:
            return
        seen_down[name] = channel
        placed[channel] = placed.get(channel, 0) + 1
        slot_text = name.rsplit("_", 1)[-1]
        slot = int(slot_text) if slot_text.isdecimal() else None
        belongs = None if slot is None else routes.get(slot)
        item = next(
            (active for active in conveyor.active if active.index == slot),
            None,
        )
        material = "" if item is None else item.material_class
        serial = item.serial if item is not None and use_ground_truth else None
        if belongs is not None and belongs != channel:
            misrouted += 1
            LOGGER.warning(
                "misroute detected: object %s (channel %s) placed in chute %s",
                name,
                belongs,
                channel,
            )
        story.placed(
            data.time,
            name,
            material,
            channel,
            belongs,
            serial=serial,
        )

    def _interruptible(iterable: Any) -> Any:
        try:
            yield from iterable
        except KeyboardInterrupt:
            LOGGER.warning("simulation interrupted by user; finalizing")

    try:
        steps = int(seconds / plan.timestep)
        for _ in _interruptible(_progress(steps, progress)):
            mujoco.mj_step(model, data)
            conveyor.step(model, data)
            # Checked every tick, not every capture. An object released
            # over a mouth falls the belt's height in 0.43 s and the pool
            # recycles it below 0.30 m, so at the half second capture
            # cadence it goes from above the surface to gone without ever
            # being seen crossing.
            routes.update({i.index: i.channel for i in conveyor.active})
            if use_ground_truth:
                for item in conveyor.active:
                    story.note(item.serial, item.material_class, item.channel)
            for name, channel in _placed(
                mujoco, model, data, conveyor, plan.chutes, mouth, surface
            ).items():
                _record_place(name, channel)
            # The feed loop runs at the physics rate and reads the simulator's
            # own arrivals, which is ground truth and is why it is named as
            # such: the tracker's estimate is not usable this far down the
            # belt, so a loop built on it would regulate an opinion.
            while len(conveyor.arrivals) > counted:
                feeding.arrived(conveyor.arrivals[counted])
                counted += 1
            conveyor.speed = feeding.update(data.time, plan.timestep)

            # The deciders run at the capture cadence and the servo runs at
            # the physics rate, because a goal half a second old is still the
            # right goal while a joint command half a second old is a lurch.
            # A planned visit is the exception in one direction only: the
            # plan was decided once, and it is sampled here every tick.
            place = _flange(indices, data)
            was_active = task.active
            active_track = None if task.plan is None else task.plan.track_id
            flown = task.tick(data.time, place)
            rise = (place[2] - last_flange_z) / plan.timestep
            accel_sample = abs(rise - last_rise) / plan.timestep
            if flown is not None or was_active:
                lurch = max(lurch, accel_sample)
                climb = max(climb, rise)
            vertical_acceleration = 0.0 if not accel_ready else accel_sample
            accel_ready = True
            last_flange_z, last_rise = place[2], rise
            command = None
            if flown is not None:
                telemetry_phase = flown.phase
                command = Command(
                    position=flown.position,
                    yaw=flown.yaw,
                    speed=norm(flown.velocity),
                    velocity=flown.velocity,
                    aim=flown.position,
                )
                armmod.hold(data, indices, flown.grip)
                if flown.phase is Phase.HOLD and len(gaps) < len(lifts) + 1:
                    # The first tick of the hold is the instant the jaw
                    # reaches the object, which is the one a pick is decided
                    # on. Later ticks are the carry.
                    nearest = _in_the_jaw(
                        mujoco, model, data, conveyor, _pinch(indices, data)
                    )
                    gaps.append(float("inf") if nearest is None else nearest[1])
                    story.grasp_reading(
                        data.time,
                        None if task.plan is None else task.plan.track_id,
                        gaps[-1],
                    )
                    # And how the jaw is turned against the object it closed
                    # on, which is ground truth and only available from it: a
                    # tracker estimate has no body to measure against. Folded
                    # into the 90 degrees a jaw is symmetric about, because
                    # closing along an object's other axis is the same grasp.
                    if nearest is not None:
                        _log_closure(
                            mujoco,
                            model,
                            data,
                            indices,
                            nearest[0],
                            command,
                            place,
                        )
                    if use_ground_truth and command.yaw is not None and nearest:
                        body = _body_yaw(mujoco, model, data, nearest[0])
                        if body is not None:
                            turned = math.degrees(command.yaw) - body
                            yaw_errors.append(abs((turned + 45.0) % 90.0 - 45.0))
                # Motion resumes from where the plan left the reference, so
                # the park move after a visit does not start by jumping back
                # to wherever the reference had been before the plan.
                motion = Reference(position=command.position, speed=command.speed)
            elif was_active:
                telemetry_phase = Phase.STANDBY
                # The plan ran out on this tick and the visit is recorded.
                # Open the jaw, stand still, and let the next capture decide.
                armmod.hold(data, indices, JAW_OPEN)
                nearest = _in_the_jaw(
                    mujoco, model, data, conveyor, _pinch(indices, data)
                )
                lift = (
                    0.0
                    if nearest is None or nearest[0] not in resting
                    else _object_place(mujoco, model, data, nearest[0])[2]
                    - resting[nearest[0]]
                )
                lifts.append(lift)
                story.visit_ended(
                    data.time,
                    active_track,
                    lift,
                    held=lift >= GRASPED_METERS,
                )
                lurches.append(lurch)
                climbs.append(climb)
                lurch, climb = 0.0, 0.0
                resting = {}
                goal = None
                moving = zeros()
                motion = Reference(position=keep_inside(place), speed=0.0)
            elif goal is not None:
                command = toward(
                    motion,
                    goal,
                    plan.timestep,
                    control.motion,
                    feeding.speed,
                    int(data.time * NANOS_PER_SECOND),
                    keep_inside,
                )
                motion = Reference(position=command.position, speed=command.speed)

            tick_refusal: str | None = None
            if command is not None:
                moving = command.velocity
                # Lead cancels lag on a moving approach. Through DESCEND/HOLD the
                # command should already match the object; extrapolating a large
                # lateral quintic residual (seed-0 peak |vy| ≈ 0.94 m/s) only
                # throws the jaw past the grasp.
                lead = (
                    0.0
                    if flown is not None and flown.phase in (Phase.DESCEND, Phase.HOLD)
                    else control.servo.lead_seconds
                )
                stepped = follow(
                    model,
                    data,
                    indices,
                    command,
                    control.servo.gain,
                    max_joint_step=joint_step,
                    lead_seconds=lead,
                    keep_inside=keep_inside,
                )
                if stepped.refusal is not None:
                    refusal = stepped.refusal
                    tick_refusal = stepped.refusal
                elif flown is not None or (
                    goal is not None and goal.phase in _VISITING
                ):
                    closest = min(closest, distance(place, command.position))

            # Written here rather than before the decisions, so a row's phase and
            # commanded pose are the ones that tick flew on, and the jaw's own
            # clearance with them: the figure that says whether the pads were
            # anywhere near the belt, which no pose in this file can answer
            # because a pose is the flange and the jaw hangs below it.
            jaw = _jaw_state(
                mujoco, model, data, indices, jaw_geoms, belt_geom, surface
            )
            clearance = min(clearance, jaw.clearance)
            contacts += int(jaw.contact)
            tilt = max(tilt, jaw.tilt_degrees)
            if task.plan is not None:
                subject = task.plan.track_id
            elif active_track is not None:
                subject = active_track
            elif goal is None:
                subject = None
            else:
                subject = goal.track_id
            tracking_error = (
                None if command is None else distance(place, command.position)
            )
            watch.observe(
                data.time,
                contact=jaw.contact,
                clearance=jaw.clearance,
                vertical_acceleration=vertical_acceleration,
                tracking_error=tracking_error,
                refusal=tick_refusal,
                track_id=subject,
            )
            if telemetry is not None:
                telemetry.write(
                    model,
                    data,
                    indices,
                    conveyor,
                    feeding.speed,
                    telemetry_phase,
                    task.active,
                    jaw,
                    command,
                )

            # The video and live window render on their own cadence. The
            # capture cadence is what the tracker decides at, and watching
            # a decision rate is watching an arm teleport. The detection
            # camera shares that cadence (`AC-CAM-01`).
            due = data.time >= next_frame
            line_view = (recorder is not None or opened) and watching is not None
            if due and (line_view or camera_recorder is not None):
                next_frame = data.time + movie_interval
            if due and camera_recorder is not None:
                _write_camera_frame(
                    camera_rgb,
                    camera_seg,
                    model,
                    data,
                    detection_camera,
                    camera_recorder,
                    render_mode=render_mode,
                )
            if due and line_view:
                frame = _painted(
                    watching,
                    data,
                    eye,
                    standing,
                    control,
                    surface,
                    task,
                    trajectory_seconds,
                    indices,
                    render_mode=render_mode,
                )
                if recorder is not None:
                    recorder.write(frame)
                if opened and tracker_cam is not None:
                    tracker_cam.update_scene(data, camera=detection_camera)
                    _without_shadows(tracker_cam)
                    gate_img = tracker_cam.render()

                    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    gate_bgr = cv2.cvtColor(gate_img, cv2.COLOR_RGB2BGR)

                    gh, gw = gate_bgr.shape[:2]
                    y1 = 15
                    y2 = y1 + gh
                    x2 = display.shape[1] - 15
                    x1 = x2 - gw

                    display[y1:y2, x1:x2] = gate_bgr
                    cv2.rectangle(
                        display,
                        (x1 - 1, y1 - 1),
                        (x2 + 1, y2 + 1),
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        "TRACKER (GATE CAM)",
                        (x1 + 6, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    phase_name = "standby" if goal is None else goal.phase.value
                    mode_tag = (
                        "DEMO" if render_mode in ("demo", "realistic") else "LITE"
                    )
                    time_str = f"{data.time:5.2f}s"
                    spd_str = f"{feeding.speed:4.2f} m/s"
                    hud = (
                        f"CLAVE SIM [{mode_tag}] | {time_str} | "
                        f"BELT: {spd_str} | PHASE: {phase_name.upper()}"
                    )
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(hud, font_face, 0.55, 2)
                    cv2.rectangle(display, (15, 10), (25 + tw, 65), (20, 20, 20), -1)
                    cv2.putText(
                        display,
                        hud,
                        (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        f"TRACKS: {len(standing)} | GRASPS: {len(lifts)}",
                        (20, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )

                    if opened:
                        cv2.imshow("clave tracker debug", display)
                        target_wall = start_wall + data.time
                        now_wall = time.perf_counter()
                        sleep_sec = target_wall - now_wall

                        if sleep_sec > 0.001:
                            delay_ms = int(sleep_sec * 1000)
                            key = cv2.waitKey(max(1, delay_ms)) & 0xFF
                        else:
                            key = cv2.waitKey(1) & 0xFF

                        if key in (27, ord("q")):
                            break

            if use_ground_truth and task.active and data.time >= next_ground_truth:
                next_ground_truth = data.time + ground_truth_steer
                now_gt = int(data.time * NANOS_PER_SECOND)
                standing = ground_truth_markers(
                    model,
                    data,
                    conveyor.active,
                    plan,
                    effector,
                    surface,
                    now_gt,
                    tracker.window_exit,
                    feeding.speed,
                )
                flange = _flange(indices, data)
                queue = selector.update(standing, flange, feeding.speed, now_gt)
                goal = task.step(queue, flange, data.time, refusal, moving)
                refusal = None

            if data.time < next_capture:
                continue
            next_capture = data.time + capture_interval
            captures += 1
            now = int(data.time * NANOS_PER_SECOND)
            # Everything that predicts along the belt reads the speed the belt
            # is running at, not the one the layout drew. The feed controller
            # moves it, and over a two second interception a one percent error
            # is millimetres against a jaw with eight of clearance.
            tracker.belt_speed = feeding.speed
            task.belt_speed = feeding.speed

            for eye_on in detecting:
                masks.update_scene(data, camera=eye_on.source_id)
                found = segment_masks(model, masks.render())
                if not found:
                    continue
                for reading in detections_from_masks(
                    found,
                    source_id=eye_on.source_id,
                    observed_at_nanos=now,
                    camera=eye_on.position,
                    optics=eye_on.optics,
                    surface_height=surface + 0.05,
                    render=render,
                    belt_surface=surface,
                ):
                    tracker.observe(reading, at_nanos=now)
            for item in conveyor.active:
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.name)
                address = model.jnt_qposadr[model.body_jntadr[body]]
                position = tuple(float(v) for v in data.qpos[address : address + 3])
                tracker.observe(
                    Evidence(
                        source_id="simulator",
                        role=Role.GROUND_TRUTH,
                        observed_at_nanos=now,
                        confidence=1.0,
                        payload=GroundTruth(
                            object_id=item.index,
                            material_class=item.material_class,
                            position=np.asarray(
                                (position[0], position[1], position[2]),
                                dtype=np.float64,
                            ),
                        ),
                    ),
                    at_nanos=now,
                )

            records = tracker.settle(at_nanos=now)
            if use_ground_truth:
                standing = ground_truth_markers(
                    model,
                    data,
                    conveyor.active,
                    plan,
                    effector,
                    surface,
                    now,
                    tracker.window_exit,
                    feeding.speed,
                )
            else:
                standing = markers_for(records, effector, surface)

            flange = _flange(indices, data)
            if not use_ground_truth:
                for record in records:
                    try:
                        chute = channel_of(record.material)
                    except KeyError:
                        chute = ""
                    story.note(record.track_id, record.material, chute)
            queue = selector.update(standing, flange, feeding.speed, now)

            routes.update({item.index: item.channel for item in conveyor.active})
            for name, channel in _placed(
                mujoco,
                model,
                data,
                conveyor,
                plan.chutes,
                (mouth[0], mouth[1]),
                surface,
            ).items():
                _record_place(name, channel)

            for trigger in queue.reasons:
                reorders[trigger] += 1
            head_id = None if queue.head is None else queue.head.track_id
            still_waiting = previous_head is not None and any(
                item.track_id == previous_head for item in queue.order
            )
            if previous_head is not None and head_id != previous_head and still_waiting:
                head_churn += 1
            if queue.recomputed or head_id != previous_head:
                story.queue(
                    data.time,
                    queue.reasons,
                    head_id,
                    previous_head,
                    still_waiting,
                )
            previous_head = head_id
            goal = task.step(queue, flange, data.time, refusal, moving)
            refusal = None
            if goal.phase in _VISITING:
                head = next(
                    (item for item in standing if item.track_id == goal.track_id),
                    None,
                )
                if head is not None:
                    closest_live = min(closest_live, distance(flange, head.flange))
            if goal.phase is Phase.FAULT:
                # The arm was told to hold, so the reference comes back to the
                # pose it is holding rather than resuming from wherever it had
                # run ahead to. It is projected on the way: the flange itself
                # can cut the corner of the hole while tracking, and reseeding
                # the reference inside it is what turned one refusal into
                # every refusal after it.
                motion = Reference(position=keep_inside(flange), speed=0.0)
                moving = zeros()
            if task.active and not resting:
                # Snapshot how high everything is lying before the arm
                # touches anything, so a lift is measured against where the
                # object actually was.
                resting = _resting(mujoco, model, data, conveyor)

            # The markers go in before the render and live only until the next
            # update_scene, so they reach this view and no other. The gate the
            # tracker reads was rendered above and carries none of them.
            if watching is None or not write_frames:
                drawn = len(standing)
                geoms = 0
            else:
                geoms = _markers_on(watching, data, eye, standing, control, surface)
                geoms += _trajectory_on(
                    watching.scene,
                    task,
                    data.time,
                    trajectory_seconds,
                    _flange(indices, data),
                )
                drawn = len(standing)
                canvas = watching.render()

                cv2.imwrite(
                    str(out / f"frame_{captures:04d}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                )
                written += 1
            _write_beliefs(believed, captures, now, records)
    finally:
        if watching is not None:
            watching.close()
        masks.close()
        if tracker_cam is not None:
            tracker_cam.close()
        if recorder is not None:
            recorder.close()
        if camera_rgb is not None:
            camera_rgb.close()
        if camera_seg is not None:
            camera_seg.close()
        if camera_recorder is not None:
            camera_recorder.close()
        if telemetry is not None:
            telemetry.close()
        if opened:
            cv2.destroyAllWindows()

    report = DebugRunReport(
        captures=captures,
        tracks=len(tracker.settle(at_nanos=int(data.time * NANOS_PER_SECOND))),
        reorders=reorders,
        head_churn=head_churn,
        served=task.served,
        missed=task.missed,
        arrivals=task.arrivals,
        lifts=tuple(lifts),
        jaw_gaps=tuple(gaps),
        abandoned=task.abandoned,
        worst_aim_drift=task.worst_aim_drift,
        grasp_yaw_errors=tuple(yaw_errors),
        lurches=tuple(lurches),
        climbs=tuple(climbs),
        placed=dict(placed),
        misrouted=misrouted,
        feed_rate=feeding.settings.rate,
        measured_rate=feeding.measured,
        belt_speed=feeding.speed,
        profile=control.task.profile.value,
        closest_approach=None if closest == float("inf") else closest,
        closest_live=None if closest_live == float("inf") else closest_live,
        faults=task.faults,
        phase="none" if goal is None else goal.phase.value,
        drawn=drawn,
        geoms=geoms,
        frames_written=written,
        output=out,
        video_path=None if recorder is None else recorder.settings.path,
        camera_video_path=(
            None if camera_recorder is None else camera_recorder.settings.path
        ),
        telemetry_path=telemetry_path,
        windowed=opened,
        reason=reason,
        ground_truth=use_ground_truth,
        min_jaw_clearance=None if clearance == float("inf") else clearance,
        belt_contacts=contacts,
        worst_tool_tilt_degrees=tilt,
        metadata_path=metadata_path,
        render_mode=render_mode,
    )
    written_report = out / "report.json"
    written_report.write_text(
        json.dumps(_report_document(report), indent=2, sort_keys=True) + "\n"
    )
    (out / "report.txt").write_text("\n".join(report_lines(report)) + "\n")
    LOGGER.info("run report written: %s and %s", written_report, out / "report.txt")
    return replace(report, report_path=written_report)


def _open_window(cv2: Any, wanted: bool) -> tuple[bool, str | None]:
    """Open a live window, or say why there is none.

    A machine with no display is the ordinary case here, so this reports rather
    than raises: the artifact is the point and the window is a convenience.
    """
    if not wanted:
        return False, "not asked for"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, "no display is attached"
    try:
        cv2.namedWindow("clave tracker debug", cv2.WINDOW_NORMAL)
    except Exception as error:  # noqa: BLE001 - any GUI failure is the same answer
        return False, f"this OpenCV build cannot open a window ({error})"
    return True, None


def _park_the_arm(
    mujoco: Any, model: Any, data: Any, indices: Any, park: NDArray[np.float64]
) -> None:
    """Put the arm at its park pose before the run starts.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state, modified in place.
        indices: The arm indices.
        park: Where the arm rests.

    Raises:
        DebugRunError: If the park pose does not solve. That is a
            configuration error rather than a run-time one, and finding it
            here beats a run that faults on every tick.
    """

    try:
        angles = armmod.solve(model, data, indices, np.array(park, dtype=float))
    except armmod.ReachError as error:
        raise DebugRunError(f"the park pose does not solve: {error}") from error
    for slot, joint in enumerate(indices.joint_ids):
        data.qpos[model.jnt_qposadr[joint]] = angles[slot]
    for slot, actuator in enumerate(indices.actuator_ids):
        data.ctrl[actuator] = angles[slot]
    mujoco.mj_forward(model, data)


def _object_position_world(
    mujoco: Any, model: Any, data: Any, name: str
) -> NDArray[np.float64]:
    """Return where one conveyor object stands, as three meters in world frame.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        name: The body's name.

    Returns:
        The center of mass, not the body origin. These meshes carry their
        origin wherever the scanner left it, which for most of the set is
        the base, so an origin height answers a different question from the
        one a lift is asking.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    place = data.xipos[body]
    return np.asarray(
        (float(place[0]), float(place[1]), float(place[2])), dtype=np.float64
    )


_object_place = _object_position_world


def _placed(
    mujoco: Any,
    model: Any,
    data: Any,
    conveyor: Any,
    chutes: dict[str, NDArray[np.float64]],
    mouth: tuple[float, float],
    surface: float,
) -> dict[str, str]:
    """Return which objects have gone down a chute, and which one.

    A place is a plane crossing, which is what
    `docs/requirements/sorting-outputs.md` settled on: an object whose
    centre is below the belt surface and inside a mouth's footprint has
    left the line through that opening. Nothing models the chute's
    interior, because nothing below the opening is in scope.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        conveyor: The belt, for the objects still on it.
        chutes: Where each channel's mouth stands.
        mouth: Mouth size, along travel and across it.
        surface: Height of the belt surface.

    Returns:
        Body name to the channel it went down, for those that have.
    """
    gone: dict[str, str] = {}
    for item in conveyor.active:
        x, y, z = _object_place(mujoco, model, data, item.name)
        if z >= surface:
            continue
        for channel, (cx, cy, _) in chutes.items():
            if abs(x - cx) <= mouth[0] / 2.0 and abs(y - cy) <= mouth[1] / 2.0:
                gone[item.name] = channel
                break
    return gone


def _resting(mujoco: Any, model: Any, data: Any, conveyor: Any) -> dict[str, float]:
    """Return how high every object on the belt is lying right now.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        conveyor: The belt.

    Returns:
        Each body's name against the height of its center of mass. Taken
        before a plan begins, so a lift is measured against where the object
        actually was rather than against the belt: these meshes rest with
        their centers anywhere from twenty to seventy millimetres up.
    """
    return {
        item.name: _object_place(mujoco, model, data, item.name)[2]
        for item in conveyor.active
    }


def _in_the_jaw(
    mujoco: Any, model: Any, data: Any, conveyor: Any, pinch: NDArray[np.float64]
) -> tuple[str, float] | None:
    """Return the object the jaw is closed on, and how far off center it sits.

    Found by proximity to the pinch site rather than by identity. Which
    object the arm believed it was going for is the tracker's claim, and a
    measurement that trusted it would report the claim rather than the
    grasp; what is in the jaw is in the jaw.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        conveyor: The belt.
        pinch: Where the jaw closes.

    Returns:
        The body's name and its distance from the pinch site, or None when
        the belt is empty.
    """
    places = [_object_place(mujoco, model, data, item.name) for item in conveyor.active]
    if not places:
        return None
    gaps = np.linalg.norm(
        np.asarray(places, dtype=np.float64) - np.asarray(pinch), axis=1
    )
    at = int(np.argmin(gaps))
    return (conveyor.active[at].name, float(gaps[at]))


def _fold_jaw_degrees(delta_degrees: float) -> float:
    """Return how far two headings are apart, folded into a jaw's symmetry.

    A jaw is symmetric about 90 degrees, so 40 degrees and 50 degrees the
    other way are the same miss. The fold is `(delta + 45) mod 90 - 45`.

    Args:
        delta_degrees: One heading minus the other, in degrees.

    Returns:
        The absolute miss, in degrees, from 0 to 45.
    """
    return abs((delta_degrees + 45.0) % 90.0 - 45.0)


def _short_axis_degrees(mujoco: Any, model: Any, data: Any, name: str) -> float | None:
    """Return the yaw of an object's short axis in the belt plane, in degrees.

    The jaw closes horizontally. The direction it has to match is the short
    direction of the object after it has been rotated into the world, projected
    onto the belt, not the body's Euler yaw. Those two agree while the object
    lies flat on the axes it was scanned in, and they come apart as soon as it
    tips.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        name: The body's name.

    Returns:
        The yaw in degrees, or None when the shape has no preferred axis.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        return None
    geom = int(model.body_geomadr[body])
    if geom < 0:
        return None
    rotation = data.geom_xmat[geom].reshape(3, 3)
    origin = np.asarray(data.geom_xpos[geom], dtype=np.float64)
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        size = np.asarray(model.geom_size[geom], dtype=np.float64)
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        local = signs * size
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        if mesh < 0:
            return None
        local = _mesh_vertices(model, mesh)
    else:
        return None
    flat = (local @ rotation.T + origin)[:, :2]
    centered = flat - flat.mean(axis=0)
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    minor = vectors[:, 0]
    return math.degrees(math.atan2(float(minor[1]), float(minor[0])))


def _free_velocity(
    mujoco: Any, model: Any, data: Any, name: str
) -> tuple[NDArray[np.float64], float] | None:
    """Return a free body's linear velocity and its spin about the belt normal.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        name: The body's name.

    Returns:
        The linear velocity in meters per second and the spin in radians per
        second, or None when the body has no free joint.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        return None
    joint = int(model.body_jntadr[body])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        return None
    address = int(model.jnt_dofadr[joint])
    linear = np.asarray(
        (
            float(data.qvel[address]),
            float(data.qvel[address + 1]),
            float(data.qvel[address + 2]),
        ),
        dtype=np.float64,
    )
    geom = int(model.body_geomadr[body])
    rotation = data.geom_xmat[geom].reshape(3, 3)
    spin = data.qvel[address + 3 : address + 6]
    about_up = float(rotation[2] @ np.asarray(spin, dtype=np.float64))
    return linear, about_up


def _log_closure(
    mujoco: Any,
    model: Any,
    data: Any,
    indices: Any,
    name: str,
    command: Command,
    flange: NDArray[np.float64],
) -> None:
    """Log where a closing jaw sat relative to the object and to its command.

    Three distances, because a grasp fails for different reasons that one
    number confuses. The flange against the command is the inverse kinematics.
    The command against the object, in the belt plane, is the plan. The pinch
    against the object is what the jaw actually did, which is both of those
    plus the tool not being vertical. The yaw is split the same way: the tool
    against the command, and the command against the object's own short axis.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        indices: The arm indices.
        name: The nearest object's body name.
        command: The pose the plan asked for on this tick.
        flange: Where the flange actually is.
    """
    pinch = _pinch(indices, data)
    center = _object_place(mujoco, model, data, name)
    origin = _body_origin(mujoco, model, data, name)
    centroid = _geometric_center(mujoco, model, data, name)

    def _offset(
        left: NDArray[np.float64], right: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.asarray(
            (
                (float(left[0]) - float(right[0])) * 1000.0,
                (float(left[1]) - float(right[1])) * 1000.0,
                (float(left[2]) - float(right[2])) * 1000.0,
            ),
            dtype=np.float64,
        )

    gap = _offset(pinch, center)
    origin_gap = None if origin is None else _offset(origin, center)
    shape_gap = None if centroid is None else _offset(centroid, center)
    pinch_shape = None if centroid is None else _offset(pinch, centroid)
    aim = tuple((command.position[axis] - center[axis]) * 1000.0 for axis in range(2))
    aim_origin = (
        None
        if origin is None
        else tuple(
            (command.position[axis] - origin[axis]) * 1000.0 for axis in range(2)
        )
    )
    tracked = distance(flange, command.position) * 1000.0
    tool = math.degrees(armmod.tool_yaw(data, indices))
    ordered = None if command.yaw is None else math.degrees(command.yaw)
    body = _body_yaw(mujoco, model, data, name)
    short = _short_axis_degrees(mujoco, model, data, name)
    motion = _free_velocity(mujoco, model, data, name)

    def _yaw(left: float | None, right: float | None) -> str:
        if left is None or right is None:
            return "n/a"
        return f"{_fold_jaw_degrees(left - right):.1f}"

    if motion is None:
        velocity, spin = "n/a", "n/a"
    else:
        linear, rate = motion
        velocity = f"({linear[0]:+.3f}, {linear[1]:+.3f}, {linear[2]:+.3f})"
        spin = f"{rate:+.2f}"

    def _millimetres(offset: NDArray[np.float64] | None) -> str:
        if offset is None:
            return "n/a"
        return f"({offset[0]:+.0f}, {offset[1]:+.0f}, {offset[2]:+.0f})"

    LOGGER.info(
        "closure at %.3f s on %s: pinch-com dx=%+.0f dy=%+.0f dz=%+.0f mm; "
        "pinch-shape %s mm; origin-com %s mm; shape-com %s mm; "
        "command-com dx=%+.0f dy=%+.0f mm; command-origin %s mm; "
        "flange-command %.1f mm; "
        "tool-command yaw %s deg; command-body yaw %s deg; "
        "command-short-axis yaw %s deg; object v=%s m/s spin=%s rad/s",
        data.time,
        name,
        gap[0],
        gap[1],
        gap[2],
        _millimetres(pinch_shape),
        _millimetres(origin_gap),
        _millimetres(shape_gap),
        aim[0],
        aim[1],
        (
            "n/a"
            if aim_origin is None
            else f"({aim_origin[0]:+.0f}, {aim_origin[1]:+.0f})"
        ),
        tracked,
        _yaw(tool, ordered),
        _yaw(ordered, body),
        _yaw(ordered, short),
        velocity,
        spin,
    )


def _body_origin(
    mujoco: Any, model: Any, data: Any, name: str
) -> NDArray[np.float64] | None:
    """Return a body's frame origin, which is where a marker is drawn from.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        name: The body's name.

    Returns:
        The origin in world frame meters, or None when the body is absent.
        This is `xpos`, not the centre of mass: scanned meshes keep their
        origin wherever the scan left it.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        return None
    place = data.xpos[body]
    return np.asarray(
        (float(place[0]), float(place[1]), float(place[2])), dtype=np.float64
    )


def _geometric_center(
    mujoco: Any, model: Any, data: Any, name: str
) -> NDArray[np.float64] | None:
    """Return the centre of an object's geometry in the world, not its origin.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        name: The body's name.

    Returns:
        The mean of the geometry in world frame meters, or None when the body
        has no box or mesh to read one from.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        return None
    geom = int(model.body_geomadr[body])
    if geom < 0:
        return None
    rotation = data.geom_xmat[geom].reshape(3, 3)
    origin = np.asarray(data.geom_xpos[geom], dtype=np.float64)
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.zeros(3)
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        if mesh < 0:
            return None
        local = _mesh_vertices(model, mesh).mean(axis=0)
    else:
        local = np.zeros(3)
    world = rotation @ local + origin
    return np.asarray(
        (float(world[0]), float(world[1]), float(world[2])), dtype=np.float64
    )


def _body_yaw(mujoco: Any, model: Any, data: Any, name: str) -> float | None:
    """Return an object's own rotation about the belt normal, in degrees.

    Read from the body's quaternion rather than from anything the tracker
    published, because the question this answers is whether the jaw closed
    along the object's own short axis -- and a figure computed from the
    estimate that asked for the pose would agree with itself however the
    object was lying.

    Args:
        mujoco: The imported module.
        model: The compiled model.
        data: Its state.
        name: The body's name in the model.

    Returns:
        The yaw in degrees, or None when the body carries no free joint to
        read a rotation from.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body < 0:
        return None
    quat = data.xquat[body]
    w, x, y, z = (float(quat[i]) for i in range(4))
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
