"""The loop: step the world, infer, propose, and time the round trip.

This is the first thing in CLAVE that runs end to end. It proves a mechanism
and nothing about quality: the models it runs saw 240 frames of parametric
primitives, so every decision it produces should be read as evidence that the
path works, never as evidence that the path is right.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from clave.errors import ClaveError
from clave.runtime.bridge import Bridge, BridgePaths, locate_binary
from clave.runtime.inference import Predictor
from clave.runtime.proposal import Proposal
from clave.taxonomy import BY_ID
from clave.tracker.evidence import Role
from clave.tracker.sensors import load_sensors, require_role
from clave.world import arm as armmod
from clave.world import belt, config, scene

LOGGER = logging.getLogger(__name__)

NANOS_PER_SECOND = 1_000_000_000
"""Nanoseconds in a second, for the monotonic times the boundary carries."""

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


class RuntimeConfigError(ClaveError):
    """The runtime configuration is missing a key or names something unknown."""


@dataclass(frozen=True)
class RoutingPolicy:
    """The operator's mapping from material class to physical channel.

    Channel numbers belong to the line rather than to the taxonomy, which is
    why they are configuration and not a table in code.

    Attributes:
        channels: Taxonomy identifier to channel number.
        reject_channel: Where everything unroutable goes.
        confidence_floor: Below this, an object goes to reject whatever it is.
    """

    channels: dict[str, int]
    reject_channel: int
    confidence_floor: float


@dataclass(frozen=True)
class RuntimeSettings:
    """Everything the loop reads before it starts.

    Attributes:
        perception: Registry name of the classifier.
        policy: Registry name of the pick policy.
        checkpoints: Directory the training runs wrote to.
        seconds: Simulated seconds to run.
        capture_interval_seconds: Simulated seconds between captures.
        frame_height: Frame height in pixels.
        frame_width: Frame width in pixels.
        seed: Seed controlling belt speed, placement and spawn timing.
        presence_floor: Probability below which the classifier abstains.
        association_radius_meters: How far an identity association may reach.
        belt_frame: Name of the coordinate frame pick points are expressed in,
            carried into every published message so a consumer knows what the
            numbers mean.
        routing: The operator's channel policy.
    """

    perception: str
    policy: str
    checkpoints: Path
    seconds: float
    capture_interval_seconds: float
    frame_height: int
    frame_width: int
    seed: int
    presence_floor: float
    association_radius_meters: float
    belt_frame: str
    routing: RoutingPolicy

    @classmethod
    def load(cls, path: Path) -> RuntimeSettings:
        """Read the settings from YAML.

        Args:
            path: The configuration file.

        Returns:
            The settings.

        Raises:
            RuntimeConfigError: If a key is missing or a class is unknown.
        """
        LOGGER.debug("loading runtime settings from %s", path)
        raw = yaml.safe_load(path.read_text())
        runtime = _require(raw, "runtime")
        routing = _require(raw, "routing")
        channels = _require(routing, "channels", "routing")
        for class_id in channels:
            if class_id not in BY_ID:
                raise RuntimeConfigError(
                    f"routing.channels names {class_id!r}, which is not in the taxonomy"
                )
        settings = cls(
            perception=str(_require(runtime, "perception", "runtime")),
            policy=str(_require(runtime, "policy", "runtime")),
            checkpoints=Path(str(_require(runtime, "checkpoints", "runtime"))),
            seconds=float(_require(runtime, "seconds", "runtime")),
            capture_interval_seconds=float(
                _require(runtime, "capture_interval_seconds", "runtime")
            ),
            frame_height=int(_require(runtime, "frame_height", "runtime")),
            frame_width=int(_require(runtime, "frame_width", "runtime")),
            seed=int(_require(runtime, "seed", "runtime")),
            presence_floor=float(_require(runtime, "presence_floor", "runtime")),
            association_radius_meters=float(
                _require(runtime, "association_radius_meters", "runtime")
            ),
            belt_frame=str(_require(runtime, "belt_frame", "runtime")),
            routing=RoutingPolicy(
                channels={str(key): int(value) for key, value in channels.items()},
                reject_channel=int(_require(routing, "reject_channel", "routing")),
                confidence_floor=float(
                    _require(routing, "confidence_floor", "routing")
                ),
            ),
        )
        LOGGER.debug(
            "runtime settings initialized: perception=%s policy=%s seconds=%.3f "
            "capture_interval=%.3f seed=%d frame=%dx%d",
            settings.perception,
            settings.policy,
            settings.seconds,
            settings.capture_interval_seconds,
            settings.seed,
            settings.frame_width,
            settings.frame_height,
        )
        return settings


def _require(mapping: Any, key: str, path: str = "") -> Any:
    """Read a key, failing with its location when it is absent.

    Args:
        mapping: The mapping to read from.
        key: The key required.
        path: Dotted path of the parent, used in the error message.

    Returns:
        The value.

    Raises:
        RuntimeConfigError: If the key is absent.
    """
    if not isinstance(mapping, dict) or key not in mapping:
        where = f"{path}.{key}" if path else key
        raise RuntimeConfigError(f"required configuration key {where!r} is missing")
    return mapping[key]


def runtime_config(world: dict[str, Any], routing: RoutingPolicy) -> dict[str, Any]:
    """Build what the Rust runtime reads, from the world and the policy.

    The envelope is derived from the world configuration rather than restated,
    so the geometry the safety layer checks against cannot drift from the
    geometry the simulation runs.

    Args:
        world: The loaded world configuration.
        routing: The operator's channel policy.

    Returns:
        The JSON document the runtime loads.
    """
    arm = config.require(world, "arm")
    conveyor = config.require(world, "belt")
    length = float(config.require(conveyor, "length_meters", "belt"))
    width = float(config.require(conveyor, "width_meters", "belt"))
    base = [
        float(value) for value in config.require(arm, "base_position_meters", "arm")
    ]
    band = [
        float(value) for value in config.require(arm, "tool_above_base_meters", "arm")
    ]
    return {
        # Where the arm stands, which is the top of its pedestal. The reach test
        # is radial about this point.
        "base_meters": base,
        # The annulus and the vertical band, both measured by sweeping the
        # compiled model with the tool held vertical rather than derived from
        # link lengths. A six-axis arm under an orientation constraint has no
        # closed-form workspace, so there is nothing to derive them from.
        "reach_meters": [
            float(config.require(arm, "reach_min_meters", "arm")),
            float(config.require(arm, "reach_max_meters", "arm")),
        ],
        "tool_above_base_meters": band,
        "belt_surface_z_meters": float(
            config.require(conveyor, "surface_height_meters", "belt")
        ),
        "belt_x_meters": [-length / 2.0, length / 2.0],
        "belt_y_meters": [-width / 2.0, width / 2.0],
        "channels": routing.channels,
        "reject_channel": routing.reject_channel,
        "confidence_floor": routing.confidence_floor,
    }


@dataclass(frozen=True)
class DecisionRecord:
    """One decision the loop published, and the object it was about.

    Attributes:
        object_id: The identity the world assigned at spawn.
        true_class: What the object actually is, from the world.
        predicted_class: What inference said it was.
        latency_seconds: Frame capture to published decision.
        verdict: What the safety layer did, as the runtime reported it.
        channel: The channel the routing policy resolved, when one was.
    """

    object_id: int
    true_class: str
    predicted_class: str
    latency_seconds: float
    verdict: str
    channel: int | None


@dataclass
class RunReport:
    """What one run did, counted and measured.

    Attributes:
        predictor: What proposed the picks.
        frames: Frames captured.
        proposals: Proposals sent across the boundary.
        silent_frames: Frames where the predictor proposed nothing.
        decisions_received: Published decisions that arrived back.
        published_to_ros: Decisions put on the ROS 2 topic.
        video_path: Where the recording went, when one was made.
        video_frames: Frames written to it.
        ros_unavailable_reason: Why nothing was published, when nothing was.
        counters: The runtime's own counts, which are the authority.
        latencies_seconds: Frame to published decision, one per proposal.
        decisions: One record per proposal that crossed the boundary, carrying
            what the object is as well as what inference said it was.
        presented: Every object that entered the reachable window, with its
            true class. This is the population a validation record describes,
            and an object that never entered the window is outside it.
        belt_speed: Belt speed this run drew, in meters per second.
        window_length: Length of the reachable window, in meters.
        machine: What the measurement was taken on.
        threads: Compute threads available.
        frame_height: Frame height in pixels.
        frame_width: Frame width in pixels.
    """

    predictor: str
    frames: int = 0
    proposals: int = 0
    silent_frames: int = 0
    decisions_received: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    latencies_seconds: list[float] = field(default_factory=list)
    decisions: list[DecisionRecord] = field(default_factory=list)
    presented: list[tuple[int, str]] = field(default_factory=list)
    belt_speed: float = 0.0
    window_length: float = 0.0
    machine: str = ""
    threads: int = 0
    frame_height: int = 0
    frame_width: int = 0
    published_to_ros: int = 0
    ros_unavailable_reason: str | None = None
    video_path: str | None = None
    video_frames: int = 0

    @property
    def budget_seconds(self) -> float:
        """How long an object stays reachable at this run's belt speed."""
        if self.belt_speed <= 0.0:
            return 0.0
        return self.window_length / self.belt_speed

    def percentile(self, fraction: float) -> float:
        """Return a latency percentile, in seconds.

        Args:
            fraction: Between 0 and 1, so 0.99 is the 99th percentile.

        Returns:
            The latency at that percentile, or 0.0 when nothing was measured.
            The value is an observed sample rather than an interpolation, so a
            reported p99 is a latency that actually happened.
        """
        if not self.latencies_seconds:
            return 0.0
        ordered = sorted(self.latencies_seconds)
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return ordered[index]

    def as_dict(self) -> dict[str, Any]:
        """Render as plain data for the run record."""
        return {
            "predictor": self.predictor,
            "frames": self.frames,
            "proposals": self.proposals,
            "silent_frames": self.silent_frames,
            "decisions_received": self.decisions_received,
            "presented": len(self.presented),
            "counters": self.counters,
            "belt_speed_meters_per_second": self.belt_speed,
            "window_length_meters": self.window_length,
            "budget_seconds": self.budget_seconds,
            "latency_seconds": {
                "samples": len(self.latencies_seconds),
                "median": self.percentile(0.50),
                "p99": self.percentile(0.99),
                "worst": max(self.latencies_seconds, default=0.0),
            },
            "published_to_ros": self.published_to_ros,
            "video_path": self.video_path,
            "video_frames": self.video_frames,
            "ros_unavailable_reason": self.ros_unavailable_reason,
            "machine": self.machine,
            "threads": self.threads,
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "caveat": (
                "The models were trained on 240 frames of parametric "
                "primitives. Nothing here measures whether a decision is "
                "correct, and nothing here ran on hardware."
            ),
        }


def run(
    root: Path,
    settings: RuntimeSettings,
    predictor: Predictor,
    directory: Path,
    publish_to_ros: bool = False,
    video: Any = None,
) -> RunReport:
    """Run the loop end to end and report what it did.

    Args:
        root: Repository root.
        settings: What to run and for how long.
        predictor: What proposes a pick per frame.
        directory: Where the sockets and the runtime configuration go.
        publish_to_ros: Put every published decision on a ROS 2 topic. A machine
            with no ROS installation runs the loop anyway and says in the report
            that nothing was published, because a missing consumer is not a
            reason to refuse to decide.
        video: Settings for recording what the simulation looked like, or None.
            The video renders on its own cadence from the same stepped world,
            because the decision cadence is far too sparse to watch.

    Returns:
        The report, with one latency sample per proposal.
    """
    import os

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
            return iterable

    from clave.candidates.bench import _machine

    raw = config.load(root / "configs" / "world" / "sorting_line.yml")
    rng = np.random.default_rng(settings.seed)
    model, data, plan = scene.build(raw, rng, root)
    spawn = config.require(raw, "spawn")
    effector = config.require(raw, "effector")
    finger_length = float(config.require(effector, "finger_length_meters", "effector"))
    conveyor = belt.Conveyor(
        plan,
        rng,
        config.require_range(spawn, "spacing_meters", "spawn"),
        config.require_range(spawn, "lateral_offset_meters", "spawn"),
        config.require_range(spawn, "drop_height_meters", "spawn"),
        entry_margin=float(config.require(spawn, "entry_margin_meters", "spawn")),
    )
    # The swept downstream edge, not half the window's length. Those agree only
    # while the arm stands at the belt centre, so the second is a latent defect
    # that moving the arm would expose.
    # Resolved by role. The line carries four cameras and only one detects.
    detection_camera = require_role(load_sensors(raw), Role.DETECTION).source_id
    exit_coordinate = belt.window_exit(conveyor.plan)
    if exit_coordinate is None:
        raise RuntimeConfigError(
            "the belt never enters the arm's reach, so this world can publish "
            "no decision"
        )
    indices = armmod.locate(
        model,
        armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
    )
    renderer = mujoco.Renderer(
        model, height=settings.frame_height, width=settings.frame_width
    )
    recorder, movie, camera = _recording(mujoco, model, video)

    machine, threads = _machine()
    report = RunReport(
        predictor=predictor.name,
        belt_speed=conveyor.report.belt_speed,
        window_length=conveyor.report.window_length,
        machine=machine,
        threads=threads,
        frame_height=settings.frame_height,
        frame_width=settings.frame_width,
    )

    publisher = _publisher(settings, report) if publish_to_ros else None

    # What each object actually is, kept as the world reveals it, so a decision
    # can be scored against the truth rather than against another prediction.
    truth: dict[int, str] = {}

    directory.mkdir(parents=True, exist_ok=True)
    paths = BridgePaths.under(directory)
    bridge = Bridge(
        locate_binary(root),
        paths,
        runtime_config(raw, settings.routing),
        on_decision=None if publisher is None else publisher.publish,
    )
    movie_renderer = (
        None
        if movie is None
        else mujoco.Renderer(model, height=movie.height, width=movie.width)
    )
    with bridge:
        next_capture = 0.0
        next_frame = 0.0

        def _interruptible(iterable: Any) -> Any:
            try:
                yield from iterable
            except KeyboardInterrupt:
                LOGGER.warning("simulation interrupted by user; finalizing")

        steps = int(settings.seconds / plan.timestep)
        progress = tqdm(range(steps), desc="Simulating", unit="step")
        for _ in _interruptible(progress):
            mujoco.mj_step(model, data)
            conveyor.step(model, data)
            labels = _labels(model, data, conveyor, exit_coordinate)
            reachable = [label for label in labels if label.in_reachable_window]
            if reachable:
                target = np.array(reachable[0].position)
                target[2] += finger_length
                armmod.step_toward(
                    model,
                    data,
                    indices,
                    target,
                    gain=ARM_GAIN,
                )
            if (
                recorder is not None
                and movie is not None
                and movie_renderer is not None
                and data.time >= next_frame
            ):
                next_frame = data.time + movie.interval_seconds
                movie_renderer.update_scene(data, camera=camera)
                recorder.write(movie_renderer.render())

            if data.time < next_capture:
                continue
            next_capture = data.time + settings.capture_interval_seconds

            renderer.update_scene(data, camera=detection_camera)
            frame = renderer.render().astype(np.uint8)
            # The clock starts once the frame exists. Rendering is what a
            # camera does on a real line, so charging it to the pipeline would
            # measure the simulator rather than CLAVE.
            began = time.perf_counter()
            report.frames += 1

            joints = tuple(
                float(angle) for angle in armmod.joint_positions(model, data, indices)
            )
            truth.update({label.object_id: label.material_class for label in labels})
            prediction = predictor.predict(frame, joints, labels, exit_coordinate)
            if prediction is None:
                report.silent_frames += 1
                continue

            now = time.monotonic_ns()
            remaining = max(
                0.0,
                (exit_coordinate - prediction.point[0]) / conveyor.report.belt_speed,
            )
            outcome = bridge.submit(
                Proposal(
                    object_id=prediction.object_id,
                    material_class=prediction.material_class,
                    confidence=min(1.0, max(0.0, prediction.confidence)),
                    point=prediction.point,
                    yaw_radians=0.0,
                    reference_time_nanos=now,
                    window_start_nanos=now,
                    window_end_nanos=now + int(remaining * NANOS_PER_SECOND),
                )
            )
            elapsed = time.perf_counter() - began
            report.latencies_seconds.append(elapsed)
            report.proposals += 1
            report.decisions.append(
                DecisionRecord(
                    object_id=prediction.object_id,
                    true_class=truth.get(prediction.object_id, ""),
                    predicted_class=prediction.material_class,
                    latency_seconds=elapsed,
                    verdict=outcome.verdict,
                    channel=outcome.channel,
                )
            )

        report.counters = bridge.counters()
        report.decisions_received = bridge.decisions_received
        report.presented = [
            (item.index, item.material_class) for item in conveyor.entered_window()
        ]
    if publisher is not None:
        report.published_to_ros = publisher.published
        publisher.close()
    if recorder is not None:
        recorder.close()
        report.video_path = str(recorder.settings.path)
        report.video_frames = recorder.frames
    return report


def _recording(mujoco: Any, model: Any, video: Any) -> tuple[Any, Any, Any]:
    """Prepare the video recorder and the camera it films from.

    Args:
        mujoco: The imported module, passed in so this stays importable without
            it.
        model: The compiled model.
        video: The settings, or None.

    Returns:
        The recorder, the settings, and the camera. All three are None when no
        video was asked for, and the recorder alone is None when the encoder is
        absent.
    """
    if video is None:
        return None, None, None
    from clave.demo.video import open_recorder

    recorder = open_recorder(video)
    if recorder is None:
        return None, None, None
    camera = mujoco.MjvCamera()
    camera.azimuth = video.azimuth
    camera.elevation = video.elevation
    camera.distance = video.distance
    camera.lookat[:] = video.lookat
    return recorder, video, camera


def _publisher(settings: RuntimeSettings, report: RunReport) -> Any:
    """Start the ROS 2 publisher, or record why there is none.

    Args:
        settings: Carries the frame the pick point is expressed in.
        report: Receives the reason when ROS is absent.

    Returns:
        The publisher, or None when ROS 2 is not installed here.
    """
    from clave.ros.publisher import DecisionPublisher, RosUnavailable

    try:
        return DecisionPublisher(frame_id=settings.belt_frame)
    except RosUnavailable as error:
        report.ros_unavailable_reason = str(error)
        return None


def _labels(
    model: Any, data: Any, conveyor: belt.Conveyor, exit_coordinate: float
) -> tuple[Any, ...]:
    """Read every active object's label straight from the world."""
    import mujoco

    from clave.data.examples import ObjectLabel

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
                in_reachable_window=belt.within_reach(position, conveyor.plan),
            )
        )
    return tuple(labels)


def write_record(path: Path, report: RunReport) -> None:
    """Write the run record a later document quotes.

    Args:
        path: Where to write it.
        report: What the run did.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
