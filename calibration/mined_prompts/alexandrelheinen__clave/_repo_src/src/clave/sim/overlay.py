"""Debug-view drawing: markers, the planned path, and the detection-camera film.

Nothing here steps the physics. The colours and the boxes are how a reader
sees a decision the loop already made.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.control.settings import Phase
from clave.control.task import TaskMachine
from clave.sim.report import DebugRunError
from clave.tracker.adapters.render import segment_masks
from clave.tracker.listing import described_fields
from clave.tracker.markers import draw, draw_park
from clave.world import arm as armmod
from clave.world import config

# Green reads on the belt without sitting on a material the line already paints.
_BOX_RGB = (0, 255, 0)


def bounds_of(runs: Sequence[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    """Return inclusive pixel bounds of run-length coverage.

    Args:
        runs: `(row, start_column, length)` triples from one object's mask.

    Returns:
        `(x_min, y_min, x_max, y_max)`.
    """
    x_min = min(start for _, start, _ in runs)
    y_min = min(row for row, _, _ in runs)
    x_max = max(start + length - 1 for _, start, length in runs)
    y_max = max(row for row, _, _ in runs)
    return x_min, y_min, x_max, y_max


def paint_boxes(
    frame: NDArray[np.uint8],
    boxes: Sequence[tuple[int, int, int, int]],
    labels: Sequence[str] | None = None,
    render_mode: str = "lite",
) -> NDArray[np.uint8]:
    """Draw inclusive box borders and ML tracking overlays on an RGB frame.

    The boxes are the pixels each object covered in a segmentation of this
    same frame (`AC-CAM-03`). Pixels outside a box stay as the camera
    rendered them.

    Args:
        frame: Height by width by three, RGB.
        boxes: Inclusive `(x_min, y_min, x_max, y_max)` bounds.
        labels: Optional label strings (material / tracking id) per box.
        render_mode: Render mode ('lite' or 'demo'/'realistic').

    Returns:
        The same frame, with borders written in.
    """
    height, width = frame.shape[:2]
    if render_mode == "lite" or not labels:
        color = np.asarray(_BOX_RGB, dtype=np.uint8)
        for x0, y0, x1, y1 in boxes:
            left = min(max(int(x0), 0), width - 1)
            right = min(max(int(x1), 0), width - 1)
            top = min(max(int(y0), 0), height - 1)
            bottom = min(max(int(y1), 0), height - 1)
            if right < left:
                left, right = right, left
            if bottom < top:
                top, bottom = bottom, top
            frame[top, left : right + 1] = color
            frame[bottom, left : right + 1] = color
            frame[top : bottom + 1, left] = color
            frame[top : bottom + 1, right] = color
        return frame

    # Demo / realistic mode: styled bounding boxes with labels
    try:
        import cv2

        for idx, (x0, y0, x1, y1) in enumerate(boxes):
            left = min(max(int(x0), 0), width - 1)
            right = min(max(int(x1), 0), width - 1)
            top = min(max(int(y0), 0), height - 1)
            bottom = min(max(int(y1), 0), height - 1)
            if right < left:
                left, right = right, left
            if bottom < top:
                top, bottom = bottom, top

            # Main bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 128), 2)

            # Corner accents
            c_len = min(8, max(3, (right - left) // 4))
            cv2.line(frame, (left, top), (left + c_len, top), (0, 255, 255), 2)
            cv2.line(frame, (left, top), (left, top + c_len), (0, 255, 255), 2)
            cv2.line(frame, (right, bottom), (right - c_len, bottom), (0, 255, 255), 2)
            cv2.line(frame, (right, bottom), (right, bottom - c_len), (0, 255, 255), 2)

            if labels and idx < len(labels):
                text = labels[idx]
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.45
                thickness = 1
                (txt_w, txt_h), baseline = cv2.getTextSize(text, font, scale, thickness)
                b_top = max(0, top - txt_h - 6)
                b_right = min(width, left + txt_w + 6)
                cv2.rectangle(
                    frame,
                    (left, b_top),
                    (b_right, b_top + txt_h + 6),
                    (0, 100, 50),
                    -1,
                )
                cv2.putText(
                    frame,
                    text,
                    (left + 3, b_top + txt_h + 2),
                    font,
                    scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
    except ImportError:
        color = np.asarray(_BOX_RGB, dtype=np.uint8)
        for x0, y0, x1, y1 in boxes:
            left = min(max(int(x0), 0), width - 1)
            right = min(max(int(x1), 0), width - 1)
            top = min(max(int(y0), 0), height - 1)
            bottom = min(max(int(y1), 0), height - 1)
            if right < left:
                left, right = right, left
            if bottom < top:
                top, bottom = bottom, top
            frame[top, left : right + 1] = color
            frame[bottom, left : right + 1] = color
            frame[top : bottom + 1, left] = color
            frame[top : bottom + 1, right] = color
    return frame


def _camera_recorder(out: Path, width: int, height: int, fps: int) -> Any:
    """Start recording the detection camera, or report that no encoder is installed.

    Args:
        out: Where the run writes.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Playback rate.

    Returns:
        The recorder, or None when `ffmpeg` is absent.
    """
    from clave.demo.video import StreamSettings, open_stream

    return open_stream(
        StreamSettings(
            path=out / "camera-debug.mp4",
            width=width,
            height=height,
            frames_per_second=fps,
        )
    )


def _write_camera_frame(
    rgb: Any,
    segmentation: Any,
    model: Any,
    data: Any,
    camera: str,
    recorder: Any,
    render_mode: str = "lite",
) -> None:
    """Paint one detection-camera frame and hand it to the encoder.

    The colour frame and the segmentation are the same camera at the same
    size, so a box sits on the pixels the object actually covered.

    Args:
        rgb: Renderer for the colour frame.
        segmentation: Renderer for geometry ids, segmentation already enabled.
        model: The compiled model, used to name geometries.
        data: The simulated state.
        camera: The detection camera's name.
        recorder: The open encoder.
        render_mode: Render mode ('lite' or 'demo'/'realistic').
    """
    rgb.update_scene(data, camera=camera)
    if render_mode in ("demo", "realistic"):
        _with_shadows(rgb)
    else:
        _without_shadows(rgb)
    frame = np.array(rgb.render(), copy=True)
    segmentation.update_scene(data, camera=camera)
    found = segment_masks(model, segmentation.render())
    boxes_and_labels = [
        (bounds_of(mask.runs), name.replace("object_", "OBJ_"))
        for name, mask in found.items()
    ]
    boxes = [item[0] for item in boxes_and_labels]
    labels = [item[1] for item in boxes_and_labels]
    paint_boxes(frame, boxes, labels=labels, render_mode=render_mode)
    recorder.write(frame)


def _recorder(view: dict[str, Any], out: Path, fps: int, interval: float) -> Any:
    """Start recording the view, or report that no encoder is installed.

    The recorder reads the same camera block the still frames render from, so
    the two cannot show the world from two different angles.

    Args:
        view: The `view` block of the debug configuration.
        out: Where the run writes.
        fps: Playback rate.
        interval: Simulated seconds between captures.

    Returns:
        The recorder, or None when `ffmpeg` is absent.
    """
    from clave.demo.video import VideoSettings, open_recorder

    return open_recorder(
        VideoSettings(
            path=out / "tracker-debug.mp4",
            width=int(config.require(view, "width", "view")),
            height=int(config.require(view, "height", "view")),
            frames_per_second=fps,
            interval_seconds=interval,
            azimuth=float(config.require(view, "azimuth_degrees", "view")),
            elevation=float(config.require(view, "elevation_degrees", "view")),
            distance=float(config.require(view, "distance_meters", "view")),
            lookat=_lookat(view),
        )
    )


def _view(raw: dict[str, Any], wanted: str | None) -> dict[str, Any]:
    """Return the view to film from, naming the alternatives when it is wrong.

    Args:
        raw: The parsed debug configuration.
        wanted: The view asked for, or None for the configured default.

    Returns:
        The view block.

    Raises:
        DebugRunError: If the view does not exist. A typo here is a run filmed
            from somewhere nobody chose, so the message lists what it could
            have been.
    """
    views = config.require(raw, "views")
    name = wanted if wanted is not None else str(config.require(raw, "default_view"))
    if name not in views:
        known = ", ".join(sorted(views))
        raise DebugRunError(f"no view named {name!r}. Known: {known}")
    return dict(views[name])


def _lookat(view: dict[str, Any]) -> NDArray[np.float64]:
    """Return what the camera points at, as three meters.

    Args:
        view: The `view` block of the debug configuration.

    Returns:
        The target.

    Raises:
        DebugRunError: If the key does not hold three numbers.
    """
    values = [float(value) for value in config.require(view, "lookat_meters", "view")]
    if len(values) != 3:
        raise DebugRunError(
            f"view.lookat_meters holds {len(values)} numbers, and a camera "
            f"target is three"
        )
    return np.asarray((values[0], values[1], values[2]), dtype=np.float64)


def _write_beliefs(path: Path, capture: int, at_nanos: int, records: Any) -> None:
    """Append what the tracker believed at one instant.

    The markers show where; this says what. Keeping the two apart is what lets
    the frame stay a render of the world with nothing written on it.

    Args:
        path: The listing file.
        capture: Which capture this is.
        at_nanos: The instant the records were settled at.
        records: The settled records.
    """
    lines = [f"# capture {capture:04d} at {at_nanos} ns, {len(records)} open"]
    for record in records:
        lines.append(f"track {record.track_id}")
        lines.extend(
            f"    {name}: {value}" for name, value in described_fields(record).items()
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n\n")


def _markers_on(
    renderer: Any,
    data: Any,
    camera: Any,
    standing: tuple[Any, ...],
    control: Any,
    surface: float,
) -> int:
    """Rebuild the scene from this camera and stand the markers in it.

    The markers go in after `update_scene` and live only until the next one,
    so they reach whoever renders next and nobody else: not physics, not the
    gate the tracker reads, and not a run that never calls this.

    Args:
        renderer: The renderer to build into.
        data: The simulation state.
        camera: The camera to build from.
        standing: The grasp markers to draw.
        control: The control settings, for the park pose and its colour.
        surface: Height of the belt surface, in meters.

    Returns:
        How many geoms were added.
    """
    renderer.update_scene(data, camera=camera)
    added = draw(renderer.scene, standing)
    return added + draw_park(
        renderer.scene,
        control.task.park_position,
        control.task.park_marker_color,
        belt_surface=surface,
    )


def _trajectory_on(
    scene: Any,
    task: TaskMachine,
    at_seconds: float,
    horizon_seconds: float,
    flange: NDArray[np.float64],
) -> int:
    """Draw the active plan's future path and discrete pose indicators."""
    import mujoco

    added = _trajectory_sphere(scene, flange, 0.018, (0.95, 0.95, 0.95, 1.0))
    plan = task.plan
    if plan is None or horizon_seconds == 0.0:
        return added

    start = max(at_seconds, plan.started_at)
    end = min(start + horizon_seconds, plan.started_at + plan.duration)
    if end <= start:
        return added
    count = max(2, int(math.ceil((end - start) * 12.0)) + 1)
    samples: list[tuple[NDArray[np.float64], tuple[float, float, float, float]]] = []
    for index in range(count):
        instant = start + (end - start) * index / (count - 1)
        sampled = plan.at(instant)
        if sampled is None:
            continue
        phase, state, _ = sampled
        samples.append((state.position, _trajectory_color(phase)))
    for before, after in zip(samples, samples[1:], strict=False):
        added += _trajectory_segment(scene, before[0], after[0], after[1])

    pick = plan.legs[1].segment.end.position if len(plan.legs) > 1 else None
    if pick is not None:
        added += _trajectory_sphere(scene, pick, 0.022, (1.0, 0.72, 0.10, 1.0))
    if plan.legs[-1].phase is Phase.DELIVER:
        added += _trajectory_sphere(
            scene,
            plan.legs[-1].segment.end.position,
            0.024,
            (0.20, 0.90, 0.35, 1.0),
        )
    del mujoco
    return added


def _trajectory_color(phase: Phase) -> tuple[float, float, float, float]:
    """Return a subtle color for one future trajectory phase."""
    return {
        Phase.TRACK: (0.20, 0.55, 0.95, 0.75),
        Phase.DESCEND: (0.95, 0.72, 0.18, 0.85),
        Phase.HOLD: (0.25, 0.85, 0.45, 0.85),
        Phase.RETREAT: (0.95, 0.48, 0.18, 0.80),
        Phase.DELIVER: (0.85, 0.35, 0.85, 0.80),
    }.get(phase, (0.70, 0.70, 0.70, 0.65))


def _trajectory_sphere(
    scene: Any,
    position: NDArray[np.float64],
    radius: float,
    color: tuple[float, float, float, float],
) -> int:
    """Add one small trajectory indicator sphere to a render scene."""
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return 0
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1
    return 1


def _trajectory_segment(
    scene: Any,
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    color: tuple[float, float, float, float],
) -> int:
    """Add a thin capsule between two future trajectory samples."""
    import mujoco

    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-9 or scene.ngeom >= scene.maxgeom:
        return 0
    axis = vector / length
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(axis[2])) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = np.cross(reference, axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    rotation = np.column_stack((first, second, axis)).ravel()
    midpoint = (np.asarray(start) + np.asarray(end)) / 2.0
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([0.004, length / 2.0, 0.0], dtype=np.float64),
        midpoint,
        rotation,
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1
    return 1


def _pinch_position_world(indices: Any, data: Any) -> NDArray[np.float64]:
    """Return where the jaw closes, as three meters in world frame.

    Args:
        indices: The arm indices.
        data: Its state, with forward kinematics already current.

    Returns:
        The pinch site's position. Every commanded pose is against the
        flange, because that is what the trusted reach was swept against,
        and this is where the object actually ends up.
    """
    place = data.site_xpos[indices.pinch_site]
    return np.asarray(
        (float(place[0]), float(place[1]), float(place[2])), dtype=np.float64
    )


_pinch = _pinch_position_world


def _flange_position_world(indices: Any, data: Any) -> NDArray[np.float64]:
    """Return where the flange stands, as three meters in world frame.

    Args:
        indices: The arm indices.
        data: Its state, with forward kinematics already current.

    Returns:
        The position.
    """
    place = armmod.end_effector_position(data, indices)
    return np.asarray(
        (float(place[0]), float(place[1]), float(place[2])), dtype=np.float64
    )


_flange = _flange_position_world


def _painted(
    renderer: Any,
    data: Any,
    camera: Any,
    standing: tuple[Any, ...],
    control: Any,
    surface: float,
    task: TaskMachine,
    trajectory_seconds: float,
    indices: Any,
    render_mode: str = "lite",
) -> Any:
    """Render one video frame with the markers standing in it.

    Args:
        renderer: The renderer to use.
        data: The simulation state.
        camera: The camera to film from.
        standing: The grasp markers settled at the last capture.
        control: The control settings, for the park pose and its colour.
        surface: Height of the belt surface, in meters.
        task: Active task machine state.
        trajectory_seconds: Trajectory horizon in seconds.
        indices: Arm model indices.
        render_mode: Render mode ('lite' or 'demo'/'realistic').

    Returns:
        The rendered frame.
    """
    _markers_on(renderer, data, camera, standing, control, surface)
    _trajectory_on(
        renderer.scene,
        task,
        float(data.time),
        trajectory_seconds,
        _flange(indices, data),
    )
    if render_mode in ("demo", "realistic"):
        _with_shadows(renderer)
    else:
        _without_shadows(renderer)
    return renderer.render()


def _without_shadows(renderer: Any) -> None:
    """Turn the shadow map off for one frame."""
    import mujoco

    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0


def _with_shadows(renderer: Any) -> None:
    """Turn the shadow map on for one frame."""
    import mujoco

    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
