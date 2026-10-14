"""What a run records, so its numbers can be traced and audited.

AC-MOVE-51, AC-MOVE-52 and AC-MOVE-53. These are the figures a run of
2026-09-21 could not answer: which tree flew it, and how close the jaws came to
the belt. Every number in that run's telemetry that described the state it
reached survived the replay; the ones that described the control path did not,
because nothing recorded the tree.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from clave.control.motion import Command
from clave.control.settings import Phase
from clave.sim.clearance import _jaw_collision_geoms, _jaw_state, _lowest_world_z
from clave.sim.debug_run import (
    _park_the_arm,
    _progress,
    _TelemetryWriter,
)
from clave.sim.report import _run_metadata
from clave.world.config import load

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "configs" / "world" / "sorting_line.yml"
CONTROL = ROOT / "configs" / "runtime" / "control.yml"


@pytest.fixture(scope="module")
def world() -> Any:
    """The shipped world, compiled once for the module."""
    numpy = pytest.importorskip("numpy")
    mujoco = pytest.importorskip("mujoco")
    from clave.world import arm as armmod
    from clave.world import scene

    model, data, plan = scene.build(load(WORLD), numpy.random.default_rng(0), ROOT)
    mujoco.mj_forward(model, data)
    return (
        mujoco,
        model,
        data,
        plan,
        armmod.locate(
            model,
            armmod.ReachBounds(plan.reach_min, plan.reach_max, plan.tool_above_base),
        ),
    )


def test_the_run_records_the_revision_and_the_configuration_digests() -> None:
    """AC-MOVE-52: a run records the revision and the configuration digests."""
    record = _run_metadata(ROOT, {"world": WORLD, "control": CONTROL})
    revision = record["revision"]
    assert revision is None or (len(revision) == 40 and int(revision, 16) >= 0)
    assert record["dirty"] is None or isinstance(record["dirty"], bool)
    digests = record["config_digests"]
    assert set(digests) == {"world", "control"}
    for digest in digests.values():
        assert len(digest) == 64 and int(digest, 16) >= 0
    assert digests["world"] != digests["control"], "two configs hashed as one"


def test_the_run_records_what_it_was_asked_for_ac_move_54() -> None:
    """AC-MOVE-54: a run records the command line that produced it.

    The regression this exists for: the seed, the duration and the view are
    not configuration, so a run's artifacts could not be run again even by
    whoever wrote them. Reproducing a published run at the seed recorded
    nowhere produced a different world, and nothing in the artifacts could say
    whether the seed or the world was the difference.
    """
    ask = {
        "seconds": 60.0,
        "seed": 4,
        "capture_interval_seconds": 0.5,
        "view": "belt",
        "ground_truth_tracker": True,
    }
    record = _run_metadata(ROOT, {"world": WORLD}, ask)
    assert record["parameters"] == ask
    for name in ("revision", "dirty", "config_digests", "config_paths"):
        assert name in record, f"the parameters replaced {name}"


def test_no_progress_is_a_plain_range() -> None:
    """AC-STORY-07: without a bar the step loop is a plain range."""
    assert isinstance(_progress(4, False), range)
    bar = _progress(4, True)
    try:
        if type(bar).__name__ == "tqdm":
            assert getattr(bar, "disable", False) is False
        else:
            assert isinstance(bar, range)
    finally:
        close = getattr(bar, "close", None)
        if close is not None:
            close()


def test_the_report_survives_the_terminal_it_printed_on_ac_move_55() -> None:
    """AC-MOVE-55: a run files its report beside its artifacts.

    The figures used to be printed and nothing else, so the run this work
    started from lost its own grasp distances when its terminal closed, and
    reading them back from telemetry meant inferring which of thirteen objects
    each visit had been about.
    """
    from clave.sim.report import (
        DebugRunReport,
        _report_document,
        report_lines,
    )

    report = DebugRunReport(
        captures=120,
        tracks=4,
        reorders={"appeared": 7, "retired": 3, "anchor": 2},
        head_churn=1,
        served=(1, 2, 3),
        missed=(4,),
        arrivals=(0.002,),
        lifts=(0.012,),
        jaw_gaps=(0.021,),
        abandoned=((5, "no interception from an aim 338 mm out"),),
        worst_aim_drift=0.338,
        grasp_yaw_errors=(7.3,),
        lurches=(52.9,),
        climbs=(1.21,),
        placed={"CH-HDPE": 2},
        misrouted=0,
        feed_rate=0.125,
        measured_rate=0.100,
        belt_speed=0.255,
        profile="full_visit",
        closest_approach=0.010,
        closest_live=0.020,
        faults=(),
        phase="standby",
        drawn=3,
        geoms=11,
        frames_written=120,
        output=Path("runs/debug/example"),
        video_path=None,
        telemetry_path=None,
        metadata_path=Path("runs/debug/example/metadata.json"),
        min_jaw_clearance=0.0071,
        belt_contacts=0,
        worst_tool_tilt_degrees=3.7,
        ground_truth=True,
    )
    text = "\n".join(report_lines(report))
    for figure in (
        "visits served   3 [1, 2, 3]",
        "visits given up 1",
        "338 mm out",
        "jaw to object   21 mm",
        "tool turned off 7.3 deg",
        "vertical lurch  52.9 m/s2",
        "jaw clearance   7.1 mm above the belt, 0 ticks in contact",
    ):
        assert figure in text, f"{figure!r} is not in the report"
    document = _report_document(report)
    assert document["abandoned"] == [[5, "no interception from an aim 338 mm out"]]
    assert document["served"] == [1, 2, 3]
    assert document["metadata_path"] == "runs/debug/example/metadata.json"
    assert document["worst_aim_drift"] == 0.338
    # AC-CAM-02: no camera file means the report does not name one.
    assert "camera video" not in text
    from dataclasses import replace

    named = replace(
        report, camera_video_path=Path("runs/debug/example/camera-debug.mp4")
    )
    named_text = "\n".join(report_lines(named))
    assert "camera video    runs/debug/example/camera-debug.mp4" in named_text
    assert _report_document(named)["camera_video_path"] == (
        "runs/debug/example/camera-debug.mp4"
    )


def test_a_telemetry_row_carries_the_command_and_the_jaw(
    world: Any, tmp_path: Path
) -> None:
    """AC-MOVE-53: a row carries the command and the jaw's own clearance.

    A pose is the flange, the jaw hangs 173.5 mm below it, and nothing in the
    old row said which of the two was anywhere near the belt.
    """
    mujoco, model, data, plan, arm = world
    _park_the_arm(
        mujoco,
        model,
        data,
        arm,
        np.asarray((0.72, -0.42, 1.20), dtype=np.float64),
    )
    geoms = _jaw_collision_geoms(model, arm)
    belt = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "belt"))
    surface = plan.belt.surface_height
    state = _jaw_state(mujoco, model, data, arm, geoms, belt, surface)
    path = tmp_path / "telemetry.csv"
    writer = _TelemetryWriter(path, model, plan.pool_size, 100.0)
    command = Command(
        position=np.asarray((0.50, 0.10, 1.20), dtype=np.float64),
        yaw=0.25,
        speed=0.0,
        velocity=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        aim=np.asarray((0.50, 0.10, 1.20), dtype=np.float64),
    )
    writer.write(
        model,
        data,
        arm,
        SimpleNamespace(active=()),
        0.31,
        Phase.STANDBY,
        False,
        state,
        command,
    )
    writer.close()

    with path.open() as handle:
        header = next(csv.reader(handle))
        row = next(csv.reader(handle))
    record = dict(zip(header, row, strict=True))
    for column in (
        "commanded_x",
        "commanded_y",
        "commanded_z",
        "commanded_yaw",
        "jaw_lowest_z",
        "jaw_clearance",
        "belt_contact",
        "tool_tilt_degrees",
    ):
        assert column in record, f"{column} is not in the telemetry"
    assert float(record["commanded_x"]) == pytest.approx(0.50)
    assert float(record["commanded_yaw"]) == pytest.approx(0.25)
    assert float(record["jaw_lowest_z"]) == pytest.approx(state.lowest_z)
    assert float(record["jaw_clearance"]) == pytest.approx(state.clearance)
    assert record["belt_contact"] == "0"
    # Parked at 1.20 m with the jaw 173.5 mm below the flange, the pads stand
    # clear of a belt at 0.90 m by a hand's width.
    assert state.clearance > 0.05
    assert state.tilt_degrees < 1.0


def test_the_jaw_state_reports_a_jaw_inside_the_belt(world: Any) -> None:
    """AC-MOVE-53: the clearance is measured from the jaw's own geometry.

    There is no pose in this repository that answers how close the pads came to
    the belt, because every pose is the flange and the jaw hangs below it. This
    drives the jaw into the belt on purpose and reads the answer.
    """
    numpy = pytest.importorskip("numpy")
    mujoco, model, data, plan, arm = world
    surface = plan.belt.surface_height
    geoms = _jaw_collision_geoms(model, arm)
    belt = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "belt"))

    _park_the_arm(
        mujoco,
        model,
        data,
        arm,
        np.asarray((0.72, -0.42, 1.20), dtype=np.float64),
    )
    parked = _jaw_state(mujoco, model, data, arm, geoms, belt, surface)
    assert parked.clearance > 0.0
    assert parked.contact is False

    from clave.world import arm as armmod

    angles = armmod.solve(model, data, arm, numpy.array([0.30, 0.0, surface + 0.15]))
    for slot, joint in enumerate(arm.joint_ids):
        data.qpos[model.jnt_qposadr[joint]] = angles[slot]
        data.ctrl[arm.actuator_ids[slot]] = angles[slot]
    mujoco.mj_forward(model, data)
    inside = _jaw_state(mujoco, model, data, arm, geoms, belt, surface)
    assert inside.clearance < 0.0, "the pads are not reporting the belt they are in"

    # Checked over the first ticks rather than only at the end: the contact is
    # stiff enough to push the jaw back out of the belt within a few of them,
    # and a run reports the contact rather than how long it lasted.
    touching = False
    for _ in range(200):
        mujoco.mj_step(model, data)
        if _jaw_state(mujoco, model, data, arm, geoms, belt, surface).contact:
            touching = True
            break
    assert touching, "the jaws are inside the belt and no contact is recorded"


def test_mesh_jaw_clearance_uses_aabb_not_vertices(world: Any) -> None:
    """AC-PERF-03: mesh jaw geoms report clearance from their local AABB."""
    mujoco, model, data, plan, arm = world
    _park_the_arm(
        mujoco,
        model,
        data,
        arm,
        np.asarray((0.72, -0.42, 1.20), dtype=np.float64),
    )
    geoms = _jaw_collision_geoms(model, arm)
    belt = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "belt"))
    surface = plan.belt.surface_height
    meshes = [
        geom for geom in geoms if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    assert meshes, "the shipped gripper still exposes mesh collision geoms"
    state = _jaw_state(mujoco, model, data, arm, geoms, belt, surface)
    # Parked clear of the belt: the AABB bound must stay a real clearance, not
    # a NaN or an inverted figure from walking vertices the wrong way.
    assert state.clearance > 0.05
    for geom in meshes:
        rotation = data.geom_xmat[geom].reshape(3, 3)
        centre = float(data.geom_xpos[geom][2])
        size = model.geom_size[geom]
        expected = centre - (
            abs(float(rotation[2, 0])) * float(size[0])
            + abs(float(rotation[2, 1])) * float(size[1])
            + abs(float(rotation[2, 2])) * float(size[2])
        )
        assert _lowest_world_z(model, data, geom) == pytest.approx(expected)


def test_headless_without_frames_skips_rgb_render(tmp_path: Path) -> None:
    """AC-PERF-01: no window, no video, no frames means no watching RGB work."""
    from clave.sim.debug_run import run

    report = run(
        root=ROOT,
        out=tmp_path,
        seconds=0.6,
        seed=0,
        window=False,
        video=False,
        frames=False,
        ground_truth_tracker=True,
        progress=False,
    )
    assert report.frames_written == 0
    assert list(tmp_path.glob("frame_*.png")) == []
    assert report.camera_video_path is None
    assert list(tmp_path.glob("camera-debug.mp4")) == []
    assert report.captures > 0


def test_camera_video_writes_the_detection_camera(tmp_path: Path) -> None:
    """AC-CAM-01: the flag writes the detection camera when an encoder exists."""
    import shutil

    from clave.sim.debug_run import run

    report = run(
        root=ROOT,
        out=tmp_path,
        seconds=0.6,
        seed=0,
        window=False,
        video=False,
        frames=False,
        camera_video=True,
        ground_truth_tracker=True,
        progress=False,
    )
    if shutil.which("ffmpeg") is None:
        assert report.camera_video_path is None
        return
    assert report.camera_video_path == tmp_path / "camera-debug.mp4"
    assert report.camera_video_path is not None
    assert report.camera_video_path.is_file()
    assert report.camera_video_path.stat().st_size > 0
    assert "camera video" in (tmp_path / "report.txt").read_text()


def test_no_window_defaults_frames_off_unless_forced(tmp_path: Path) -> None:
    """AC-PERF-02: headless without video skips PNGs; --frames forces them."""
    from clave.sim.debug_run import run

    quiet = run(
        root=ROOT,
        out=tmp_path / "quiet",
        seconds=0.6,
        seed=0,
        window=False,
        video=False,
        frames=None,
        ground_truth_tracker=True,
        progress=False,
    )
    assert quiet.frames_written == 0

    forced = run(
        root=ROOT,
        out=tmp_path / "forced",
        seconds=0.6,
        seed=0,
        window=False,
        video=False,
        frames=True,
        ground_truth_tracker=True,
        progress=False,
    )
    assert forced.frames_written > 0
    assert list((tmp_path / "forced").glob("frame_*.png"))


def test_a_plan_reports_each_leg_for_its_own_duration() -> None:
    """AC-MOVE-38: the phase a tick carries is the leg the tick is in.

    Sampled here rather than in a run, because the run's phase column is what a
    reader trusts when a figure does not reproduce: a `DESCEND` column spanning
    twice the leg the configuration asks for is a reporting fault or a planning
    one, and this says which.
    """
    from clave.control.pick import plan_pick
    from clave.control.trajectory import State

    plan = plan_pick(
        flange=State.at_rest(np.asarray((0.45, -1.00, 1.20), dtype=np.float64)),
        track_id=1,
        object_position=np.asarray((0.30, 0.0, 1.035), dtype=np.float64),
        belt_velocity=np.asarray((0.314, 0.0, 0.0), dtype=np.float64),
        z_offset=0.050,
        approach_speed=0.25,
        dwell_seconds=0.30,
        max_speed=1.00,
        max_acceleration=2.50,
        latest=4.00,
        at_seconds=0.0,
        margin=1.0,
        drift_horizon=0.30,
        minimum_segment_seconds=0.50,
        closing_rise_seconds=0.20,
        segment_sample_count=64,
        bisection_passes=40,
        minimum_delivery_seconds=0.05,
        correction_steps=32,
    )
    assert plan is not None
    tick = 0.001
    spans: list[tuple[Phase, float]] = []
    at = plan.started_at
    while at < plan.started_at + plan.duration:
        sampled = plan.at(at)
        if sampled is None:
            break
        phase = sampled[0]
        if not spans or spans[-1][0] is not phase:
            spans.append((phase, 0.0))
        spans[-1] = (phase, spans[-1][1] + tick)
        at += tick
    assert [phase for phase, _ in spans] == [leg.phase for leg in plan.legs]
    for (_, measured), leg in zip(spans, plan.legs, strict=True):
        assert measured == pytest.approx(leg.segment.duration, abs=2 * tick)
    assert all(duration > 0.0 for _, duration in spans), "a leg was sampled for no time"
    assert sum(duration for _, duration in spans) == pytest.approx(
        plan.duration, abs=2 * tick
    )
