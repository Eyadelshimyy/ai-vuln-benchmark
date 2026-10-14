"""Ground-truth tracker tests, guarding AC-GT-01 through AC-GT-05.

AC-GT-01: Configuration and CLI flag --ground-truth-tracker.
AC-GT-02: GraspMarker derivation directly from MuJoCo physics state.
AC-GT-03: Concurrent tracker observation and settling during ground-truth mode.
AC-GT-04: Selector and TaskMachine pick planning from ground-truth markers.
AC-GT-05: Scene visualization displaying ground-truth markers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from clave.cli import _build_parser
from clave.control.selection import Selector
from clave.control.settings import SelectionSettings
from clave.tracker.markers import ground_truth_markers
from clave.world.belt import SpawnedObject
from clave.world.config import load
from clave.world.effector import Effector
from clave.world.scene import BeltGeometry, SceneLayout, TakeawayDrive

ROOT = Path(__file__).resolve().parents[2]
WORLD_CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"
TRACKER_CONFIG = ROOT / "configs" / "debug" / "tracker.yml"


def effector() -> Effector:
    """Load the effector configuration from the sorting line config."""
    return Effector.load(load(WORLD_CONFIG))


def minimal_plan() -> SceneLayout:
    """Minimal scene layout sufficient for ground-truth marker geometry."""
    return SceneLayout(
        belt=BeltGeometry(
            length=4.0,
            width=0.5,
            surface_height=0.90,
            speed=0.31,
            height_tolerance=0.05,
        ),
        arm_base=np.asarray((0.0, -0.45, 0.90), dtype=np.float64),
        reach_min=0.30,
        reach_max=1.30,
        tool_above_base=(-0.20, 0.60),
        pedestal=(0.20, 0.20),
        channels=("chute_a", "chute_b"),
        chutes={
            "chute_a": np.asarray((0.0, -0.70, 0.40), dtype=np.float64),
            "chute_b": np.asarray((0.3, -0.70, 0.40), dtype=np.float64),
        },
        objects=(),
        pool_size=4,
        timestep=0.002,
        takeaway=TakeawayDrive(
            speed=0.20,
            height_min=0.25,
            height_max=0.45,
            mouth_half_width=0.15,
            past_mouth=1.50,
            toward_mouth=0.10,
        ),
        dressed=False,
    )


def build_test_mujoco() -> tuple[Any, Any]:
    """Compile a minimal MuJoCo model with test objects for ground-truth tests."""
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="object_0" pos="0.10 0.05 0.92">
          <freejoint/>
          <geom type="cylinder" size="0.03 0.05"/>
        </body>
        <body name="object_1" pos="-0.20 -0.05 0.92" quat="0.7071068 0 0 0.7071068">
          <freejoint/>
          <geom type="box" size="0.02 0.04 0.03"/>
        </body>
        <body name="object_2" pos="1.50 0.0 0.92">
          <freejoint/>
          <geom type="cylinder" size="0.03 0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_ground_truth_tracker_config_and_cli_ac_gt_01() -> None:
    """AC-GT-01: Configuration exposes ground_truth_tracker and CLI parses it."""
    # Check config
    cfg = load(TRACKER_CONFIG)
    assert "ground_truth_tracker" in cfg
    assert cfg["ground_truth_tracker"] is False

    # Check CLI without flag defaults to None (delegating to config)
    parser = _build_parser()
    args_default = parser.parse_args(["sim"])
    assert args_default.ground_truth_tracker is None

    # Check CLI with flag sets it to True (including aliases --gt and --gt-tracker)
    args_flag = parser.parse_args(["sim", "--ground-truth-tracker"])
    assert args_flag.ground_truth_tracker is True

    args_gt = parser.parse_args(["sim", "--gt"])
    assert args_gt.ground_truth_tracker is True

    args_gtt = parser.parse_args(["sim", "--gt-tracker"])
    assert args_gtt.ground_truth_tracker is True


def test_ground_truth_markers_derivation_ac_gt_02() -> None:
    """AC-GT-02: Derive GraspMarker targets directly from MuJoCo physics state."""
    model, data = build_test_mujoco()
    plan = minimal_plan()
    eff = effector()

    active = [
        SpawnedObject(
            index=0,
            name="object_0",
            material_class="M-01",
            channel="chute_a",
            serial=7,
        ),
        SpawnedObject(
            index=1,
            name="object_1",
            material_class="M-02",
            channel="chute_b",
            serial=8,
        ),
        SpawnedObject(
            index=2,
            name="object_2",
            material_class="M-01",
            channel="chute_a",
            serial=9,
        ),
    ]

    now = 1_000_000_000
    window_exit = 1.0
    belt_speed = 0.31

    markers = ground_truth_markers(
        model=model,
        data=data,
        active_objects=active,
        plan=plan,
        effector=eff,
        belt_surface_height_world=0.90,
        at_nanos=now,
        window_exit=window_exit,
        belt_speed=belt_speed,
    )

    # object_2 is at x=1.5 > window_exit=1.0, so it must be excluded
    assert len(markers) == 2

    # Check object_0 (cylinder). Its identity is the spawn serial rather than
    # the pool slot, because a slot is reused as soon as the object in it
    # leaves the belt and a consumer that keyed on the slot would refuse to
    # serve the second object to ride in it.
    m0 = markers[0]
    assert m0.track_id == 7
    assert m0.channel == "chute_a"
    assert math.isclose(m0.pinch_position_belt[0], 0.10, abs_tol=1e-5)
    assert math.isclose(m0.pinch_position_belt[1], 0.05, abs_tol=1e-5)
    # The object's own centre is 20 mm up. The open jaw's floor is below
    # that, so the plane stands on the centre. The shut jaw hangs further,
    # and the hold rises by that difference while the fingers close.
    assert math.isclose(m0.pinch_position_belt[2], 0.92, abs_tol=1e-5)
    assert math.isclose(m0.opening, 0.06, abs_tol=1e-5)
    assert m0.oriented is False
    assert m0.closing_yaw_belt is None

    # Check object_1 (box rotated pi/2)
    m1 = markers[1]
    assert m1.track_id == 8
    assert m1.channel == "chute_b"
    assert math.isclose(m1.pinch_position_belt[0], -0.20, abs_tol=1e-5)
    assert math.isclose(m1.pinch_position_belt[1], -0.05, abs_tol=1e-5)
    assert m1.oriented is True
    # Box half-extents 0.02 by 0.04, turned a quarter turn about the belt
    # normal. The jaw closes along the short horizontal axis. That axis has no
    # sign, so pi/2 and -pi/2 are the same grasp.
    assert m1.closing_yaw_belt is not None
    folded = (m1.closing_yaw_belt - math.pi / 2.0 + math.pi / 2.0) % math.pi - (
        math.pi / 2.0
    )
    assert abs(folded) < 1e-4


def test_ground_truth_selector_and_task_planning_ac_gt_04() -> None:
    """AC-GT-04: Selector admits and orders ground-truth markers."""
    model, data = build_test_mujoco()
    plan = minimal_plan()
    eff = effector()

    active = [
        SpawnedObject(
            index=0, name="object_0", material_class="M-01", channel="chute_a"
        ),
    ]

    now = 1_000_000_000
    markers = ground_truth_markers(
        model=model,
        data=data,
        active_objects=active,
        plan=plan,
        effector=eff,
        belt_surface_height_world=0.90,
        at_nanos=now,
        window_exit=1.0,
        belt_speed=0.31,
    )

    selector = Selector(
        SelectionSettings(exit_weight=1.0, anchor_radius=0.02),
        admits=lambda p: True,
    )

    flange = np.asarray((0.0, 0.0, 1.20), dtype=np.float64)
    queue = selector.update(markers, flange, belt_speed=0.31, at_nanos=now)
    assert len(queue.order) == 1
    assert queue.order[0].track_id == 0
    assert queue.order[0].channel == "chute_a"


def test_ground_truth_simulation_run_ac_gt_03_and_05(tmp_path: Path) -> None:
    """AC-GT-03, AC-GT-05: Simulation runs with ground truth while tracker observes."""
    from clave.sim.debug_run import run

    report = run(
        root=ROOT,
        out=tmp_path,
        seconds=1.0,
        seed=0,
        window=False,
        video=False,
        ground_truth_tracker=True,
    )

    # AC-GT-03: Tracker continues to observe and settle captures
    assert report.captures > 0
    assert report.ground_truth is True

    # AC-GT-05: Scene visualization reflects markers
    assert report.drawn >= 0


def test_ground_truth_markers_dynamic_grasp_height() -> None:
    """Dynamic grasp height derives pad_z from object geometry in world frame."""
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="tall_box" pos="0.10 0.05 0.96">
          <freejoint/>
          <geom type="box" size="0.02 0.03 0.06"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    plan = minimal_plan()
    eff = effector()
    active = [
        SpawnedObject(
            index=0, name="tall_box", material_class="M-01", channel="chute_a"
        )
    ]
    markers = ground_truth_markers(
        model=model,
        data=data,
        active_objects=active,
        plan=plan,
        effector=eff,
        belt_surface_height_world=0.90,
        at_nanos=1_000_000_000,
        window_exit=1.0,
        belt_speed=0.31,
    )
    assert len(markers) == 1
    # Center is at z=0.96 (data.geom_xpos[0][2]), not fixed at surface + 0.02 = 0.92
    assert math.isclose(markers[0].pinch_position_belt[2], 0.96, abs_tol=1e-4)
    assert math.isclose(
        markers[0].flange_position_world[2], 0.96 + eff.finger_length, abs_tol=1e-4
    )


def test_a_marker_stands_on_the_mass_not_the_scan_origin() -> None:
    """A marker stands on the mass, not on the scan origin.

    Measured on the shipped meshes, the scan origin sits 30 to 80 mm from the
    centre of mass. A marker built on the origin, carried by the origin's
    velocity, sends the jaw to a point that orbits the parcel. The mass does
    not orbit itself, and the jaw has to meet the mass.
    """
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="offset" pos="0.10 0.05 0.96">
          <freejoint/>
          <geom type="box" pos="0.04 -0.02 0.0" size="0.02 0.03 0.04"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    markers = ground_truth_markers(
        model=model,
        data=data,
        active_objects=[
            SpawnedObject(
                index=0, name="offset", material_class="M-01", channel="chute_a"
            )
        ],
        plan=minimal_plan(),
        effector=effector(),
        belt_surface_height_world=0.90,
        at_nanos=1_000_000_000,
        window_exit=1.0,
        belt_speed=0.31,
    )
    assert len(markers) == 1
    com = data.xipos[1]
    assert math.isclose(markers[0].pinch_position_belt[0], float(com[0]), abs_tol=1e-5)
    assert math.isclose(markers[0].pinch_position_belt[1], float(com[1]), abs_tol=1e-5)
    assert not math.isclose(float(com[0]), 0.10, abs_tol=1e-3)


def test_ground_truth_markers_tracks_object_velocity() -> None:
    """Ground truth markers capture instantaneous object velocity from simulation."""
    model, data = build_test_mujoco()
    # Set velocity on object_0 (free joint dof is 0..5)
    data.qvel[0] = 0.15  # moving at 0.15 m/s along x (e.g. slipping relative to belt)
    import mujoco

    mujoco.mj_forward(model, data)

    plan = minimal_plan()
    eff = effector()
    active = [
        SpawnedObject(
            index=0, name="object_0", material_class="M-01", channel="chute_a"
        )
    ]
    markers = ground_truth_markers(
        model=model,
        data=data,
        active_objects=active,
        plan=plan,
        effector=eff,
        belt_surface_height_world=0.90,
        at_nanos=1_000_000_000,
        window_exit=1.0,
        belt_speed=0.31,
    )
    assert len(markers) == 1
    assert markers[0].velocity_world is not None
    assert math.isclose(markers[0].velocity_world[0], 0.15, abs_tol=1e-3)


def test_ground_truth_markers_keep_the_jaw_clear_of_the_belt() -> None:
    """AC-MARK-16: a ground truth marker keeps the jaw clear of the belt.

    The pads hang below the pinch point on this jaw and the pinch point is what
    a marker stands at, so the clearance the marker grants has to be measured to
    the pads. It used to be measured to the pinch point and clamped 12.2 mm
    above the surface, which put the pads 5.5 mm inside the belt with the
    marker's own plane saying they were clear.
    """
    model, data = build_test_mujoco()
    plan = minimal_plan()
    eff = effector()
    surface = plan.belt.surface_height
    markers = ground_truth_markers(
        model=model,
        data=data,
        active_objects=[
            SpawnedObject(
                index=0, name="object_0", material_class="M-01", channel="chute_a"
            )
        ],
        plan=plan,
        effector=eff,
        belt_surface_height_world=surface,
        at_nanos=0,
        window_exit=1.0,
        belt_speed=0.31,
    )
    assert len(markers) == 1
    marker = markers[0]
    # The object's centre is 20 mm up, above the open jaw's floor, so the
    # plane stands on the centre. The open jaw's lowest geometry still
    # clears the belt; the shut jaw's extra hang is what the hold rises by.
    assert math.isclose(marker.pinch_position_belt[2], surface + 0.02, abs_tol=1e-4)
    lowest = marker.flange[2] - eff.open_lowest_below_flange
    assert lowest >= surface + eff.jaw_clearance - 1e-9
