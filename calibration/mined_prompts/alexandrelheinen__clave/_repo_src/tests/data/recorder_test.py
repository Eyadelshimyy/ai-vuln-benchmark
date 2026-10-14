"""Tests for the rollout recorder.

These need MuJoCo and the pinned menagerie submodule, and skip when either is
absent, as the world tests do.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from clave.data.examples import Rollout
from clave.world import arm as armmod

ROOT = Path(__file__).resolve().parents[2]

_PROBE = (
    "import os;os.environ.setdefault('MUJOCO_GL','osmesa');"
    "import mujoco;"
    "m=mujoco.MjModel.from_xml_string("
    '\'<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>\');'
    "mujoco.Renderer(m,height=8,width=8)"
)


def _rendering_available() -> bool:
    """Whether offscreen rendering works here.

    MuJoCo aborts the process when no GL backend is present, so this cannot be
    a try/except around an import: the probe runs in a subprocess where an abort
    kills the child rather than the test session.
    """

    import subprocess
    import sys

    try:
        return (
            subprocess.run(
                [sys.executable, "-c", _PROBE], capture_output=True, timeout=60
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


RENDERS = _rendering_available()
needs_rendering = pytest.mark.skipif(
    not RENDERS, reason="no offscreen GL backend available for MuJoCo"
)


def record_short(seed: int, rollout_id: str = "r0") -> Rollout:
    """Record a short rollout for testing.

    Sixteen simulated seconds rather than four. The belt runs at 0.02 to 0.05
    m/s since v1.0.2, and an object entering 0.52 m upstream of the camera
    needs about thirteen seconds to reach the frame. At four seconds no object
    was ever in view, which is a property of a slow belt rather than of the
    recorder.
    """
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from clave.data.recorder import record

    return record(
        root=ROOT,
        seed=seed,
        seconds=16.0,
        capture_interval_seconds=1.0,
        height=48,
        width=64,
        rollout_id=rollout_id,
    )


@needs_rendering
def test_a_rollout_captures_labeled_frames() -> None:
    """A rollout captures labeled frames."""
    rollout = record_short(0)
    assert rollout.examples
    example = rollout.examples[-1]
    assert example.frame.shape == (48, 64, 3)
    assert example.frame.dtype == np.uint8
    for label in example.labels:
        assert label.material_class.startswith("M-")
        assert label.channel.startswith("CH-")


@needs_rendering
def test_every_example_carries_seed_time_and_config_digest() -> None:
    """A dataset must say what produced it."""
    rollout = record_short(2)
    for example in rollout.examples:
        assert example.seed == 2
        assert example.simulated_time >= 0.0
        assert len(example.config_digest) == 64


@needs_rendering
def test_one_seed_twice_records_identical_examples() -> None:
    """Two runs must be comparable."""
    first, second = record_short(7), record_short(7)
    assert len(first.examples) == len(second.examples)
    assert np.array_equal(first.examples[-1].frame, second.examples[-1].frame)
    assert first.examples[-1].material_classes == second.examples[-1].material_classes


@needs_rendering
def test_two_seeds_record_different_examples() -> None:
    """The guard above would pass for a constant renderer."""
    first, second = record_short(7), record_short(8)
    assert not np.array_equal(first.examples[-1].frame, second.examples[-1].frame)


@needs_rendering
def test_visible_labels_carry_pixel_boxes_inside_the_frame() -> None:
    """Boxes come from a segmentation render, so they are ground truth."""
    rollout = record_short(0, "boxes")
    visible = [
        label for example in rollout.examples for label in example.visible_labels
    ]
    assert visible, "no object was ever in frame"
    for label in visible:
        x_min, y_min, x_max, y_max = label.bbox or (0, 0, 0, 0)
        assert 0 <= x_min <= x_max < 64
        assert 0 <= y_min <= y_max < 48


@needs_rendering
def test_some_labeled_objects_are_outside_the_camera_view() -> None:
    """The camera sees part of the belt, so labels and visibility differ."""
    rollout = record_short(0, "coverage")
    labeled = sum(len(example.labels) for example in rollout.examples)
    visible = sum(len(example.visible_labels) for example in rollout.examples)
    assert labeled > 0
    assert visible <= labeled


@needs_rendering
def test_recorded_examples_carry_a_moving_arm() -> None:
    """The arm must actually move, or proprioception carries no information."""
    rollout = record_short(0, "proprio")
    joints = np.array([example.arm_joints for example in rollout.examples])
    # Six, because the arm is a six-axis UR10e. The count is asserted rather
    # than inferred so a silent change of manipulator fails here.
    assert joints.shape[1] == len(armmod.ARM_JOINTS) == 6
    assert float(np.ptp(joints, axis=0).max()) > 1e-3
