"""Tests for the loop that closes CLAVE.

The end-to-end case needs MuJoCo, offscreen rendering and the built runtime,
and skips when any is absent, as the world tests do.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from clave.runtime import loop
from clave.runtime.inference import ScriptedPredictor, associate
from clave.world import config

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "configs" / "runtime" / "sitl.yml"

_PROBE = (
    "import os;os.environ.setdefault('MUJOCO_GL','osmesa');"
    "import mujoco;"
    "m=mujoco.MjModel.from_xml_string("
    '\'<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>\');'
    "mujoco.Renderer(m,height=8,width=8)"
)


def _rendering_available() -> bool:
    """Whether offscreen rendering works here.

    MuJoCo aborts the process when no GL backend is present, so the probe runs
    in a subprocess where an abort kills the child rather than the session.
    """
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", _PROBE], capture_output=True, timeout=60
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def test_the_shipped_settings_load() -> None:
    """Every key the loop reads is present in the committed configuration."""
    settings = loop.RuntimeSettings.load(SETTINGS)
    assert settings.perception == "resnet50-baseline"
    assert settings.routing.reject_channel == 0
    assert settings.routing.channels["M-11"] == settings.routing.reject_channel


def test_a_missing_key_fails_naming_itself(tmp_path: Path) -> None:
    """Nothing in the runtime carries a default, as nothing in the world does."""
    raw = yaml.safe_load(SETTINGS.read_text())
    del raw["runtime"]["seed"]
    path = tmp_path / "sitl.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(loop.RuntimeConfigError, match="runtime.seed"):
        loop.RuntimeSettings.load(path)


def test_a_channel_map_naming_an_unknown_class_is_refused(tmp_path: Path) -> None:
    """The taxonomy owns the class set, and the line owns the channel numbers."""
    raw = yaml.safe_load(SETTINGS.read_text())
    raw["routing"]["channels"]["M-99"] = 3
    path = tmp_path / "sitl.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(loop.RuntimeConfigError, match="M-99"):
        loop.RuntimeSettings.load(path)


def test_the_envelope_is_derived_from_the_world_rather_than_restated() -> None:
    """A second copy of the belt geometry is a second copy that can drift."""
    settings = loop.RuntimeSettings.load(SETTINGS)
    world = config.load(ROOT / "configs" / "world" / "sorting_line.yml")
    derived = loop.runtime_config(world, settings.routing)
    belt = config.require(world, "belt")
    arm = config.require(world, "arm")
    length = float(config.require(belt, "length_meters", "belt"))
    width = float(config.require(belt, "width_meters", "belt"))
    # Derived, not restated: the assertion reads the world rather than repeating
    # numbers that would have to be edited here every time the line is rescaled.
    assert derived["belt_x_meters"] == [-length / 2.0, length / 2.0]
    assert derived["belt_y_meters"] == [-width / 2.0, width / 2.0]
    assert derived["belt_surface_z_meters"] == float(
        config.require(belt, "surface_height_meters", "belt")
    )
    base = [float(v) for v in config.require(arm, "base_position_meters", "arm")]
    # The arm's mounting face, which is what the reach test is radial about.
    assert derived["base_meters"] == base
    assert derived["reach_meters"] == [
        float(config.require(arm, "reach_min_meters", "arm")),
        float(config.require(arm, "reach_max_meters", "arm")),
    ]
    assert derived["tool_above_base_meters"] == [
        float(v) for v in config.require(arm, "tool_above_base_meters", "arm")
    ]


def test_a_percentile_is_an_observed_sample_rather_than_an_interpolation() -> None:
    """A reported p99 should be a latency that actually happened."""
    report = loop.RunReport(predictor="test")
    report.latencies_seconds = [float(index) / 100.0 for index in range(100)]
    assert report.percentile(0.99) == pytest.approx(0.99)
    assert report.percentile(0.50) == pytest.approx(0.50)
    assert loop.RunReport(predictor="test").percentile(0.99) == 0.0


def test_the_budget_comes_from_the_window_and_the_belt_speed() -> None:
    """The budget is what every latency has to fit inside."""
    report = loop.RunReport(predictor="test", belt_speed=0.30, window_length=0.339)
    assert report.budget_seconds == pytest.approx(1.13)
    assert loop.RunReport(predictor="test").budget_seconds == 0.0


def test_a_prediction_matching_no_object_is_associated_with_nothing() -> None:
    """CLAVE has no tracker, and a point on empty belt names no object."""
    from clave.data.examples import ObjectLabel

    label = ObjectLabel(
        object_id=3,
        material_class="M-01",
        channel="CH-PET",
        position=np.asarray((0.0, 0.0, 0.36), dtype=np.float64),
        in_reachable_window=True,
    )
    assert (
        associate(np.asarray((0.01, 0.0, 0.36), dtype=np.float64), (label,), 0.30)
        is label
    )
    assert (
        associate(np.asarray((5.0, 0.0, 0.36), dtype=np.float64), (label,), 0.30)
        is None
    )
    assert associate(np.asarray((0.0, 0.0, 0.0), dtype=np.float64), (), 0.30) is None


def test_the_loop_runs_end_to_end_and_reports_what_it_did(
    tmp_path: Path, runtime_binary: Path
) -> None:
    """One command steps the world, decides, publishes and counts."""
    pytest.importorskip("mujoco")
    if not _rendering_available():
        pytest.skip("no offscreen GL backend here")

    settings = loop.RuntimeSettings.load(SETTINGS)
    short = loop.RuntimeSettings(**{**vars(settings), "seconds": 6.0})
    report = loop.run(ROOT, short, ScriptedPredictor(), tmp_path / "run")

    assert report.frames > 0
    assert report.predictor == "scripted-expert"
    assert report.machine
    # The scripted expert declines when nothing is reachable, so every frame
    # is either a proposal or a silence.
    assert report.proposals + report.silent_frames == report.frames
    assert report.counters["proposals"] == report.proposals
    assert report.decisions_received == report.counters["published"]
    assert report.budget_seconds > 0.0
    if report.proposals:
        assert report.percentile(0.99) > 0.0

    record = tmp_path / "sitl.json"
    loop.write_record(record, report)
    assert "p99" in record.read_text()
