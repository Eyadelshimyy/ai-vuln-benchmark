"""Tests for what one demonstration builds and prints."""

import shutil
from pathlib import Path

import pytest

from clave.demo.scenario import Scenario
from clave.runtime.inference import ScriptedPredictor
from clave.runtime.loop import RunReport, RuntimeSettings
from clave.runtime.play import predictor_for, summary

ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / "configs" / "demos"
RUNTIME = ROOT / "configs" / "runtime" / "sitl.yml"


def played(**overrides: object) -> RunReport:
    """A report shaped like one a scenario produces."""
    report = RunReport(predictor="scripted-expert", belt_speed=0.25, window_length=0.34)
    report.frames = 40
    report.proposals = 30
    report.presented = [(0, "M-01"), (1, "M-05")]
    report.latencies_seconds = [0.001, 0.002, 0.003]
    report.counters = {
        "published": 21,
        "overridden_reach": 7,
        "overridden_belt_surface": 2,
        "overridden_belt_extent": 0,
    }
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def test_a_scripted_scenario_builds_the_teacher(tmp_path: Path) -> None:
    """The scripted predictor needs no checkpoint and no deep learning library."""
    scenario = Scenario.load(DEMOS / "sorting_line.yml", tmp_path)
    settings = RuntimeSettings.load(RUNTIME)
    assert isinstance(predictor_for(scenario, settings, ROOT), ScriptedPredictor)


def test_the_summary_reports_what_the_scenario_did(tmp_path: Path) -> None:
    """Decisions published and what the safety layer overrode."""
    scenario = Scenario.load(DEMOS / "sorting_line.yml", tmp_path)
    rendered = summary(scenario, played())
    assert "sorting-line" in rendered
    assert "2 objects reached the arm" in rendered
    assert "21 decisions" in rendered
    assert "9 by the safety layer" in rendered
    assert "median" in rendered and "p99" in rendered


def test_the_summary_names_a_recorded_video(tmp_path: Path) -> None:
    """A person should be told where the recording went."""
    scenario = Scenario.load(DEMOS / "sorting_line.yml", tmp_path)
    rendered = summary(
        scenario, played(video_path="runs/demos/x.mp4", video_frames=489)
    )
    assert "runs/demos/x.mp4" in rendered
    assert "489 frames" in rendered


def test_the_summary_says_when_no_encoder_recorded_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent encoder is stated, not silently skipped."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    scenario = Scenario.load(DEMOS / "sorting_line.yml", tmp_path)
    rendered = summary(scenario, played())
    assert "NO VIDEO" in rendered


def test_the_summary_says_when_ros_was_asked_for_and_absent(tmp_path: Path) -> None:
    """A scenario that asked to publish should say whether it did."""
    scenario = Scenario.load(DEMOS / "sorting_line.yml", tmp_path)
    rendered = summary(
        scenario, played(ros_unavailable_reason="rclpy is not importable")
    )
    assert "NO ROS" in rendered


def _rendering_available() -> bool:
    """Whether offscreen rendering works here.

    MuJoCo aborts the process when no GL backend is present, so the probe runs
    in a subprocess where an abort kills the child rather than the session.
    """
    import subprocess
    import sys

    model = '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
    probe = (
        "import os;os.environ.setdefault('MUJOCO_GL','osmesa');"
        "import mujoco;"
        f"m=mujoco.MjModel.from_xml_string({model!r});"
        "mujoco.Renderer(m,height=8,width=8)"
    )
    try:
        return (
            subprocess.run(
                [sys.executable, "-c", probe], capture_output=True, timeout=60
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def test_a_scenario_runs_end_to_end_and_writes_its_record(
    tmp_path: Path, runtime_binary: Path
) -> None:
    """One scenario name is the whole input."""
    pytest.importorskip("mujoco")
    if not _rendering_available():
        pytest.skip("no offscreen GL backend here")
    import yaml

    from clave.runtime.play import play

    # A short run, because what this proves is that the path works rather than
    # what the run decided.
    raw = yaml.safe_load((DEMOS / "sorting_line.yml").read_text())
    raw["scenario"]["seconds"] = 4.0
    raw["video"]["width"] = 160
    raw["video"]["height"] = 120
    path = tmp_path / "short.yml"
    path.write_text(yaml.safe_dump(raw))

    scenario = Scenario.load(path, tmp_path)
    report = play(ROOT, scenario, Path("configs/runtime/sitl.yml"), tmp_path)

    assert report.frames > 0
    assert (tmp_path / f"{scenario.name}.json").is_file()
    assert summary(scenario, report)
