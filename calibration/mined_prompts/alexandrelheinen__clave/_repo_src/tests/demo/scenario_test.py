"""Tests for the one-line demonstration."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from clave.demo.scenario import Scenario, ScenarioError
from clave.demo.video import VideoSettings, available, open_recorder

ROOT = Path(__file__).resolve().parents[2]
DEMOS = ROOT / "configs" / "demos"


def test_every_shipped_scenario_loads(tmp_path: Path) -> None:
    """A scenario name is all a person should have to supply."""
    files = sorted(DEMOS.glob("*.yml"))
    assert files, "no scenario ships with the project"
    for path in files:
        scenario = Scenario.load(path, tmp_path)
        assert scenario.description
        assert scenario.seconds > 0.0
        assert scenario.video is not None


def test_every_tunable_comes_from_the_file(tmp_path: Path) -> None:
    """Including the angle the video is filmed from."""
    scenario = Scenario.load(DEMOS / "sorting_line.yml", tmp_path)
    assert scenario.video is not None
    assert scenario.video.width > 0
    assert scenario.video.height > 0
    assert scenario.video.frames_per_second > 0
    assert scenario.video.distance > 0.0
    assert scenario.video.path == tmp_path / "sorting-line.mp4"


def test_a_missing_key_fails_naming_itself(tmp_path: Path) -> None:
    """A scenario carries no default, as no configuration here does."""
    raw = yaml.safe_load((DEMOS / "sorting_line.yml").read_text())
    del raw["scenario"]["seed"]
    path = tmp_path / "broken.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ScenarioError, match="scenario.seed"):
        Scenario.load(path, tmp_path)


def test_a_scenario_naming_an_unknown_predictor_is_refused(tmp_path: Path) -> None:
    """The demo builds two kinds of predictor and refuses to guess a third."""
    raw = yaml.safe_load((DEMOS / "sorting_line.yml").read_text())
    raw["scenario"]["predictor"] = "vibes"
    path = tmp_path / "broken.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ScenarioError, match="vibes"):
        Scenario.load(path, tmp_path)


def settings(path: Path) -> VideoSettings:
    """Recording settings small enough for a test."""
    return VideoSettings(
        path=path,
        width=64,
        height=48,
        frames_per_second=10,
        interval_seconds=0.1,
        azimuth=150.0,
        elevation=-20.0,
        distance=2.0,
        lookat=np.asarray((0.0, 0.0, 0.4), dtype=np.float64),
    )


@pytest.mark.skipif(not available(), reason="ffmpeg is not installed here")
def test_the_recorder_writes_a_playable_file(tmp_path: Path) -> None:
    """The demonstration ends in something a person can watch."""
    path = tmp_path / "demo.mp4"
    recorder = open_recorder(settings(path))
    assert recorder is not None
    for index in range(10):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        recorder.write(frame)
    recorder.close()

    assert recorder.frames == 10
    assert path.stat().st_size > 0
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "64,48"


@pytest.mark.skipif(not available(), reason="ffmpeg is not installed here")
def test_the_recorder_writes_what_it_was_given_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Nothing is drawn on a frame after the simulator made it.

    One flat gray frame in, one flat gray frame out. An overlay, a caption or a
    composited box would show up as a pixel that is not the value written.
    """
    path = tmp_path / "flat.mp4"
    recorder = open_recorder(settings(path))
    assert recorder is not None
    for _ in range(10):
        recorder.write(np.full((48, 64, 3), 128, dtype=np.uint8))
    recorder.close()

    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8)
    assert pixels.size == 48 * 64 * 3
    # Lossy encoding moves a value by a little; an overlay moves it by a lot.
    assert int(pixels.min()) >= 120
    assert int(pixels.max()) <= 136


def test_an_absent_encoder_is_reported_rather_than_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No encoder means no video, not a failed demonstration."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert not available()
    assert open_recorder(settings(tmp_path / "nothing.mp4")) is None


def test_the_recorder_drains_the_encoder_rather_than_deadlocking(
    tmp_path: Path,
) -> None:
    """The encoder's stderr is a pipe and an undrained pipe is a deadlock.

    Its buffer is about 64 kB. An encoder that fills it blocks writing
    there, stops reading its stdin, and never exits, and the caller then
    waits on a process that is waiting on the caller. Short recordings hide
    it, which is why this writes enough to matter: the run that found it
    was sixty seconds long and the one before it, at five, was fine.
    """
    from clave.demo.video import VideoSettings, open_recorder

    recorder = open_recorder(
        VideoSettings(
            path=tmp_path / "drained.mp4",
            width=64,
            height=48,
            frames_per_second=30,
            interval_seconds=0.03,
            azimuth=90.0,
            elevation=-20.0,
            distance=3.0,
            lookat=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        )
    )
    if recorder is None:
        pytest.skip("no encoder installed")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(400):
        recorder.write(frame)
    recorder.close()
    assert (tmp_path / "drained.mp4").stat().st_size > 0
