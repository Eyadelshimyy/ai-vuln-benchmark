"""Encoding what the simulator rendered.

Frames go to `ffmpeg` on standard input as raw `rgb24`. That needs no imaging
library in the wheel and no new dependency, and it keeps the encoder a tool the
operator already has rather than one this project pins.

A machine without `ffmpeg` runs the scenario anyway. A missing encoder means no
video, not a failed demonstration.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from clave.errors import ClaveError

ENCODER = "ffmpeg"

CLOSE_TIMEOUT_SECONDS = 120.0
"""How long to give the encoder to finish after the last frame.

Generous, because the encoder still has queued frames to write when the
last one arrives. Bounded, because the alternative to a bound is a run that
never ends.
"""


class VideoError(ClaveError):
    """The encoder failed, so the recording cannot be trusted."""


"""The tool that turns raw frames into a file."""


@dataclass(frozen=True)
class VideoSettings:
    """How to record, all of it from the scenario file.

    Attributes:
        path: Where the video goes.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frames_per_second: Playback rate.
        interval_seconds: Simulated seconds between rendered frames. The loop
            decides every half simulated second, which is far too sparse to
            watch, so the video renders on its own cadence.
        azimuth: Camera azimuth in degrees.
        elevation: Camera elevation in degrees.
        distance: Camera distance from its target, in meters.
        lookat: What the camera points at, in world meters.
    """

    path: Path
    width: int
    height: int
    frames_per_second: int
    interval_seconds: float
    azimuth: float
    elevation: float
    distance: float
    lookat: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lookat", np.asarray(self.lookat, dtype=np.float64))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VideoSettings):
            return NotImplemented
        return (
            self.path == other.path
            and self.width == other.width
            and self.height == other.height
            and self.frames_per_second == other.frames_per_second
            and self.interval_seconds == other.interval_seconds
            and self.azimuth == other.azimuth
            and self.elevation == other.elevation
            and self.distance == other.distance
            and bool(np.allclose(self.lookat, other.lookat, rtol=0.0, atol=1e-12))
        )

    @property
    def speed(self) -> float:
        """How fast the recording plays against simulated time."""
        return self.frames_per_second * self.interval_seconds


@dataclass(frozen=True)
class StreamSettings:
    """Where a sequence of already-rendered frames should be encoded.

    The free-camera pose on `VideoSettings` is for a demonstration that chooses
    its own viewpoint. A corpus preview already has the pixels.

    Attributes:
        path: Where the video goes.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frames_per_second: Playback rate.
    """

    path: Path
    width: int
    height: int
    frames_per_second: int


class VideoRecorder:
    """An open encoder, taking one frame at a time."""

    def __init__(self, settings: VideoSettings | StreamSettings) -> None:
        """Start the encoder.

        Args:
            settings: Where and how to record.

        Raises:
            FileNotFoundError: If the encoder is not installed.
        """
        settings.path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.frames = 0
        """How many frames were written."""
        self._process = subprocess.Popen(
            [
                ENCODER,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{settings.width}x{settings.height}",
                "-r",
                str(settings.frames_per_second),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                # Chroma subsampling every player understands. Without it the
                # file plays in ffplay and in almost nothing else.
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                str(settings.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            # Captured rather than discarded, so a failed encode says why.
            # It is drained in `close`, which is what keeps it from filling.
            stderr=subprocess.PIPE,
        )

    def write(self, frame: NDArray[np.uint8]) -> None:
        """Write one rendered frame.

        Args:
            frame: Height by width by three bytes, as the renderer produced it.
                Nothing is drawn on it and nothing is composited into it.
        """
        if self._process.stdin is None:
            return
        self._process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        """Finish the file and wait for the encoder.

        Draining stderr is not optional and leaving it undrained is a
        deadlock rather than untidiness. The encoder's stderr is a pipe with
        a buffer of about 64 kB, and an encoder that fills it blocks writing
        there, stops reading its stdin, and never exits; the caller then
        waits on a process that is waiting on the caller. Short recordings
        hide it because the encoder never writes enough to fill the buffer,
        so it appears as a run that hangs only once it is long enough to
        matter.

        `communicate` reads both pipes and waits in one step, which is the
        only combination that cannot deadlock.

        Raises:
            VideoError: If the encoder failed, carrying what it wrote to
                stderr. A recording that silently produced an unplayable
                file is worse than one that says it did not work.
        """
        # `communicate` closes stdin itself, so closing it first leaves it
        # with a closed file to flush and raises rather than waiting.
        try:
            _, complaint = self._process.communicate(timeout=CLOSE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # Killing beats leaving it behind. A caller that has returned its
            # report is about to exit, and an encoder still holding the
            # output file keeps the process alive with nothing left to do.
            self._process.kill()
            _, complaint = self._process.communicate()
            raise VideoError(
                f"{ENCODER} did not finish within {CLOSE_TIMEOUT_SECONDS} s "
                f"and was killed, so {self.settings.path} is incomplete"
            ) from None
        if self._process.returncode:
            raise VideoError(
                f"{ENCODER} exited {self._process.returncode}: "
                f"{complaint.decode(errors='replace').strip() or 'no message'}"
            )


def available() -> bool:
    """Whether an encoder is installed here."""
    return shutil.which(ENCODER) is not None


def open_stream(settings: StreamSettings) -> VideoRecorder | None:
    """Start encoding already-rendered frames, or say there is no encoder.

    Args:
        settings: Path, size and playback rate.

    Returns:
        The recorder, or None when `ffmpeg` is absent.
    """
    if not available():
        return None
    return VideoRecorder(settings)


def open_recorder(settings: VideoSettings) -> VideoRecorder | None:
    """Start recording, or say there is no encoder.

    Args:
        settings: Where and how to record.

    Returns:
        The recorder, or None when `ffmpeg` is absent. A caller that gets None
        runs the scenario and reports that no video was recorded.
    """
    if not available():
        return None
    return VideoRecorder(settings)
