"""One demonstration's settings.

Every tunable lives in the scenario file, including the camera the video films
from. A demonstration whose belt speed or viewing angle is buried in Python is
one nobody can vary without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from clave.demo.video import VideoSettings
from clave.errors import ClaveError


class ScenarioError(ClaveError):
    """A scenario is missing a key or names something unknown."""


@dataclass(frozen=True)
class Scenario:
    """What one demonstration runs.

    Attributes:
        name: The scenario's own name, used in output paths.
        description: What it is meant to show, printed before it runs.
        predictor: `scripted` for the teacher, `checkpoints` for trained models.
        perception: Registry name of the classifier, when there is one.
        policy: Registry name of the pick policy, when there is one.
        seed: Seed controlling belt speed, placement and spawn timing.
        seconds: Simulated seconds to run.
        publish_to_ros: Whether to put every decision on a ROS 2 topic.
        video: How to record it, or None when the scenario records nothing.
    """

    name: str
    description: str
    predictor: str
    perception: str | None
    policy: str | None
    seed: int
    seconds: float
    publish_to_ros: bool
    video: VideoSettings | None

    @classmethod
    def load(cls, path: Path, output: Path) -> Scenario:
        """Read a scenario from YAML.

        Args:
            path: The scenario file.
            output: Where the video goes, since that is a property of the run
                rather than of the scenario.

        Returns:
            The scenario.

        Raises:
            ScenarioError: If a key is absent or the predictor is unknown.
        """
        raw = yaml.safe_load(path.read_text())
        scenario = _require(raw, "scenario")
        name = str(_require(scenario, "name", "scenario"))
        predictor = str(_require(scenario, "predictor", "scenario"))
        if predictor not in {"scripted", "checkpoints"}:
            raise ScenarioError(
                f"{name}: predictor {predictor!r} is neither 'scripted' nor "
                "'checkpoints'"
            )
        perception = policy = None
        if predictor == "checkpoints":
            perception = str(_require(scenario, "perception", "scenario"))
            policy = str(_require(scenario, "policy", "scenario"))
        return cls(
            name=name,
            description=str(_require(scenario, "description", "scenario")).strip(),
            predictor=predictor,
            perception=perception,
            policy=policy,
            seed=int(_require(scenario, "seed", "scenario")),
            seconds=float(_require(scenario, "seconds", "scenario")),
            publish_to_ros=bool(scenario.get("publish_to_ros", False)),
            video=_video(raw.get("video"), output / f"{name}.mp4"),
        )


def _video(raw: Any, path: Path) -> VideoSettings | None:
    """Read the recording settings, or None when the scenario records nothing."""
    if raw is None:
        return None
    lookat = [float(value) for value in _require(raw, "lookat_meters", "video")]
    if len(lookat) != 3:
        raise ScenarioError("video.lookat_meters needs three numbers")
    return VideoSettings(
        path=path,
        width=int(_require(raw, "width", "video")),
        height=int(_require(raw, "height", "video")),
        frames_per_second=int(_require(raw, "frames_per_second", "video")),
        interval_seconds=float(_require(raw, "interval_seconds", "video")),
        azimuth=float(_require(raw, "azimuth_degrees", "video")),
        elevation=float(_require(raw, "elevation_degrees", "video")),
        distance=float(_require(raw, "distance_meters", "video")),
        lookat=np.asarray((lookat[0], lookat[1], lookat[2]), dtype=np.float64),
    )


def _require(mapping: Any, key: str, path: str = "") -> Any:
    """Read a key, failing with its location when it is absent.

    Args:
        mapping: The mapping to read from.
        key: The key required.
        path: Name of the parent, used in the error message.

    Returns:
        The value.

    Raises:
        ScenarioError: If the key is absent.
    """
    if not isinstance(mapping, dict) or key not in mapping:
        where = f"{path}.{key}" if path else key
        raise ScenarioError(f"required scenario key {where!r} is missing")
    return mapping[key]
