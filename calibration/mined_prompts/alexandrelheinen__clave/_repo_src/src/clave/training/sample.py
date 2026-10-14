"""Which frames a training run keeps.

Consecutive captures show the same objects. The pick is a list of indexes,
one stride apart while an object crosses the camera, drawn once and stored.
A later resume loads that list. It does not draw again.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clave.data.dataset import DatasetDescription
from clave.errors import ClaveError


class SampleError(ClaveError):
    """A frame sample cannot be drawn, or a checkpoint names a different one."""


@dataclass(frozen=True)
class FrameSample:
    """The indexes one training run is allowed to see.

    Attributes:
        samples_per_crossing: Looks kept while an object crosses the footprint.
        dataset_digest: Corpus the indexes were drawn from.
        along_travel_meters: Footprint used for the crossing count.
        capture_interval_seconds: Seconds between stored frames. Zero when no
            rollout recorded a belt speed and the interval was not needed.
        seed: Seed the phases were drawn from.
        picks: Rollout id to the frame indexes kept, in capture order.
    """

    samples_per_crossing: int
    dataset_digest: str
    along_travel_meters: float
    capture_interval_seconds: float
    seed: int
    picks: dict[str, tuple[int, ...]]

    @property
    def frame_count(self) -> int:
        """How many frames the pick keeps."""
        return sum(len(indexes) for indexes in self.picks.values())

    def payload(self) -> dict[str, object]:
        """A JSON-ready mapping. The digest is not part of it."""
        return {
            "along_travel_meters": self.along_travel_meters,
            "capture_interval_seconds": self.capture_interval_seconds,
            "dataset_digest": self.dataset_digest,
            "picks": {key: list(indexes) for key, indexes in self.picks.items()},
            "samples_per_crossing": self.samples_per_crossing,
            "seed": self.seed,
        }

    @property
    def digest(self) -> str:
        """Digest of the pick, so two different lists cannot share a name."""
        encoded = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def write(self, path: Path) -> None:
        """Write the manifest beside the checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"digest": self.digest, "sample": self.payload()}
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_payload(cls, raw: dict[str, object]) -> FrameSample:
        """Read a pick stored in a checkpoint or a manifest.

        Args:
            raw: The sample mapping.

        Returns:
            The pick.

        Raises:
            SampleError: If a field is missing or the indexes are not lists.
        """
        picks_raw = raw.get("picks")
        if not isinstance(picks_raw, dict):
            raise SampleError("stored frame sample has no picks")
        picks: dict[str, tuple[int, ...]] = {}
        for key, indexes in picks_raw.items():
            if not isinstance(indexes, list):
                raise SampleError(f"stored picks for {key!r} are not a list")
            picks[str(key)] = tuple(int(index) for index in indexes)
        try:
            return cls(
                samples_per_crossing=_as_int(raw["samples_per_crossing"]),
                dataset_digest=str(raw["dataset_digest"]),
                along_travel_meters=_as_float(raw["along_travel_meters"]),
                capture_interval_seconds=_as_float(raw["capture_interval_seconds"]),
                seed=_as_int(raw["seed"]),
                picks=picks,
            )
        except KeyError as err:
            raise SampleError(f"stored frame sample is missing {err}") from err


def _as_int(value: object) -> int:
    """Read an integer field, refusing a bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SampleError(f"stored frame sample has {value!r} where an integer belongs")
    return value


def _as_float(value: object) -> float:
    """Read a real field, refusing a bool."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SampleError(f"stored frame sample has {value!r} where a number belongs")
    return float(value)


def crossing_stride(frames_in_crossing: float, samples_per_crossing: int) -> int:
    """Frames between kept looks.

    Args:
        frames_in_crossing: How many captures an object spends in the footprint.
        samples_per_crossing: Looks to keep over that crossing.

    Returns:
        The stride. One means every frame is kept.

    Raises:
        SampleError: If `samples_per_crossing` is below one.
    """
    if samples_per_crossing < 1:
        raise SampleError(
            f"samples_per_crossing is {samples_per_crossing}, and a crossing "
            f"needs at least one look"
        )
    stepped = math.floor(frames_in_crossing / samples_per_crossing)
    if stepped < 1:
        return 1
    return stepped


def kept_indexes(
    frame_count: int,
    *,
    belt_speed: float | None,
    along_travel_meters: float,
    capture_interval_seconds: float,
    samples_per_crossing: int,
    phase: int,
) -> tuple[int, ...]:
    """Indexes kept in one rollout.

    Args:
        frame_count: Frames the archive holds.
        belt_speed: Meters per second, or None when the rollout did not record one.
        along_travel_meters: Camera footprint along the belt.
        capture_interval_seconds: Seconds between captures.
        samples_per_crossing: Looks per crossing.
        phase: First kept index. Ignored when every frame is kept.

    Returns:
        Increasing frame indexes.
    """
    if (
        belt_speed is None
        or belt_speed <= 0.0
        or capture_interval_seconds <= 0.0
        or along_travel_meters <= 0.0
    ):
        return tuple(range(frame_count))
    frames_in_crossing = along_travel_meters / (belt_speed * capture_interval_seconds)
    stride = crossing_stride(frames_in_crossing, samples_per_crossing)
    if stride <= 1:
        return tuple(range(frame_count))
    start = phase % stride
    return tuple(range(start, frame_count, stride))


def build_frame_sample(
    description: DatasetDescription,
    *,
    along_travel_meters: float,
    capture_interval_seconds: float,
    samples_per_crossing: int,
    seed: int,
    part: str = "train",
    dataset: Path | None = None,
) -> FrameSample:
    """Draw one phase per rollout and keep the resulting indexes.

    Args:
        description: The training dataset.
        along_travel_meters: Footprint of the camera the loader reads.
        capture_interval_seconds: Seconds between captures.
        samples_per_crossing: Looks per crossing.
        seed: Seed for the phases. The same seed redraws the same phases.
        part: Split part to walk, `train` or `validation`.
        dataset: Directory of the archives. Used when a legacy description
            names a rollout and does not list a file record.

    Returns:
        The pick, in archive order.

    Raises:
        SampleError: If the description has no such part.
    """
    members = description.parts.get(part)
    if members is None:
        raise SampleError(f"dataset has no {part} part to sample")
    by_name = {item.name: item for item in description.files}
    rng = np.random.Generator(np.random.PCG64(seed))
    picks: dict[str, tuple[int, ...]] = {}
    for rollout_id in members:
        recorded = by_name.get(f"{rollout_id}.npz")
        if recorded is None:
            if dataset is None:
                raise SampleError(
                    f"{part} part names {rollout_id!r}, which has no file record"
                )
            speed = None
            frames = _archive_frame_count(dataset, rollout_id)
        else:
            speed = recorded.belt_speed_meters_per_second
            frames = recorded.frame_count
        stride = 1
        if speed is not None and speed > 0.0 and capture_interval_seconds > 0.0:
            crossing = along_travel_meters / (speed * capture_interval_seconds)
            stride = crossing_stride(crossing, samples_per_crossing)
        phase = 0 if stride <= 1 else int(rng.integers(0, stride))
        picks[rollout_id] = kept_indexes(
            frames,
            belt_speed=speed,
            along_travel_meters=along_travel_meters,
            capture_interval_seconds=capture_interval_seconds,
            samples_per_crossing=samples_per_crossing,
            phase=phase,
        )
    return FrameSample(
        samples_per_crossing=samples_per_crossing,
        dataset_digest=description.digest,
        along_travel_meters=along_travel_meters,
        capture_interval_seconds=capture_interval_seconds,
        seed=seed,
        picks=picks,
    )


def _archive_frame_count(dataset: Path, rollout_id: str) -> int:
    """Frame count of an archive whose description has no file record."""
    archive = np.load(dataset / f"{rollout_id}.npz", allow_pickle=False)
    try:
        return int(archive["frames"].shape[0])
    finally:
        archive.close()


def capture_interval_seconds(dataset: Path, rollout_id: str) -> float:
    """Median seconds between captures in one archive.

    Args:
        dataset: Dataset directory.
        rollout_id: Archive stem.

    Returns:
        The interval in seconds.

    Raises:
        SampleError: If the archive has fewer than two timestamps.
    """
    path = dataset / f"{rollout_id}.npz"
    archive = np.load(path, allow_pickle=False)
    try:
        times = np.asarray(archive["times"], dtype=np.float64)
    finally:
        archive.close()
    deltas = np.diff(times)
    if deltas.size == 0:
        raise SampleError(f"{path} has fewer than two timestamps")
    interval = float(np.median(deltas))
    if interval <= 0.0:
        raise SampleError(f"{path} has capture interval {interval}")
    return interval


def resolve_stored_sample(
    stored: FrameSample | None,
    *,
    checkpoint: bool,
    samples_per_crossing: int,
) -> FrameSample | None:
    """Choose the checkpoint pick, or signal that a new one must be drawn.

    Args:
        stored: The pick loaded from a checkpoint, if it had one.
        checkpoint: Whether a checkpoint file exists.
        samples_per_crossing: The value in the training configuration.

    Returns:
        The stored pick, or None when the caller should draw one.

    Raises:
        SampleError: If the checkpoint exists and has no pick, or its `N`
            disagrees with the configuration.
    """
    if not checkpoint:
        return None
    if stored is None:
        raise SampleError(
            "checkpoint has no stored frame sample, so the pick that produced "
            "these weights is every frame. Remove the checkpoint to draw "
            "samples_per_crossing for a new run"
        )
    if stored.samples_per_crossing != samples_per_crossing:
        raise SampleError(
            f"checkpoint sampled {stored.samples_per_crossing} times per crossing; "
            f"this configuration asks for {samples_per_crossing}"
        )
    return stored
