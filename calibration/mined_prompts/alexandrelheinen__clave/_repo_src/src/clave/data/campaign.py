"""A paired train and validation corpus.

One command records both halves from the same factor grid and disjoint seeds,
so a model trained on one digest is scored on the other. The grid lives in
configuration. This module turns that grid into assignments and writes the two
datasets the rest of the pipeline already knows how to read.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from clave.corpus.artifacts import digest_of
from clave.data.dataset import (
    DatasetDescription,
    _describe,
    _write_corpus_archive,
)
from clave.data.recorder import record
from clave.demo.video import StreamSettings, open_stream
from clave.errors import ClaveError
from clave.tracker.evidence import Role
from clave.tracker.sensors import load_sensors, of_role
from clave.world import config as world_config

LOGGER = logging.getLogger(__name__)


class CampaignError(ClaveError):
    """The corpus configuration does not describe a recordable campaign."""


@dataclass(frozen=True)
class Assignment:
    """One rollout the campaign will record.

    Attributes:
        role: `train` or `validation`.
        rollout_id: Identity inside that half.
        seed: Seed for placement, lighting and which object is drawn.
        belt_speed: Meters per second, held for the whole rollout.
        spacing_meters: Metres of belt between releases.
    """

    role: str
    rollout_id: str
    seed: int
    belt_speed: float
    spacing_meters: float


@dataclass(frozen=True)
class CampaignConfig:
    """The grid, the capture, and where the two halves are written.

    Attributes:
        seed: First seed. Train takes the next block, validation the block after
            that, so the two halves cannot share a rollout.
        seconds_per_rollout: Simulated seconds each rollout runs.
        capture_interval_seconds: Simulated seconds between stored frames.
        frame_height: Stored frame height in pixels.
        frame_width: Stored frame width in pixels.
        belt_speeds: Meters per second, one level of the grid.
        spacings: Metres between releases, the other level.
        train_rollouts_per_cell: How many seeds each train cell gets.
        validation_rollouts_per_cell: How many seeds each validation cell gets.
        out: Directory the two halves are written under.
        preview_seconds: Simulated seconds of the preview video.
        preview_belt_speed: Belt speed of that video.
        preview_spacing_meters: Release gap of that video.
        preview_frames_per_second: Playback rate.
        preview_out: Where the video is written, relative to the repository.
    """

    seed: int
    seconds_per_rollout: float
    capture_interval_seconds: float
    frame_height: int
    frame_width: int
    belt_speeds: tuple[float, ...]
    spacings: tuple[float, ...]
    train_rollouts_per_cell: int
    validation_rollouts_per_cell: int
    out: Path
    preview_seconds: float
    preview_belt_speed: float
    preview_spacing_meters: float
    preview_frames_per_second: int
    preview_out: Path

    def factors(self) -> dict[str, object]:
        """The grid, in an order a hash can depend on."""
        return {
            "seed": self.seed,
            "seconds_per_rollout": self.seconds_per_rollout,
            "capture_interval_seconds": self.capture_interval_seconds,
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "belt_speeds_meters_per_second": list(self.belt_speeds),
            "spacing_meters": list(self.spacings),
            "train_rollouts_per_cell": self.train_rollouts_per_cell,
            "validation_rollouts_per_cell": self.validation_rollouts_per_cell,
        }

    @classmethod
    def load(cls, path: Path) -> CampaignConfig:
        """Read a corpus configuration.

        Args:
            path: The YAML file.

        Returns:
            The configuration.

        Raises:
            CampaignError: If a required key is absent or a list is empty.
        """
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "corpus" not in raw:
            raise CampaignError(f"{path} has no 'corpus' section")
        section: dict[str, Any] = raw["corpus"]

        def need(mapping: dict[str, Any], key: str, where: str) -> Any:
            if key not in mapping:
                raise CampaignError(f"required configuration key {where!r} is missing")
            return mapping[key]

        speeds = _positive_list(
            need(
                section,
                "belt_speeds_meters_per_second",
                "corpus.belt_speeds_meters_per_second",
            )
        )
        spacings = _positive_list(
            need(section, "spacing_meters", "corpus.spacing_meters")
        )
        preview = section.get("preview")
        if not isinstance(preview, dict):
            raise CampaignError(f"{path} has no 'corpus.preview' section")
        return cls(
            seed=int(need(section, "seed", "corpus.seed")),
            seconds_per_rollout=float(
                need(section, "seconds_per_rollout", "corpus.seconds_per_rollout")
            ),
            capture_interval_seconds=float(
                need(
                    section,
                    "capture_interval_seconds",
                    "corpus.capture_interval_seconds",
                )
            ),
            frame_height=int(need(section, "frame_height", "corpus.frame_height")),
            frame_width=int(need(section, "frame_width", "corpus.frame_width")),
            belt_speeds=speeds,
            spacings=spacings,
            train_rollouts_per_cell=int(
                need(
                    section,
                    "train_rollouts_per_cell",
                    "corpus.train_rollouts_per_cell",
                )
            ),
            validation_rollouts_per_cell=int(
                need(
                    section,
                    "validation_rollouts_per_cell",
                    "corpus.validation_rollouts_per_cell",
                )
            ),
            out=Path(need(section, "out", "corpus.out")),
            preview_seconds=float(need(preview, "seconds", "corpus.preview.seconds")),
            preview_belt_speed=float(
                need(
                    preview,
                    "belt_speed_meters_per_second",
                    "corpus.preview.belt_speed_meters_per_second",
                )
            ),
            preview_spacing_meters=float(
                need(preview, "spacing_meters", "corpus.preview.spacing_meters")
            ),
            preview_frames_per_second=int(
                need(preview, "frames_per_second", "corpus.preview.frames_per_second")
            ),
            preview_out=Path(need(preview, "out", "corpus.preview.out")),
        )


def assignments(config: CampaignConfig) -> tuple[Assignment, ...]:
    """Expand the grid into rollouts with disjoint seeds.

    Train occupies the first block of seeds, validation the next. Both halves
    visit every speed and every spacing. Rollout ids restart in each half
    because the halves are separate directories.

    Args:
        config: The corpus configuration.

    Returns:
        One assignment per rollout, train first.
    """
    cells = [
        (speed, spacing) for speed in config.belt_speeds for spacing in config.spacings
    ]
    seed = config.seed
    planned: list[Assignment] = []
    for role, per_cell in (
        ("train", config.train_rollouts_per_cell),
        ("validation", config.validation_rollouts_per_cell),
    ):
        index = 0
        for speed, spacing in cells:
            for _ in range(per_cell):
                planned.append(
                    Assignment(
                        role=role,
                        rollout_id=f"rollout_{index:03d}",
                        seed=seed,
                        belt_speed=speed,
                        spacing_meters=spacing,
                    )
                )
                seed += 1
                index += 1
    return tuple(planned)


def campaign_id(world_digest: str, config: CampaignConfig) -> str:
    """Digest the world and the grid, so the same campaign is the same id.

    The output directory is not part of the digest. Two machines that record
    the same grid against the same world are the same campaign, wherever they
    write the files.
    """
    payload = json.dumps(
        {"world": world_digest, "factors": config.factors()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CampaignResult:
    """The two halves, and whether a preview video was written.

    Attributes:
        campaign_id: Digest pairing the halves.
        train: Description of the training corpus.
        validation: Description of the validation corpus.
        train_dir: Where the training corpus was written.
        validation_dir: Where the validation corpus was written.
        preview: Path of the preview video, or None when no encoder was present.
    """

    campaign_id: str
    train: DatasetDescription
    validation: DatasetDescription
    train_dir: Path
    validation_dir: Path
    preview: Path | None


def _release_freed_pages() -> None:
    """Return allocator arenas to the OS after a rollout's frames are dropped.

    CPython frees the arrays, and glibc keeps the pages. Across ninety rollouts
    that retained heap is larger than this machine.
    """
    import ctypes

    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)


def _without_pixels(rollout: Any) -> Any:
    """Drop frame buffers once the archive is on disk.

    Composition only reads labels. Holding every camera of every rollout until
    the half is finished does not fit in the memory this command runs on.
    """
    import numpy as np

    blank = np.zeros((1, 1, 3), dtype=np.uint8)
    examples = tuple(
        replace(example, frame=blank, captures=()) for example in rollout.examples
    )
    return replace(rollout, examples=examples)


def record_campaign(root: Path, config: CampaignConfig) -> CampaignResult:
    """Record both halves and a short preview of a dense stream.

    Args:
        root: Repository root.
        config: The corpus configuration. Paths in it are relative to `root`
            unless already absolute.

    Returns:
        The two descriptions and the preview path.
    """
    world = root / "configs" / "world" / "sorting_line.yml"
    world_digest = digest_of(world)
    identifier = campaign_id(world_digest, config)
    raw = world_config.load(world)
    camera_ids = tuple(
        sensor.source_id for sensor in of_role(load_sensors(raw), Role.DETECTION)
    )
    if not camera_ids:
        raise CampaignError("the world has no detection camera to record")

    out = config.out if config.out.is_absolute() else root / config.out
    descriptions: dict[str, DatasetDescription] = {}
    directories: dict[str, Path] = {}
    for role in ("train", "validation"):
        chosen = tuple(item for item in assignments(config) if item.role == role)
        directory = out / role
        directory.mkdir(parents=True, exist_ok=True)
        kept: list[Any] = []
        written: list[Path] = []
        for number, item in enumerate(chosen, start=1):
            LOGGER.info(
                "recording %s %s (%s/%s) seed %s at %.2f m/s, spacing %.2f m",
                role,
                item.rollout_id,
                number,
                len(chosen),
                item.seed,
                item.belt_speed,
                item.spacing_meters,
            )
            rollout = record(
                root=root,
                seed=item.seed,
                seconds=config.seconds_per_rollout,
                capture_interval_seconds=config.capture_interval_seconds,
                height=config.frame_height,
                width=config.frame_width,
                rollout_id=item.rollout_id,
                belt_speed=item.belt_speed,
                spacing_meters=item.spacing_meters,
                drive_arm=False,
                cameras=camera_ids,
            )
            path = directory / f"{rollout.rollout_id}.npz"
            _write_corpus_archive(path, rollout)
            written.append(path)
            kept.append(_without_pixels(rollout))
            del rollout
            gc.collect()
            _release_freed_pages()
        descriptions[role] = _describe(
            directory,
            tuple(kept),
            {role: tuple(item.rollout_id for item in chosen)},
            config.seed,
            world_digest,
            role,
            identifier,
            written,
        )
        directories[role] = directory

    preview = _preview(root, config, camera_ids[0])
    return CampaignResult(
        campaign_id=identifier,
        train=descriptions["train"],
        validation=descriptions["validation"],
        train_dir=directories["train"],
        validation_dir=directories["validation"],
        preview=preview,
    )


def _preview(root: Path, config: CampaignConfig, camera_id: str) -> Path | None:
    """Film one dense rollout from the upstream detection camera."""
    destination = (
        config.preview_out
        if config.preview_out.is_absolute()
        else root / config.preview_out
    )
    recorder = open_stream(
        StreamSettings(
            path=destination,
            width=config.frame_width,
            height=config.frame_height,
            frames_per_second=config.preview_frames_per_second,
        )
    )
    rollout = record(
        root=root,
        seed=config.seed,
        seconds=config.preview_seconds,
        capture_interval_seconds=config.capture_interval_seconds,
        height=config.frame_height,
        width=config.frame_width,
        rollout_id="preview",
        belt_speed=config.preview_belt_speed,
        spacing_meters=config.preview_spacing_meters,
        drive_arm=False,
        cameras=(camera_id,),
    )
    if recorder is None:
        return None
    for example in rollout.examples:
        recorder.write(example.frame)
    recorder.close()
    return destination


def _positive_list(value: object) -> tuple[float, ...]:
    """Read a non-empty list of positive numbers."""
    if not isinstance(value, list) or not value:
        raise CampaignError("a corpus factor list is empty")
    numbers = tuple(float(item) for item in value)
    if any(item <= 0.0 for item in numbers):
        raise CampaignError("a corpus factor is not positive")
    return numbers
