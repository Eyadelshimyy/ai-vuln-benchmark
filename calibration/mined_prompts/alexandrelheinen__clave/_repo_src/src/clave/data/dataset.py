"""Writing and reading datasets.

A dataset is a claim about what a model saw, so it resolves by digest. Two
training runs can then be shown to have seen the same bytes rather than assumed
to have.

Nothing is committed. Datasets are produced by a command and live outside the
repository, the same way corpora do.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from clave.corpus.artifacts import digest_of
from clave.data.composition import PartComposition, compose_parts
from clave.data.examples import Example, ObjectLabel, Rollout
from clave.errors import ClaveError

DESCRIPTION_FILE = "dataset.json"
FORMAT_LEGACY = 1
FORMAT_CORPUS = 2


class DatasetError(ClaveError):
    """A dataset is malformed, or its contents do not match its digest."""


@dataclass(frozen=True)
class DatasetFile:
    """One archive inside a dataset, and the condition it was recorded under.

    Attributes:
        name: File name inside the dataset directory.
        sha256: Digest of that file alone, distinct from the dataset digest.
        byte_count: Size in bytes.
        frame_count: Captured instants in the archive.
        seed: Seed the rollout ran under.
        belt_speed_meters_per_second: Fixed belt speed, when the rollout had one.
        spacing_meters: Fixed release gap, when the rollout had one.
        camera_ids: Detection cameras stored in the archive, in order.
    """

    name: str
    sha256: str
    byte_count: int
    frame_count: int
    seed: int
    belt_speed_meters_per_second: float | None = None
    spacing_meters: float | None = None
    camera_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetDescription:
    """Everything needed to identify a dataset and know what it holds.

    Attributes:
        digest: SHA-256 over the rollout archives, in a fixed order.
        seed: Seed the dataset was recorded under.
        config_digest: Digest of the world configuration used.
        example_count: Total captured frames.
        parts: Split part name to the rollout identities it holds.
        composition: Part name to its measured composition.
        format_version: 1 is a single camera and a split. 2 is a corpus role
            with every detection camera and the instance map.
        role: `train` or `validation` for a corpus dataset. None for a split
            dataset produced by `record-dataset`.
        campaign_id: Digest that pairs a train corpus with its validation
            corpus. None when the dataset is not part of a campaign.
        files: One record per archive.
    """

    digest: str
    seed: int
    config_digest: str
    example_count: int
    parts: dict[str, list[str]]
    composition: dict[str, dict[str, object]]
    format_version: int = FORMAT_LEGACY
    role: str | None = None
    campaign_id: str | None = None
    files: tuple[DatasetFile, ...] = field(default_factory=tuple)


def _archive_digest(paths: list[Path]) -> str:
    """Digest a set of archives in a fixed order."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write(
    root: Path,
    rollouts: tuple[Rollout, ...],
    parts: dict[str, tuple[str, ...]],
    seed: int,
    config_digest: str,
    *,
    role: str | None = None,
    campaign_id: str | None = None,
) -> DatasetDescription:
    """Write a dataset and describe it.

    Args:
        root: Directory to write into; created if absent.
        rollouts: The recorded rollouts.
        parts: Split part name to rollout identities.
        seed: Seed the recording ran under.
        config_digest: Digest of the world configuration used.
        role: `train` or `validation` when this dataset is one half of a
            corpus. None keeps the single-camera archive `record-dataset` writes.
        campaign_id: Digest pairing the two halves. Required when `role` is set.

    Returns:
        The description, which is also written as JSON beside the archives.

    Raises:
        DatasetError: If a corpus rollout has no camera captures.
    """
    root.mkdir(parents=True, exist_ok=True)
    corpus = role is not None
    if corpus and campaign_id is None:
        raise DatasetError("a corpus dataset needs a campaign id")
    written: list[Path] = []
    for rollout in rollouts:
        path = root / f"{rollout.rollout_id}.npz"
        if corpus:
            _write_corpus_archive(path, rollout)
        else:
            _write_legacy_archive(path, rollout)
        written.append(path)
    return _describe(
        root, rollouts, parts, seed, config_digest, role, campaign_id, written
    )


def _describe(
    root: Path,
    rollouts: tuple[Rollout, ...],
    parts: dict[str, tuple[str, ...]],
    seed: int,
    config_digest: str,
    role: str | None,
    campaign_id: str | None,
    written: list[Path],
) -> DatasetDescription:
    """Write dataset.json for archives that are already on disk."""
    compositions = compose_parts(rollouts, parts)
    by_id = {rollout.rollout_id: rollout for rollout in rollouts}
    description = DatasetDescription(
        digest=_archive_digest(written),
        seed=seed,
        config_digest=config_digest,
        example_count=sum(len(rollout.examples) for rollout in rollouts),
        parts={name: list(members) for name, members in parts.items()},
        composition={
            name: _composition_as_dict(value) for name, value in compositions.items()
        },
        format_version=FORMAT_CORPUS if role is not None else FORMAT_LEGACY,
        role=role,
        campaign_id=campaign_id,
        files=tuple(_file_record(path, by_id[path.stem]) for path in written),
    )
    (root / DESCRIPTION_FILE).write_text(
        json.dumps(asdict(description), indent=2, sort_keys=True) + "\n"
    )
    return description


def _write_legacy_archive(path: Path, rollout: Rollout) -> None:
    """Write the single-camera archive `record-dataset` has always written."""
    np.savez_compressed(
        path,
        frames=np.stack([example.frame for example in rollout.examples]),
        times=np.array([example.simulated_time for example in rollout.examples]),
        arm_joints=np.array(
            [example.arm_joints for example in rollout.examples], dtype=np.float64
        ),
        labels=np.array(
            json.dumps(
                [
                    [label.as_dict() for label in example.labels]
                    for example in rollout.examples
                ]
            )
        ),
    )


def _write_corpus_archive(path: Path, rollout: Rollout) -> None:
    """Write every detection camera, the instance map, and the fixed factors."""
    if not rollout.examples or not rollout.examples[0].captures:
        raise DatasetError(
            f"{rollout.rollout_id} has no camera captures, so it cannot be a "
            "corpus archive"
        )
    camera_ids = [capture.camera_id for capture in rollout.examples[0].captures]
    frames = np.stack(
        [
            np.stack([capture.frame for capture in example.captures])
            for example in rollout.examples
        ]
    )
    instances = np.stack(
        [
            np.stack(
                [
                    (
                        capture.instance_ids
                        if capture.instance_ids is not None
                        else np.zeros(capture.frame.shape[:2], dtype=np.uint16)
                    )
                    for capture in example.captures
                ]
            )
            for example in rollout.examples
        ]
    )
    payload = {
        "version": FORMAT_CORPUS,
        "cameras": {
            camera_id: [
                [label.as_dict() for label in example.captures[index].labels]
                for example in rollout.examples
            ]
            for index, camera_id in enumerate(camera_ids)
        },
    }
    np.savez_compressed(
        path,
        frames=frames,
        times=np.array([example.simulated_time for example in rollout.examples]),
        arm_joints=np.array(
            [example.arm_joints for example in rollout.examples], dtype=np.float64
        ),
        labels=np.array(json.dumps(payload)),
        instance_ids=instances,
        camera_ids=np.array(camera_ids),
    )


def _file_record(path: Path, rollout: Rollout) -> DatasetFile:
    """Describe one archive after it has been written."""
    camera_ids: tuple[str, ...] = ()
    if rollout.examples and rollout.examples[0].captures:
        camera_ids = tuple(
            capture.camera_id for capture in rollout.examples[0].captures
        )
    return DatasetFile(
        name=path.name,
        sha256=digest_of(path),
        byte_count=path.stat().st_size,
        frame_count=len(rollout.examples),
        seed=rollout.seed,
        belt_speed_meters_per_second=rollout.belt_speed,
        spacing_meters=rollout.spacing_meters,
        camera_ids=camera_ids,
    )


def _composition_as_dict(value: PartComposition) -> dict[str, object]:
    """Render a composition as plain data."""
    return {
        "example_count": value.example_count,
        "class_counts": value.class_counts,
        "absent_classes": list(value.absent_classes),
    }


def read(root: Path) -> DatasetDescription:
    """Read a dataset description and verify its contents.

    Args:
        root: The dataset directory.

    Returns:
        The description.

    Raises:
        DatasetError: If the description is absent, or the archives no longer
            digest to what the description records. A silently changed dataset
            is the failure this whole module exists to prevent.
    """
    path = root / DESCRIPTION_FILE
    if not path.is_file():
        raise DatasetError(f"{path} is missing; this is not a dataset directory")
    raw = json.loads(path.read_text())
    observed = _archive_digest(list(root.glob("*.npz")))
    if observed != raw["digest"]:
        raise DatasetError(
            f"{root} digests to {observed}, but its description records "
            f"{raw['digest']}; the contents changed"
        )
    return _description_from_json(raw)


def _description_from_json(raw: dict[str, object]) -> DatasetDescription:
    """Read a description, filling the fields a format-1 file does not have."""
    files_raw = raw.get("files", ())
    files: list[DatasetFile] = []
    if isinstance(files_raw, list):
        for item in files_raw:
            if not isinstance(item, dict):
                continue
            cameras = item.get("camera_ids", ())
            files.append(
                DatasetFile(
                    name=str(item["name"]),
                    sha256=str(item["sha256"]),
                    byte_count=int(item["byte_count"]),
                    frame_count=int(item["frame_count"]),
                    seed=int(item["seed"]),
                    belt_speed_meters_per_second=_optional_float(
                        item.get("belt_speed_meters_per_second")
                    ),
                    spacing_meters=_optional_float(item.get("spacing_meters")),
                    camera_ids=tuple(str(camera) for camera in cameras),
                )
            )
    parts = raw["parts"]
    composition = raw["composition"]
    if not isinstance(parts, dict) or not isinstance(composition, dict):
        raise DatasetError("dataset description parts are not objects")
    role = raw.get("role")
    campaign = raw.get("campaign_id")
    return DatasetDescription(
        digest=str(raw["digest"]),
        seed=_as_int(raw["seed"], "seed"),
        config_digest=str(raw["config_digest"]),
        example_count=_as_int(raw["example_count"], "example_count"),
        parts={
            str(name): [str(item) for item in members]
            for name, members in parts.items()
        },
        composition={
            str(name): dict(value) if isinstance(value, dict) else {}
            for name, value in composition.items()
        },
        format_version=_as_int(
            raw.get("format_version", FORMAT_LEGACY), "format_version"
        ),
        role=str(role) if isinstance(role, str) else None,
        campaign_id=str(campaign) if isinstance(campaign, str) else None,
        files=tuple(files),
    )


def _as_int(value: object, field_name: str) -> int:
    """Read an integer field, refusing a bool or anything else."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(f"dataset description {field_name} is not an integer")
    return value


def _optional_float(value: object) -> float | None:
    """Read a JSON number, treating null and absence as missing."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError("dataset description expected a number")
    return float(value)


def load_split(root: Path, part: str, camera: str | None = None) -> tuple[Rollout, ...]:
    """Load the rollouts belonging to one split part.

    The archive format is this module's, so reading it is too. Training reads
    through here rather than opening archives itself.

    Args:
        root: The dataset directory.
        part: Split part name, such as `train`.
        camera: Detection camera to expose as `Example.frame` on a corpus
            archive. None uses the first camera stored. Ignored on a format-1
            archive, which has one camera.

    Returns:
        The rollouts in that part, with frames and labels restored.

    Raises:
        DatasetError: If the dataset does not verify, the part is unknown, or
            the named camera was not stored. Verification happens first, so
            training can never read a dataset whose contents changed since it
            was described.
    """
    description = read(root)
    if part not in description.parts:
        known = ", ".join(sorted(description.parts))
        raise DatasetError(f"unknown split part {part!r}; this dataset has {known}")

    file_by_name = {item.name: item for item in description.files}
    rollouts: list[Rollout] = []
    for rollout_id in description.parts[part]:
        archive = np.load(root / f"{rollout_id}.npz", allow_pickle=False)
        recorded = file_by_name.get(f"{rollout_id}.npz")
        if description.format_version >= FORMAT_CORPUS:
            examples = _examples_from_corpus(archive, description, camera)
        else:
            examples = _examples_from_legacy(archive, description)
        rollouts.append(
            Rollout(
                rollout_id=rollout_id,
                seed=recorded.seed if recorded is not None else description.seed,
                examples=examples,
                belt_speed=None
                if recorded is None
                else recorded.belt_speed_meters_per_second,
                spacing_meters=None if recorded is None else recorded.spacing_meters,
            )
        )
    return tuple(rollouts)


def iter_split(root: Path, part: str, camera: str | None = None) -> Iterator[Rollout]:
    """Yield one rollout at a time, then release its pixels.

    `load_split` keeps every frame. A full corpus does not fit beside a model
    on the machine this command runs on, so training and scoring walk the
    archives instead.

    Args:
        root: The dataset directory.
        part: Split part name, such as `train`.
        camera: Detection camera to expose as `Example.frame` on a corpus
            archive. None uses the first camera stored.

    Yields:
        One rollout, with a copy of the chosen camera and no other captures.

    Raises:
        DatasetError: If the dataset does not verify, the part is unknown, or
            the named camera was not stored.
    """
    description = read(root)
    # Hashing the archives allocates large temporary buffers. glibc keeps that
    # heap, and the frames copied below would sit on top of it.
    from clave.training.memory import release_freed_pages

    release_freed_pages()
    if part not in description.parts:
        known = ", ".join(sorted(description.parts))
        raise DatasetError(f"unknown split part {part!r}; this dataset has {known}")

    file_by_name = {item.name: item for item in description.files}
    for rollout_id in description.parts[part]:
        # Load in a function so the archive and the second camera die before
        # this generator suspends. A suspended frame would keep both.
        yield _load_rollout(root, description, file_by_name, rollout_id, camera)


def load_rollout(
    root: Path,
    description: DatasetDescription,
    rollout_id: str,
    camera: str | None = None,
) -> Rollout:
    """Load one archive and release the rest.

    Args:
        root: The dataset directory.
        description: The description already read for this directory.
        rollout_id: Archive name without the suffix.
        camera: Detection camera to expose as `Example.frame`. None uses the
            first camera stored.

    Returns:
        The rollout, with owned frames.
    """
    file_by_name = {item.name: item for item in description.files}
    return _load_rollout(root, description, file_by_name, rollout_id, camera)


def _load_rollout(
    root: Path,
    description: DatasetDescription,
    file_by_name: dict[str, DatasetFile],
    rollout_id: str,
    camera: str | None,
) -> Rollout:
    """Load one archive and return owned frames, releasing the rest."""
    archive = np.load(root / f"{rollout_id}.npz", allow_pickle=False)
    try:
        if description.format_version >= FORMAT_CORPUS:
            examples = _examples_from_corpus(archive, description, camera)
        else:
            examples = _examples_from_legacy(archive, description)
        owned = tuple(_owned_frame(item) for item in examples)
    finally:
        archive.close()
    recorded = file_by_name.get(f"{rollout_id}.npz")
    return Rollout(
        rollout_id=rollout_id,
        seed=recorded.seed if recorded is not None else description.seed,
        examples=owned,
        belt_speed=None if recorded is None else recorded.belt_speed_meters_per_second,
        spacing_meters=None if recorded is None else recorded.spacing_meters,
    )


def _owned_frame(example: Example) -> Example:
    """Copy the frame so it survives closing the archive it was read from."""
    return replace(example, frame=np.array(example.frame, copy=True), captures=())


def _joints_at(joints: object, index: int) -> tuple[float, ...]:
    """Read one example's arm joints, or none when the archive predates them."""
    if joints is None or len(joints) == 0:  # type: ignore[arg-type]
        return ()
    return tuple(float(value) for value in joints[index])  # type: ignore[index]


def _label_from_dict(item: dict[str, object]) -> ObjectLabel:
    """Restore one label, including the pose fields a format-1 file may omit."""
    bbox_raw = item.get("bbox")
    bbox: tuple[int, int, int, int] | None = None
    if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
        bbox = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))
    orientation = item.get("orientation")
    linear = item.get("linear_velocity")
    angular = item.get("angular_velocity")
    return ObjectLabel(
        object_id=_as_int(item["object_id"], "object_id"),
        material_class=str(item["material_class"]),
        channel=str(item["channel"]),
        position=np.asarray(item["position"], dtype=np.float64),
        in_reachable_window=bool(item["in_reachable_window"]),
        bbox=bbox,
        orientation=(
            (
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            )
            if isinstance(orientation, list) and len(orientation) == 4
            else None
        ),
        linear_velocity=(
            (float(linear[0]), float(linear[1]), float(linear[2]))
            if isinstance(linear, list) and len(linear) == 3
            else None
        ),
        angular_velocity=(
            (float(angular[0]), float(angular[1]), float(angular[2]))
            if isinstance(angular, list) and len(angular) == 3
            else None
        ),
        object_name=str(item.get("object_name") or ""),
    )


def _examples_from_legacy(
    archive: np.lib.npyio.NpzFile, description: DatasetDescription
) -> tuple[Example, ...]:
    """Restore a single-camera archive."""
    frames = archive["frames"]
    times = archive["times"]
    joints = archive["arm_joints"] if "arm_joints" in archive.files else None
    per_frame = json.loads(str(archive["labels"]))
    return tuple(
        Example(
            frame=frames[index],
            labels=tuple(
                _label_from_dict(item) for item in labels if isinstance(item, dict)
            ),
            simulated_time=float(times[index]),
            seed=description.seed,
            config_digest=description.config_digest,
            arm_joints=_joints_at(joints, index),
        )
        for index, labels in enumerate(per_frame)
    )


def _examples_from_corpus(
    archive: np.lib.npyio.NpzFile,
    description: DatasetDescription,
    camera: str | None,
) -> tuple[Example, ...]:
    """Restore every detection camera, exposing one of them as the frame."""
    from clave.data.examples import CameraCapture

    frames = archive["frames"]
    times = archive["times"]
    joints = archive["arm_joints"] if "arm_joints" in archive.files else None
    instances = archive["instance_ids"] if "instance_ids" in archive.files else None
    camera_ids = [str(item) for item in archive["camera_ids"]]
    if camera is None:
        chosen = 0
    elif camera not in camera_ids:
        raise DatasetError(
            f"camera {camera!r} is not in this archive; it stores {camera_ids}"
        )
    else:
        chosen = camera_ids.index(camera)
    payload = json.loads(str(archive["labels"]))
    by_camera = payload["cameras"]
    examples: list[Example] = []
    for index in range(frames.shape[0]):
        captures: list[CameraCapture] = []
        for camera_index, camera_id in enumerate(camera_ids):
            labels = tuple(
                _label_from_dict(item)
                for item in by_camera[camera_id][index]
                if isinstance(item, dict)
            )
            instance = instances[index, camera_index] if instances is not None else None
            captures.append(
                CameraCapture(
                    camera_id=camera_id,
                    frame=frames[index, camera_index],
                    labels=labels,
                    instance_ids=instance,
                )
            )
        primary = captures[chosen]
        examples.append(
            Example(
                frame=primary.frame,
                labels=primary.labels,
                simulated_time=float(times[index]),
                seed=description.seed,
                config_digest=description.config_digest,
                arm_joints=_joints_at(joints, index),
                captures=tuple(captures),
            )
        )
    return tuple(examples)
