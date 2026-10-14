"""Finding a dataset by directory or by digest.

Training and validation name a corpus by the digest the campaign printed.
A path still works, which is what `record-dataset` writes and what the
existing training files point at.
"""

from __future__ import annotations

import re
from pathlib import Path

from clave.corpus.artifacts import digest_of
from clave.data.dataset import DatasetDescription, DatasetError, read

_DIGEST = re.compile(r"[0-9a-f]{64}")


def resolve_dataset(root: Path, reference: Path, *, role: str) -> Path:
    """Resolve a dataset reference and check it is the half the caller asked for.

    Args:
        root: Repository root. A relative reference and the digest cache are
            resolved against it.
        reference: A directory, or a 64-character digest.
        role: `train` or `validation`. A corpus of the other role is refused.
            A format-1 dataset has no role and is accepted for `train` only.

    Returns:
        The local directory, verified.

    Raises:
        DatasetError: If the digest is not local and cannot be fetched, if the
            role does not match, or if a corpus was recorded against a different
            world configuration than the one in the tree.
    """
    directory = reference if reference.is_absolute() else root / reference
    name = reference.name
    if _DIGEST.fullmatch(name) and not directory.is_dir():
        directory = root / "datasets" / "by-digest" / name
        if not (directory / "dataset.json").is_file():
            _pull(name, directory)
    description = read(directory)
    _check_role(description, role)
    if description.role is not None:
        world = digest_of(root / "configs" / "world" / "sorting_line.yml")
        if description.config_digest != world:
            recorded = description.config_digest[:12]
            raise DatasetError(
                f"{directory} was recorded against world {recorded}, "
                f"and the tree is {world[:12]}"
            )
    return directory


def _check_role(description: DatasetDescription, role: str) -> None:
    """Refuse a corpus half the caller did not ask for."""
    if description.role is None:
        if role != "train":
            raise DatasetError(
                "this dataset has no validation role; record a corpus with "
                "`clave corpus`"
            )
        return
    if description.role != role:
        raise DatasetError(
            f"dataset {description.digest[:12]} is a {description.role} corpus, "
            f"and this command reads {role}"
        )


def _pull(digest: str, destination: Path) -> None:
    """Download a dataset that is not already on disk."""
    from clave.storage.config import StorageConfigError
    from clave.storage.r2 import R2Client
    from clave.storage.sync import pull_dataset

    try:
        from clave.storage.config import load_r2_config

        client = R2Client(load_r2_config())
    except StorageConfigError as exc:
        raise DatasetError(
            f"dataset {digest[:12]} is not on disk and remote storage is "
            f"unconfigured: {exc}"
        ) from exc
    pull_dataset(digest, destination, client)
