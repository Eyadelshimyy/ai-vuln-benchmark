"""The resident-memory budget a proof-of-concept training run has to keep.

The ceiling and the input side live in configuration. Raising either one is
the follow-up in docs/requirements/training-memory.md, for a machine that can
hold a full-resolution batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clave.errors import ClaveError


class MemoryBudgetError(ClaveError):
    """The run cannot keep the configured resident-memory budget."""


@dataclass(frozen=True)
class MemoryBudget:
    """How large a training step is allowed to be.

    Attributes:
        resident_limit_bytes: Stop before the next batch once the process
            resident set is above this.
        input_side_pixels: Square side each training image is resized to.
    """

    resident_limit_bytes: int
    input_side_pixels: int

    @classmethod
    def load(cls, path: Path) -> MemoryBudget:
        """Read the budget.

        Args:
            path: The YAML file.

        Returns:
            The budget.

        Raises:
            MemoryBudgetError: If a key is absent.
        """
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "training_memory" not in raw:
            raise MemoryBudgetError(f"{path} has no 'training_memory' section")
        section: dict[str, Any] = raw["training_memory"]

        def need(key: str) -> int:
            if key not in section:
                raise MemoryBudgetError(
                    f"required configuration key 'training_memory.{key}' is missing"
                )
            return int(section[key])

        return cls(
            resident_limit_bytes=need("resident_limit_bytes"),
            input_side_pixels=need("input_side_pixels"),
        )


def parse_resident_bytes(status: str) -> int:
    """Read VmRSS from a `/proc/self/status` document.

    Args:
        status: The text of that file.

    Returns:
        The resident set, in bytes.

    Raises:
        MemoryBudgetError: If the line is absent.
    """
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise MemoryBudgetError("VmRSS is missing, so the budget cannot be checked")


def current_resident_bytes() -> int:
    """Resident set of this process, in bytes."""
    return parse_resident_bytes(Path("/proc/self/status").read_text())


def release_freed_pages() -> None:
    """Return freed heap pages to the operating system.

    CPython frees an array and glibc keeps the arena. A rollout's camera
    frames are that arena, and leaving them resident pushes the process over
    the proof-of-concept ceiling.
    """
    import ctypes

    ctypes.CDLL("libc.so.6").malloc_trim(0)


def require_within_budget(limit_bytes: int) -> int:
    """Stop the run when the resident set is over the ceiling.

    Args:
        limit_bytes: The configured ceiling.

    Returns:
        The resident set, in bytes, when it is inside the ceiling.

    Raises:
        MemoryBudgetError: If the resident set is over the ceiling.
    """
    resident = current_resident_bytes()
    if resident > limit_bytes:
        raise MemoryBudgetError(
            f"resident memory is {resident} bytes, over the {limit_bytes} byte budget"
        )
    return resident
