"""The candidate interface.

Every architecture the v0.1.2 shortlist advances is described here by one spec,
and loaded on demand. Description and loading are separate so the registry can
be read on a machine with none of the heavy libraries installed, which is what
keeps the quality gate independent of PyTorch.

Loading returns a result rather than raising. An absent optional dependency is a
normal outcome on this project's hardware, not an error condition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

MATERIAL_CLASS_COUNT = 11
"""Number of material classes fixed by docs/waste-taxonomy.md, M-01 to M-11.

A perception candidate's output head is sized to this. Changing the taxonomy
changes every trained head, which is why that document marks its identifiers
append-only.
"""


class Stage(Enum):
    """Which learned stage a candidate belongs to."""

    PERCEPTION = "perception"
    POLICY = "policy"


@dataclass(frozen=True)
class CandidateSpec:
    """A shortlisted architecture, described without being loaded.

    Attributes:
        name: Stable identifier used in reports.
        stage: Whether it perceives or decides.
        license: The license the upstream project ships under.
        source: Where the implementation comes from.
        requires: Import name of the dependency its adapter needs, or None when
            the candidate is implemented in this project.
    """

    name: str
    stage: Stage
    license: str
    source: str
    requires: str | None = None


@dataclass(frozen=True)
class LoadResult:
    """The outcome of asking a candidate to load.

    Exactly one of `model` and `unavailable_reason` is set.

    Attributes:
        spec: The candidate that was asked for.
        model: The loaded object, when loading succeeded.
        unavailable_reason: Why it did not load, when it did not.
    """

    spec: CandidateSpec
    model: Any | None = None
    unavailable_reason: str | None = None

    @property
    def loaded(self) -> bool:
        """Whether a model is present."""
        return self.model is not None


@dataclass(frozen=True)
class Candidate:
    """A spec paired with the callable that builds it.

    The builder is not invoked until :meth:`load` is called, so constructing a
    candidate imports nothing heavy.
    """

    spec: CandidateSpec
    build: Callable[[], Any]

    def load(self) -> LoadResult:
        """Build the model, reporting failure rather than raising.

        Returns:
            A result carrying either the model or the reason it is unavailable.
            A missing optional dependency names that dependency; any other
            failure is recorded with its exception type, and no substitute
            candidate is returned in either case.
        """
        try:
            return LoadResult(spec=self.spec, model=self.build())
        except ImportError as exc:
            missing = self.spec.requires or "an optional dependency"
            return LoadResult(
                spec=self.spec,
                unavailable_reason=f"{missing} is not installed ({exc})",
            )
        except Exception as exc:  # noqa: BLE001 - a failed build must not abort the sweep
            return LoadResult(
                spec=self.spec,
                unavailable_reason=f"{type(exc).__name__}: {exc}",
            )
