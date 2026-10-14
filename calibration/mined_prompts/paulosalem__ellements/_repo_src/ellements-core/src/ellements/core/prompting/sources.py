"""Structural protocols for persona and guideline sources.

These Protocols describe the **read-only** surface a consumer needs in
order to look up personas/guidelines by ID. The concrete
:class:`ellements.core.PersonaLibrary` and
:class:`ellements.core.GuidelineLibrary` classes implement these
Protocols (structurally, no explicit inheritance), and they also offer
additional capabilities such as :meth:`add_from_path`.

Decoupling **consumers** of personas/guidelines (e.g.
:class:`ellements.core.PromptContext`, agent builders, application
runners) from the **file-backed** libraries enables:

- Test doubles backed by an in-memory ``dict``.
- Alternative backends (HTTP, database, S3, git) without touching
  callers.
- Composing multiple sources behind a single facade (e.g. project
  overrides → org defaults → built-ins).

Why not nominal inheritance? Because every consumer that needs a
persona/guideline by ID already speaks this exact protocol — there is
no advantage to forcing a base class, and Protocols allow third-party
libraries (or simple ``dict``-backed test fakes) to satisfy the API
without coupling to ellements internals.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PersonaSource(Protocol):
    """Read-only persona lookup by ID.

    Implementations may be file-system libraries, in-memory dicts,
    remote services, or anything else that can answer the four
    questions below. See :class:`ellements.core.PersonaLibrary` for the
    canonical file-backed implementation.
    """

    def list_ids(self) -> list[str]:
        """Return every persona ID known to this source.

        The order is implementation-defined but should be stable for
        callers that present the list to humans.
        """

    def __contains__(self, persona_id: object) -> bool:
        """Return ``True`` when *persona_id* identifies a known persona."""

    def load(self, persona_id: str) -> str:
        """Return the rendered persona text for *persona_id*.

        Raises:
            PersonaNotFoundError: When *persona_id* is unknown.
        """

    def get_default_id(self) -> str | None:
        """Return a recommended default persona ID, or ``None``.

        Implementations may return the first listed persona, a
        configured favourite, or ``None`` when no opinionated default
        exists.
        """


@runtime_checkable
class GuidelineSource(Protocol):
    """Read-only guideline lookup by ID.

    Implementations may be file-system libraries, in-memory dicts,
    remote services, or anything else that can answer the three
    questions below. See :class:`ellements.core.GuidelineLibrary` for
    the canonical file-backed implementation.
    """

    def list_ids(self) -> list[str]:
        """Return every guideline ID known to this source."""

    def __contains__(self, guideline_id: object) -> bool:
        """Return ``True`` when *guideline_id* identifies a known guideline."""

    def load(self, guideline_id: str) -> str:
        """Return the rendered guideline text for *guideline_id*.

        Raises:
            GuidelineNotFoundError: When *guideline_id* is unknown.
        """


__all__ = ["GuidelineSource", "PersonaSource"]
