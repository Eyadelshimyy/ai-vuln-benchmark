"""Structural-conformance tests for the core protocols.

These verify that the concrete library classes shipped with
ellements satisfy the read-only Protocols documented at
:mod:`ellements.core.prompting.sources`. They also check
that :class:`ellements.core.LLMClient` satisfies
:class:`ellements.core.LLMClientProtocol` (which only checks
method *names* at runtime, but together with mypy --strict that
is enough to guarantee the structural contract).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ellements.core import (
    GuidelineLibrary,
    LLMClient,
    LLMClientProtocol,
    PersonaLibrary,
)
from ellements.core.prompting import GuidelineSource, PersonaSource


def test_persona_library_satisfies_persona_source(tmp_path: Path) -> None:
    library = PersonaLibrary(tmp_path)
    assert isinstance(library, PersonaSource)


def test_guideline_library_satisfies_guideline_source(tmp_path: Path) -> None:
    library = GuidelineLibrary(tmp_path)
    assert isinstance(library, GuidelineSource)


def test_llm_client_satisfies_llm_client_protocol() -> None:
    client = LLMClient(model="openai/gpt-4o-mini")
    assert isinstance(client, LLMClientProtocol)


def test_persona_source_protocol_rejects_unrelated_object() -> None:
    assert not isinstance(object(), PersonaSource)


def test_guideline_source_protocol_rejects_unrelated_object() -> None:
    assert not isinstance(object(), GuidelineSource)


def test_in_memory_dict_can_pose_as_persona_source() -> None:
    """Demonstrate a non-library implementation of :class:`PersonaSource`."""

    class _InMemoryPersonas:
        def __init__(self, data: dict[str, str], default: str | None = None) -> None:
            self._data = data
            self._default = default

        def list_ids(self) -> list[str]:
            return list(self._data)

        def __contains__(self, persona_id: object) -> bool:
            return persona_id in self._data

        def load(self, persona_id: str) -> str:
            return self._data[persona_id]

        def get_default_id(self) -> str | None:
            return self._default

    source = _InMemoryPersonas({"hero": "Be heroic."}, default="hero")
    assert isinstance(source, PersonaSource)
    assert source.list_ids() == ["hero"]
    assert "hero" in source
    assert source.load("hero") == "Be heroic."
    assert source.get_default_id() == "hero"


def test_in_memory_dict_can_pose_as_guideline_source() -> None:
    """Demonstrate a non-library implementation of :class:`GuidelineSource`."""

    class _InMemoryGuidelines:
        def __init__(self, data: dict[str, str]) -> None:
            self._data = data

        def list_ids(self) -> list[str]:
            return list(self._data)

        def __contains__(self, guideline_id: object) -> bool:
            return guideline_id in self._data

        def load(self, guideline_id: str) -> str:
            return self._data[guideline_id]

    source = _InMemoryGuidelines({"concise": "Be concise."})
    assert isinstance(source, GuidelineSource)
    assert source.list_ids() == ["concise"]
    assert "concise" in source
    assert source.load("concise") == "Be concise."


@pytest.mark.parametrize(
    "method_name",
    [
        "complete",
        "complete_structured",
        "complete_with_tools",
        "stream",
        "continue_conversation",
        "loglikelihood",
        "generate_image",
    ],
)
def test_llm_client_protocol_lists_known_methods(method_name: str) -> None:
    """Sanity check that the Protocol covers every method LLMClient exposes."""
    client = LLMClient(model="openai/gpt-4o-mini")
    assert hasattr(client, method_name)
    assert hasattr(LLMClientProtocol, method_name)
