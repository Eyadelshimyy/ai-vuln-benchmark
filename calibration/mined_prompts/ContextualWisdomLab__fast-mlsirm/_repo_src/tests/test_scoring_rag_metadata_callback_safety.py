"""Callback-safety contracts for caller-provided RAG metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
from pathlib import Path
import runpy
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, ScoringRequest
from fast_mlsirm.scoring.rag import (
    RAGCandidateVisibility,
    RAGEvidenceRegime,
    build_rag_scoring_request,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]

QUERY_FP = hashlib.sha256(b"rag-callback-query").hexdigest()
SYSTEM_FP = hashlib.sha256(b"rag-callback-system").hexdigest()
RETRIEVAL_FP = hashlib.sha256(b"rag-callback-retrieval").hexdigest()
RESPONSE_FP = hashlib.sha256(b"rag-response-content").hexdigest()
_SECRET = "raw_sensitive_callback_payload"


class _HostileIdentifier(str):
    """Expose valid identifier content while rejecting ordinary callbacks."""

    def __str__(self) -> str:
        raise RuntimeError(_SECRET)

    def strip(self, chars: str | None = None) -> str:
        del chars
        raise RuntimeError(_SECRET)


class _ContainsTrap(Mapping[str, Any]):
    """Expose valid entries while rejecting alien membership callbacks."""

    def __init__(self) -> None:
        self._values = {"evaluation_split": "offline_holdout"}

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        del key
        raise RuntimeError(_SECRET)


class _IterationTrap(Mapping[str, Any]):
    """Raise sensitive caller text when metadata keys are enumerated."""

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError(_SECRET)

    def __len__(self) -> int:
        return 1


class _LateIterationTrap(Mapping[str, Any]):
    """Yield one valid key before key iteration fails."""

    def __getitem__(self, key: str) -> Any:
        del key
        return "offline_holdout"

    def __iter__(self) -> Iterator[str]:
        yield "evaluation_split"
        raise AssessmentSpecError(
            "caller_callback_failure",
            "$.metadata",
            _SECRET,
        )

    def __len__(self) -> int:
        return 2


class _ValueTrap(Mapping[str, Any]):
    """Expose one key whose caller-controlled value callback fails."""

    def __init__(self, key: str = "evaluation_split") -> None:
        self._key = key

    def __getitem__(self, key: str) -> Any:
        del key
        raise AssessmentSpecError(
            "caller_callback_failure",
            "$.metadata",
            _SECRET,
        )

    def __iter__(self) -> Iterator[str]:
        return iter((self._key,))

    def __len__(self) -> int:
        return 1


class _LateUnauthorizedKeyTrap(Mapping[str, Any]):
    """Expose an allowed key before a later unauthorized key."""

    def __init__(self) -> None:
        self.value_count = 0

    def __getitem__(self, key: str) -> Any:
        del key
        self.value_count += 1
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[str]:
        return iter(("evaluation_split", "unapproved_metadata"))

    def __len__(self) -> int:
        return 2


class _DuplicateKeyTrap(Mapping[str, Any]):
    """Yield the one supported caller key twice."""

    def __getitem__(self, key: str) -> Any:
        del key
        return "offline_holdout"

    def __iter__(self) -> Iterator[str]:
        return iter(("evaluation_split", "evaluation_split"))

    def __len__(self) -> int:
        return 2


class _NonStringKeyTrap(Mapping[Any, Any]):
    """Expose a non-string key whose value must remain unreachable."""

    def __getitem__(self, key: Any) -> Any:
        del key
        raise RuntimeError(_SECRET)

    def __iter__(self) -> Iterator[Any]:
        return iter((7,))

    def __len__(self) -> int:
        return 1


class _NestedTraversalTrap(Mapping[str, Any]):
    """Reveal whether an invalid identifier value is traversed as metadata."""

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError(_SECRET)

    def __len__(self) -> int:
        return 1


class _SinglePassMapping(Mapping[str, Any]):
    """Reject any second enumeration of the caller key authority."""

    def __init__(self) -> None:
        self.iteration_count = 0
        self.value_count = 0

    def __getitem__(self, key: str) -> Any:
        assert key == "evaluation_split"
        self.value_count += 1
        return "offline_holdout"

    def __iter__(self) -> Iterator[str]:
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise RuntimeError(_SECRET)
        return iter(("evaluation_split",))

    def __len__(self) -> int:
        return 1


def _request(metadata: Any) -> ScoringRequest:
    """Build one deterministic request around caller-controlled metadata."""
    return build_rag_scoring_request(
        request_id="rag_callback_request",
        assessment=assessment(),
        rubric=rubric(),
        query_id="refund_policy_query",
        query_revision_fingerprint=QUERY_FP,
        query_testlet_id="evidence_review",
        evidence_regime=RAGEvidenceRegime.RETRIEVED_CONTEXT,
        candidate_visibility=RAGCandidateVisibility.CANDIDATE_BLIND,
        system_configuration_id="retrieval_stack_a",
        system_configuration_fingerprint=SYSTEM_FP,
        system_run_id="retrieval_stack_a_run_001",
        response_id="generated_response_001",
        retrieval_run_fingerprint=RETRIEVAL_FP,
        response_content_fingerprint=RESPONSE_FP,
        occasion_id="evaluation_wave_001",
        criterion_ids=("grounded_generation",),
        response_character_count=120,
        response_unit_count=3,
        metadata=metadata,
    )


def _assert_error(metadata: Any, code: str) -> None:
    """Assert one stable non-reflective request-construction error."""
    with pytest.raises(AssessmentSpecError) as caught:
        _request(metadata)

    assert caught.value.code == code
    assert _SECRET not in str(caught.value)


def test_rag_metadata_does_not_invoke_alien_membership_callbacks() -> None:
    """Valid mappings are inspected through iteration, not ``__contains__``."""
    request = _request(_ContainsTrap())

    assert request.to_dict()["metadata"]["evaluation_split"] == "offline_holdout"


def test_rag_metadata_normalizes_hostile_string_subclasses() -> None:
    """Base-string content is accepted without invoking subclass callbacks."""
    request = _request(
        {_HostileIdentifier("evaluation_split"): _HostileIdentifier("offline_holdout")}
    )

    assert request.to_dict()["metadata"]["evaluation_split"] == "offline_holdout"


def test_rag_metadata_enumerates_keys_and_reads_values_once() -> None:
    """Authorization and value capture use one stable caller snapshot."""
    metadata = _SinglePassMapping()

    request = _request(metadata)

    assert request.to_dict()["metadata"]["evaluation_split"] == "offline_holdout"
    assert metadata.iteration_count == 1
    assert metadata.value_count == 1


def test_rag_metadata_iteration_failures_are_non_reflective() -> None:
    """First-step and late iterator failures remain package-owned."""
    _assert_error(_IterationTrap(), "invalid_rag_metadata")
    _assert_error(_LateIterationTrap(), "invalid_rag_metadata")


def test_rag_metadata_rejects_unknown_and_reserved_keys_before_values() -> None:
    """Disallowed key classes never authorize a caller value callback."""
    _assert_error(_ValueTrap("unapproved_metadata"), "unsupported_rag_metadata")
    _assert_error(_ValueTrap("rag_evidence_regime"), "reserved_rag_metadata")


def test_rag_metadata_authorizes_entire_key_stream_before_values() -> None:
    """A late unauthorized key fails before any earlier allowed value is read."""
    metadata = _LateUnauthorizedKeyTrap()

    _assert_error(metadata, "unsupported_rag_metadata")

    assert metadata.value_count == 0


def test_rag_metadata_value_failure_is_non_reflective() -> None:
    """An authorized value callback cannot forge trusted package evidence."""
    _assert_error(_ValueTrap(), "invalid_rag_metadata")


def test_rag_metadata_rejects_duplicate_and_non_string_keys() -> None:
    """Malformed key streams fail before any unsafe value access."""
    _assert_error(_DuplicateKeyTrap(), "duplicate_metadata_key")
    _assert_error(_NonStringKeyTrap(), "invalid_metadata_key")


def test_rag_metadata_rejects_non_mapping_input() -> None:
    """The established public mapping contract remains fail closed."""
    _assert_error(object(), "invalid_rag_metadata")


def test_rag_metadata_rejects_invalid_split_without_nested_traversal() -> None:
    """The only caller field is an identifier, not arbitrary nested JSON."""
    _assert_error(
        {"evaluation_split": _NestedTraversalTrap()},
        "invalid_evaluation_split",
    )


def test_rag_metadata_none_preserves_package_managed_provenance() -> None:
    """The optional caller path retains established managed RAG metadata."""
    payload = _request(None).to_dict()["metadata"]

    assert "evaluation_split" not in payload
    assert payload["rag_evidence_regime"] == "retrieved_context"
