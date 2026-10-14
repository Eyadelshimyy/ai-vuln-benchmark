"""Callback-safe preflight for governed RAG request metadata and replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import rag as _base
from ._validation import (
    AssessmentSpecError,
    _metadata_key,
    assessment_error,
    descriptive_identifier,
)

_ORIGINAL_RAG_METADATA = _base._rag_metadata
_ORIGINAL_CANONICAL_RAG_REQUEST = _base._canonical_rag_request


def _caller_metadata(value: Any) -> Any:
    """Snapshot the one supported caller field without alien callbacks."""
    raw_metadata = {} if value is None else value
    if not isinstance(raw_metadata, Mapping):
        return raw_metadata
    try:
        iterator = iter(raw_metadata)
    except Exception:
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata keys could not be inspected safely",
        ) from None

    authorized_keys: list[str] = []
    seen: set[str] = set()
    index = 0
    while True:
        try:
            raw_key = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata keys could not be materialized safely",
            ) from None
        safe_key = str.__str__(raw_key) if isinstance(raw_key, str) else raw_key
        try:
            key = _metadata_key(safe_key, f"$.metadata.keys[{index}]")
        except AssessmentSpecError as exc:
            if exc.code == "sensitive_metadata_field":
                raise assessment_error(
                    "unsupported_rag_metadata",
                    "$.metadata",
                    "metadata key is not allowed for RAG scoring requests",
                ) from None
            raise
        if key in seen:
            raise assessment_error(
                "duplicate_metadata_key",
                f"$.metadata.keys[{index}]",
                "metadata keys must be unique",
            )
        seen.add(key)
        if key in _base._MANAGED_METADATA_KEYS:
            raise assessment_error(
                "reserved_rag_metadata",
                "$.metadata",
                "RAG provenance metadata is package-managed",
            )
        if key not in _base._ALLOWED_CALLER_METADATA_KEYS:
            raise assessment_error(
                "unsupported_rag_metadata",
                "$.metadata",
                "metadata key is not allowed for RAG scoring requests",
            )
        authorized_keys.append(key)
        index += 1

    output: dict[str, str] = {}
    for key in authorized_keys:
        try:
            raw_value = raw_metadata[key]
        except Exception:
            raise assessment_error(
                "invalid_rag_metadata",
                "$.metadata",
                "metadata values could not be inspected safely",
            ) from None
        safe_value = (
            str.__str__(raw_value) if isinstance(raw_value, str) else raw_value
        )
        output[key] = descriptive_identifier(
            safe_value,
            "evaluation_split",
            "$.metadata.evaluation_split",
        )
    return output


def _rag_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    evidence_regime: _base.RAGEvidenceRegime,
    candidate_visibility: _base.RAGCandidateVisibility,
    system_configuration_id: str,
    system_configuration_fingerprint: str,
    retrieval_run_fingerprint: str,
    query_revision_fingerprint: str,
) -> dict[str, Any]:
    """Build RAG metadata after one callback-safe caller-field snapshot."""
    return _ORIGINAL_RAG_METADATA(
        metadata=_caller_metadata(metadata),
        evidence_regime=evidence_regime,
        candidate_visibility=candidate_visibility,
        system_configuration_id=system_configuration_id,
        system_configuration_fingerprint=system_configuration_fingerprint,
        retrieval_run_fingerprint=retrieval_run_fingerprint,
        query_revision_fingerprint=query_revision_fingerprint,
    )


def _canonical_rag_request(value: Any, name: str) -> _base.ScoringRequest:
    """Admit only an exact factory-sealed request before replaying its fields."""
    if type(value) is not _base.ScoringRequest:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be a ScoringRequest",
        )
    return _ORIGINAL_CANONICAL_RAG_REQUEST(value, name)


def install(module: Any) -> None:
    """Install callback-safe metadata and request-replay boundaries."""
    module._rag_metadata = _rag_metadata
    module._canonical_rag_request = _canonical_rag_request


__all__: list[str] = []
