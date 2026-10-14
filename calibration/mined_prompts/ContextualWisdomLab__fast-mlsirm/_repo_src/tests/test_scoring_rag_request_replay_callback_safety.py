"""Callback-safety regressions for governed RAG request replay boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError, ScoringRequest
from fast_mlsirm.scoring.rag import (
    build_rag_perturbation_anchor,
    build_rag_scoring_request,
)
from fast_mlsirm.scoring.rag_calibration import build_rag_facets_rating_records

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
assessment = _FIXTURES["assessment"]
rubric = _FIXTURES["rubric"]

_QUERY_FP = hashlib.sha256(b"rag-replay-query").hexdigest()
_SYSTEM_FP = hashlib.sha256(b"rag-replay-system").hexdigest()
_RETRIEVAL_FP = hashlib.sha256(b"rag-replay-retrieval").hexdigest()
_SECOND_RESPONSE_FP = hashlib.sha256(b"rag-replay-response-second").hexdigest()
_SPECIFICATION_FP = hashlib.sha256(b"rag-replay-specification").hexdigest()
_RUN_FP = hashlib.sha256(b"rag-replay-run").hexdigest()
_SECRET = "caller_subclass_callback_secret"


class _HostileScoringRequest(ScoringRequest):
    """Expose the risk of reading fields from an unsealed request subclass."""

    def __getattribute__(self, name: str):
        if name in {"metadata", "task_revision_fingerprint", "request_fingerprint"}:
            raise RuntimeError(_SECRET)
        return super().__getattribute__(name)


def _request(*, request_id: str, response_id: str, response_fingerprint: str) -> ScoringRequest:
    """Build one canonical RAG request for the non-hostile side of a replay."""
    return build_rag_scoring_request(
        request_id=request_id,
        assessment=assessment(),
        rubric=rubric(),
        query_id="refund_policy_query",
        query_revision_fingerprint=_QUERY_FP,
        query_testlet_id="evidence_review",
        evidence_regime="retrieved_context",
        candidate_visibility="candidate_blind",
        system_configuration_id="retrieval_stack_a",
        system_configuration_fingerprint=_SYSTEM_FP,
        system_run_id="retrieval_stack_a_run_001",
        response_id=response_id,
        retrieval_run_fingerprint=_RETRIEVAL_FP,
        response_content_fingerprint=response_fingerprint,
        occasion_id="evaluation_wave_001",
        criterion_ids=("grounded_generation",),
        response_character_count=120,
        response_unit_count=3,
        metadata={"evaluation_split": "offline_holdout"},
    )


def _hostile_request() -> ScoringRequest:
    """Return an uninitialized subclass whose field access executes caller code."""
    return object.__new__(_HostileScoringRequest)


def test_perturbation_replay_rejects_request_subclass_before_field_callbacks() -> None:
    """Perturbation provenance accepts only exact factory-sealed requests."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_perturbation_anchor(
            anchor_id="callback_safety_anchor",
            baseline_request=_hostile_request(),
            perturbed_request=_request(
                request_id="rag_replay_perturbed_request",
                response_id="generated_response_002",
                response_fingerprint=_SECOND_RESPONSE_FP,
            ),
            perturbation_specification_fingerprint=_SPECIFICATION_FP,
            perturbation_run_fingerprint=_RUN_FP,
            perturbation_kind="unsupported_claim",
        )

    assert caught.value.code == "invalid_baseline_request"
    assert _SECRET not in str(caught.value)


def test_facets_replay_rejects_request_subclass_before_field_callbacks() -> None:
    """Calibration projection rejects an unsealed request before other inputs."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_facets_rating_records(
            request=_hostile_request(),
            result=object(),  # type: ignore[arg-type]
            engine=object(),  # type: ignore[arg-type]
        )

    assert caught.value.code == "invalid_rag_scoring_request"
    assert _SECRET not in str(caught.value)
