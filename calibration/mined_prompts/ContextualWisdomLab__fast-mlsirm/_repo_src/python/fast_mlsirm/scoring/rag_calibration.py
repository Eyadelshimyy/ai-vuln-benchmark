"""Project governed RAG scoring executions into shared many-facet designs.

This adapter replays the package-managed RAG request provenance before handing
criterion observations to the existing scoring calibration contracts. Python
only validates identities and marshals records. All many-facet likelihood,
threshold, estimation, optimization, and uncertainty arithmetic remains owned
by the existing Rust-backed calibration path.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._contract_safety import bounded_values
from ._validation import assessment_error
from .calibration import (
    MAX_SCORING_FACETS_RATINGS,
    ScoringFacetsCalibrationBundle,
    ScoringFacetsRatingRecord,
    build_scoring_facets_calibration_bundle,
    build_scoring_facets_rating_records,
)
from .execution import EngineDescriptor, ScoringRequest, ScoringResult
from .rag import _canonical_rag_request

MAX_RAG_CALIBRATION_EXECUTIONS = MAX_SCORING_FACETS_RATINGS
"""Maximum governed RAG executions accepted by one facets bundle assembly."""


def build_rag_facets_rating_records(
    *,
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[ScoringFacetsRatingRecord, ...]:
    """Project one provenance-complete RAG execution into shared facets records.

    The RAG request must contain the package-managed evidence-regime, candidate
    visibility, system-configuration, retrieval-run, and query-revision
    provenance produced by :func:`fast_mlsirm.scoring.rag.build_rag_scoring_request`.
    The shared projector remains authoritative for exact request/result/engine
    binding, terminal observation states, criterion scales, and rater identity.

    Passing this boundary establishes replayable measurement provenance only.
    It does not establish factual truth, retrieval recall, model adequacy,
    validity, fairness, invariance, or a buyer-facing quality conclusion.
    """
    normalized_request = _canonical_rag_request(request, "rag_scoring_request")
    return build_scoring_facets_rating_records(
        request=normalized_request,
        result=result,
        engine=engine,
    )


def build_rag_facets_calibration_bundle(
    executions: Iterable[tuple[ScoringRequest, ScoringResult, EngineDescriptor]],
    *,
    require_connected: bool = True,
) -> ScoringFacetsCalibrationBundle:
    """Assemble governed RAG executions into the existing many-facet bundle.

    Each execution is an exact three-value tuple containing a canonical RAG
    scoring request, its bound scoring result, and the producing engine
    descriptor. Every request is replay-validated as RAG provenance before its
    observations are projected into the shared criterion-separated design.

    The returned value is the existing :class:`ScoringFacetsCalibrationBundle`;
    this module introduces no parallel psychometric result hierarchy and no new
    numerical estimator. System-run identity occupies the respondent axis,
    query revision occupies the task-revision axis, and engine fingerprint
    occupies the rater axis, preserving the hierarchy already encoded by the
    canonical RAG scoring request.
    """
    values = bounded_values(
        executions,
        "executions",
        minimum=1,
        maximum=MAX_RAG_CALIBRATION_EXECUTIONS,
    )

    def validated_records() -> Iterable[ScoringFacetsRatingRecord]:
        """Yield replay-validated RAG records for bounded shared assembly."""
        for index, execution in enumerate(values):
            if type(execution) is not tuple or len(execution) != 3:
                raise assessment_error(
                    "invalid_rag_calibration_execution",
                    f"$.executions[{index}]",
                    (
                        "each execution must be an exact three-value tuple of "
                        "request, result, and engine"
                    ),
                )
            request, result, engine = execution
            yield from build_rag_facets_rating_records(
                request=request,
                result=result,
                engine=engine,
            )

    return build_scoring_facets_calibration_bundle(
        validated_records(),
        require_connected=require_connected,
    )


__all__ = [
    "MAX_RAG_CALIBRATION_EXECUTIONS",
    "build_rag_facets_calibration_bundle",
    "build_rag_facets_rating_records",
]
