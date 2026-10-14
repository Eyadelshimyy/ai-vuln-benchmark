"""Governed adapters for reference-free RAG scoring requests.

This module deliberately owns validation and provenance marshalling only. It
reuses the canonical :class:`ScoringRequest` contract and performs no retrieval,
LLM inference, metric arithmetic, thresholding, or truth adjudication.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ._contract_safety import artifact_digest, enum_value, freeze_metadata
from ._validation import (
    CanonicalContract,
    assessment_error,
    descriptive_identifier,
    fingerprint,
    thaw_json_value,
)
from .assessment import AssessmentSpec
from .authorization import build_scoring_request
from .execution import ObservationGranularity, ScoringRequest
from fast_mlsirm.rubric.models import RubricSpecification


class RAGEvidenceRegime(str, Enum):
    """Evidence available to one governed reference-free RAG evaluation."""

    PROMPT_ONLY = "prompt_only"
    RETRIEVED_CONTEXT = "retrieved_context"
    POOLED_CORPUS = "pooled_corpus"
    AUTHORITATIVE_CORPUS = "authoritative_corpus"
    HUMAN_ANCHOR = "human_anchor"


class RAGCandidateVisibility(str, Enum):
    """Whether scoring evidence can expose candidate-system identity/content."""

    CANDIDATE_BLIND = "candidate_blind"
    CANDIDATE_VISIBLE_CROSSFIT = "candidate_visible_crossfit"


class RAGPerturbationKind(str, Enum):
    """Finite controlled perturbations with preregistered construct semantics."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    EXPLICIT_CONTRADICTION = "explicit_contradiction"
    IRRELEVANT_CONTEXT = "irrelevant_context"
    REQUIRED_EVIDENCE_REMOVAL = "required_evidence_removal"
    CITATION_TARGET_SWAP = "citation_target_swap"
    SEMANTIC_QUERY_PARAPHRASE = "semantic_query_paraphrase"
    STYLE_ONLY_REWRITE = "style_only_rewrite"
    UNANSWERABLE_QUERY = "unanswerable_query"


class RAGPerturbationDirection(str, Enum):
    """Expected direction for one controlled perturbation construct."""

    DECREASE = "decrease"
    INVARIANT = "invariant"
    INCREASE = "increase"


class RAGPerturbationConstructBasis(str, Enum):
    """Whether cited primary sources directly align with the named construct."""

    LITERATURE_ALIGNED_CONSTRUCT = "literature_aligned_construct"
    MODEL_DESIGN_HYPOTHESIS = "model_design_hypothesis"


_PERTURBATION_EXPECTATIONS = MappingProxyType(
    {
        RAGPerturbationKind.UNSUPPORTED_CLAIM: (
            "grounded_generation",
            RAGPerturbationDirection.DECREASE,
            RAGPerturbationConstructBasis.LITERATURE_ALIGNED_CONSTRUCT,
        ),
        RAGPerturbationKind.EXPLICIT_CONTRADICTION: (
            "grounded_generation",
            RAGPerturbationDirection.DECREASE,
            RAGPerturbationConstructBasis.LITERATURE_ALIGNED_CONSTRUCT,
        ),
        RAGPerturbationKind.IRRELEVANT_CONTEXT: (
            "retrieval_relevance",
            RAGPerturbationDirection.DECREASE,
            RAGPerturbationConstructBasis.LITERATURE_ALIGNED_CONSTRUCT,
        ),
        RAGPerturbationKind.REQUIRED_EVIDENCE_REMOVAL: (
            "coverage_or_completeness_proxy",
            RAGPerturbationDirection.DECREASE,
            RAGPerturbationConstructBasis.MODEL_DESIGN_HYPOTHESIS,
        ),
        RAGPerturbationKind.CITATION_TARGET_SWAP: (
            "citation_attribution",
            RAGPerturbationDirection.DECREASE,
            RAGPerturbationConstructBasis.MODEL_DESIGN_HYPOTHESIS,
        ),
        RAGPerturbationKind.SEMANTIC_QUERY_PARAPHRASE: (
            "robustness",
            RAGPerturbationDirection.INVARIANT,
            RAGPerturbationConstructBasis.MODEL_DESIGN_HYPOTHESIS,
        ),
        RAGPerturbationKind.STYLE_ONLY_REWRITE: (
            "robustness",
            RAGPerturbationDirection.INVARIANT,
            RAGPerturbationConstructBasis.MODEL_DESIGN_HYPOTHESIS,
        ),
        RAGPerturbationKind.UNANSWERABLE_QUERY: (
            "answerability_and_abstention",
            RAGPerturbationDirection.INCREASE,
            RAGPerturbationConstructBasis.MODEL_DESIGN_HYPOTHESIS,
        ),
    }
)

_RESPONSE_PERTURBATIONS = frozenset(
    {
        RAGPerturbationKind.UNSUPPORTED_CLAIM,
        RAGPerturbationKind.EXPLICIT_CONTRADICTION,
        RAGPerturbationKind.CITATION_TARGET_SWAP,
        RAGPerturbationKind.STYLE_ONLY_REWRITE,
    }
)
_RETRIEVAL_PERTURBATIONS = frozenset(
    {
        RAGPerturbationKind.IRRELEVANT_CONTEXT,
        RAGPerturbationKind.REQUIRED_EVIDENCE_REMOVAL,
    }
)
_QUERY_PERTURBATIONS = frozenset(
    {
        RAGPerturbationKind.SEMANTIC_QUERY_PARAPHRASE,
        RAGPerturbationKind.UNANSWERABLE_QUERY,
    }
)
_COMMON_REQUEST_FIELDS = (
    "assessment_fingerprint",
    "rubric_id",
    "rubric_fingerprint",
    "construct_id",
    "response_format",
    "granularity",
    "respondent_id",
    "task_id",
    "task_family_id",
    "occasion_id",
    "criterion_ids",
    "allowed_scores",
    "schema_version",
)
_RAG_RELATION_METADATA_KEYS = frozenset(
    {
        "rag_evidence_regime",
        "rag_candidate_visibility",
        "rag_system_configuration_id",
        "rag_system_configuration_fingerprint",
        "rag_retrieval_run_fingerprint",
        "rag_query_revision_fingerprint",
    }
)
_RELATION_AXIS_METADATA_KEYS = frozenset(
    {"rag_retrieval_run_fingerprint", "rag_query_revision_fingerprint"}
)
_RAG_PERTURBATION_ANCHOR_TOKEN = object()


@dataclass(frozen=True)
class RAGPerturbationAnchor(CanonicalContract):
    """Verified identity and preregistered direction for one RAG perturbation.

    The factory binds two canonical governed scoring requests and validates the
    kind-specific changed axis before this source-free artifact is constructed.
    ``construct_basis`` distinguishes constructs aligned with the cited RAGAS /
    ARES dimensions from package-owned model-design hypotheses. Every expected
    direction remains a preregistered hypothesis rather than an observed effect.
    """

    anchor_id: str
    baseline_request_fingerprint: str
    perturbed_request_fingerprint: str
    perturbation_specification_fingerprint: str
    perturbation_run_fingerprint: str
    perturbation_kind: RAGPerturbationKind
    expected_construct: str = field(init=False)
    expected_direction: RAGPerturbationDirection = field(init=False)
    construct_basis: RAGPerturbationConstructBasis = field(init=False)
    _anchor_token: InitVar[object | None] = None

    def __post_init__(self, _anchor_token: object | None) -> None:
        """Reject direct construction and derive immutable perturbation semantics."""
        if _anchor_token is not _RAG_PERTURBATION_ANCHOR_TOKEN:
            raise assessment_error(
                "unverified_rag_perturbation_anchor",
                "$",
                "use build_rag_perturbation_anchor",
            )
        normalized_anchor_id = descriptive_identifier(
            self.anchor_id,
            "anchor_id",
            "$.anchor_id",
        )
        normalized_baseline = fingerprint(
            self.baseline_request_fingerprint,
            "baseline_request_fingerprint",
            "$.baseline_request_fingerprint",
        )
        normalized_perturbed = fingerprint(
            self.perturbed_request_fingerprint,
            "perturbed_request_fingerprint",
            "$.perturbed_request_fingerprint",
        )
        normalized_specification = fingerprint(
            self.perturbation_specification_fingerprint,
            "perturbation_specification_fingerprint",
            "$.perturbation_specification_fingerprint",
        )
        normalized_run = fingerprint(
            self.perturbation_run_fingerprint,
            "perturbation_run_fingerprint",
            "$.perturbation_run_fingerprint",
        )
        normalized_kind = enum_value(
            self.perturbation_kind,
            RAGPerturbationKind,
            "rag_perturbation_kind",
            "$.perturbation_kind",
        )
        if normalized_baseline == normalized_perturbed:
            raise assessment_error(
                "identical_rag_perturbation_requests",
                "$.perturbed_request_fingerprint",
                "baseline and perturbed request fingerprints must differ",
            )
        expected_construct, expected_direction, construct_basis = (
            _PERTURBATION_EXPECTATIONS[normalized_kind]
        )
        object.__setattr__(self, "anchor_id", normalized_anchor_id)
        object.__setattr__(
            self,
            "baseline_request_fingerprint",
            normalized_baseline,
        )
        object.__setattr__(
            self,
            "perturbed_request_fingerprint",
            normalized_perturbed,
        )
        object.__setattr__(
            self,
            "perturbation_specification_fingerprint",
            normalized_specification,
        )
        object.__setattr__(
            self,
            "perturbation_run_fingerprint",
            normalized_run,
        )
        object.__setattr__(self, "perturbation_kind", normalized_kind)
        object.__setattr__(self, "expected_construct", expected_construct)
        object.__setattr__(self, "expected_direction", expected_direction)
        object.__setattr__(self, "construct_basis", construct_basis)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical anchor content without the derived content digest."""
        return {
            "anchor_id": self.anchor_id,
            "baseline_request_fingerprint": self.baseline_request_fingerprint,
            "perturbed_request_fingerprint": self.perturbed_request_fingerprint,
            "perturbation_specification_fingerprint": (
                self.perturbation_specification_fingerprint
            ),
            "perturbation_run_fingerprint": self.perturbation_run_fingerprint,
            "perturbation_kind": self.perturbation_kind.value,
            "expected_construct": self.expected_construct,
            "expected_direction": self.expected_direction.value,
            "construct_basis": self.construct_basis.value,
        }

    @property
    def anchor_fingerprint(self) -> str:
        """Return SHA-256 over the complete immutable perturbation-anchor content."""
        return artifact_digest(self)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical anchor content plus its deterministic fingerprint."""
        return {
            **self._content_dict(),
            "anchor_fingerprint": self.anchor_fingerprint,
        }


def _canonical_rag_request(value: Any, name: str) -> ScoringRequest:
    """Return one canonical RAG request with complete managed provenance."""
    if not isinstance(value, ScoringRequest):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be a ScoringRequest",
        )
    metadata = thaw_json_value(value.metadata)
    if not isinstance(metadata, dict) or not (
        _RAG_RELATION_METADATA_KEYS.issubset(metadata)
    ):
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}.metadata",
            f"{name} must contain complete managed RAG provenance",
        )
    query_revision = fingerprint(
        metadata["rag_query_revision_fingerprint"],
        "rag_query_revision_fingerprint",
        f"$.{name}.metadata.rag_query_revision_fingerprint",
    )
    if query_revision != value.task_revision_fingerprint:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}.metadata.rag_query_revision_fingerprint",
            "RAG query revision must replay the shared task revision",
        )
    fingerprint(
        metadata["rag_system_configuration_fingerprint"],
        "rag_system_configuration_fingerprint",
        f"$.{name}.metadata.rag_system_configuration_fingerprint",
    )
    fingerprint(
        metadata["rag_retrieval_run_fingerprint"],
        "rag_retrieval_run_fingerprint",
        f"$.{name}.metadata.rag_retrieval_run_fingerprint",
    )
    descriptive_identifier(
        metadata["rag_system_configuration_id"],
        "rag_system_configuration_id",
        f"$.{name}.metadata.rag_system_configuration_id",
    )
    enum_value(
        metadata["rag_evidence_regime"],
        RAGEvidenceRegime,
        "rag_evidence_regime",
        f"$.{name}.metadata.rag_evidence_regime",
    )
    enum_value(
        metadata["rag_candidate_visibility"],
        RAGCandidateVisibility,
        "rag_candidate_visibility",
        f"$.{name}.metadata.rag_candidate_visibility",
    )
    return value


def _request_metadata(request: ScoringRequest) -> dict[str, Any]:
    """Return mutable JSON metadata from one immutable scoring request."""
    metadata = thaw_json_value(request.metadata)
    if not isinstance(metadata, dict):  # pragma: no cover - sealed request invariant
        raise assessment_error(
            "invalid_rag_perturbation_request",
            "$.metadata",
            "RAG request metadata must be a mapping",
        )
    return metadata


def _raise_relationship_error(message: str) -> None:
    """Raise one stable kind-specific relationship failure."""
    raise assessment_error(
        "invalid_rag_perturbation_relationship",
        "$.perturbed_request",
        message,
    )


def _validate_common_request_provenance(
    baseline: ScoringRequest,
    perturbed: ScoringRequest,
    baseline_metadata: Mapping[str, Any],
    perturbed_metadata: Mapping[str, Any],
) -> None:
    """Reject pairs that differ outside controlled RAG treatment axes."""
    if any(
        getattr(baseline, field_name) != getattr(perturbed, field_name)
        for field_name in _COMMON_REQUEST_FIELDS
    ):
        raise assessment_error(
            "unrelated_rag_perturbation_requests",
            "$.perturbed_request",
            "requests must share assessment, rubric, system run, task, "
            "and occasion provenance",
        )
    baseline_invariants = {
        key: value
        for key, value in baseline_metadata.items()
        if key not in _RELATION_AXIS_METADATA_KEYS
    }
    perturbed_invariants = {
        key: value
        for key, value in perturbed_metadata.items()
        if key not in _RELATION_AXIS_METADATA_KEYS
    }
    if baseline_invariants != perturbed_invariants:
        raise assessment_error(
            "unrelated_rag_perturbation_requests",
            "$.perturbed_request.metadata",
            "requests must share evidence, visibility, system, policy, "
            "and split provenance",
        )


def _validate_perturbation_relationship(
    baseline: ScoringRequest,
    perturbed: ScoringRequest,
    kind: RAGPerturbationKind,
) -> None:
    """Validate the exact governed request relation permitted for ``kind``."""
    baseline_metadata = _request_metadata(baseline)
    perturbed_metadata = _request_metadata(perturbed)
    _validate_common_request_provenance(
        baseline,
        perturbed,
        baseline_metadata,
        perturbed_metadata,
    )
    if baseline.request_id == perturbed.request_id:
        _raise_relationship_error(
            "baseline and perturbed request identifiers must differ"
        )

    same_query = (
        baseline.task_revision_fingerprint == perturbed.task_revision_fingerprint
    )
    same_retrieval = (
        baseline_metadata["rag_retrieval_run_fingerprint"]
        == perturbed_metadata["rag_retrieval_run_fingerprint"]
    )
    same_response = (
        baseline.response_content_fingerprint
        == perturbed.response_content_fingerprint
    )

    if kind in _RESPONSE_PERTURBATIONS:
        if not same_query or not same_retrieval or same_response:
            _raise_relationship_error(
                "response perturbations must change only the governed response artifact"
            )
        if baseline.response_id == perturbed.response_id:
            _raise_relationship_error(
                "a changed response artifact requires a distinct response identifier"
            )
        return

    if kind in _RETRIEVAL_PERTURBATIONS:
        if not same_query or same_retrieval or not same_response:
            _raise_relationship_error(
                "retrieval perturbations must change only retrieval provenance"
            )
        if (
            baseline.response_id != perturbed.response_id
            or baseline.response_character_count != perturbed.response_character_count
            or baseline.response_unit_count != perturbed.response_unit_count
        ):
            _raise_relationship_error(
                "retrieval perturbations must preserve the exact governed "
                "response artifact"
            )
        return

    if kind in _QUERY_PERTURBATIONS:
        if same_query:
            _raise_relationship_error(
                "query perturbations require a distinct governed query revision"
            )
        return

    raise AssertionError(f"unhandled RAG perturbation kind: {kind}")


def build_rag_perturbation_anchor(
    *,
    anchor_id: str,
    baseline_request: ScoringRequest,
    perturbed_request: ScoringRequest,
    perturbation_specification_fingerprint: str,
    perturbation_run_fingerprint: str,
    perturbation_kind: RAGPerturbationKind | str,
) -> RAGPerturbationAnchor:
    """Build one verified, source-free controlled RAG perturbation anchor.

    Both inputs must be canonical governed requests. The factory validates
    shared provenance and the kind-specific changed axis, then binds the pair
    to an externally governed perturbation specification and execution run. It
    stores only content-addressed identities, accepts no raw query, context,
    response, or source text, and performs no scoring or truth adjudication.
    """
    normalized_baseline = _canonical_rag_request(
        baseline_request,
        "baseline_request",
    )
    normalized_perturbed = _canonical_rag_request(
        perturbed_request,
        "perturbed_request",
    )
    normalized_specification = fingerprint(
        perturbation_specification_fingerprint,
        "perturbation_specification_fingerprint",
        "$.perturbation_specification_fingerprint",
    )
    normalized_run = fingerprint(
        perturbation_run_fingerprint,
        "perturbation_run_fingerprint",
        "$.perturbation_run_fingerprint",
    )
    normalized_kind = enum_value(
        perturbation_kind,
        RAGPerturbationKind,
        "rag_perturbation_kind",
        "$.perturbation_kind",
    )
    if (
        normalized_baseline.request_fingerprint
        == normalized_perturbed.request_fingerprint
    ):
        raise assessment_error(
            "identical_rag_perturbation_requests",
            "$.perturbed_request",
            "baseline and perturbed requests must differ",
        )
    _validate_perturbation_relationship(
        normalized_baseline,
        normalized_perturbed,
        normalized_kind,
    )
    return RAGPerturbationAnchor(
        anchor_id=anchor_id,
        baseline_request_fingerprint=normalized_baseline.request_fingerprint,
        perturbed_request_fingerprint=normalized_perturbed.request_fingerprint,
        perturbation_specification_fingerprint=normalized_specification,
        perturbation_run_fingerprint=normalized_run,
        perturbation_kind=normalized_kind,
        _anchor_token=_RAG_PERTURBATION_ANCHOR_TOKEN,
    )


_MANAGED_METADATA_KEYS = frozenset(
    {
        "rag_evidence_regime",
        "rag_candidate_visibility",
        "rag_system_configuration_id",
        "rag_system_configuration_fingerprint",
        "rag_retrieval_run_fingerprint",
        "rag_query_revision_fingerprint",
    }
)
_ALLOWED_CALLER_METADATA_KEYS = frozenset({"evaluation_split"})


def _rag_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    evidence_regime: RAGEvidenceRegime,
    candidate_visibility: RAGCandidateVisibility,
    system_configuration_id: str,
    system_configuration_fingerprint: str,
    retrieval_run_fingerprint: str,
    query_revision_fingerprint: str,
) -> dict[str, Any]:
    """Return allowlisted caller metadata plus package-managed RAG provenance."""
    raw_metadata: Mapping[str, Any] = {} if metadata is None else metadata
    if not isinstance(raw_metadata, Mapping):
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata must be a mapping",
        )
    if any(key in raw_metadata for key in _MANAGED_METADATA_KEYS):
        raise assessment_error(
            "reserved_rag_metadata",
            "$.metadata",
            "RAG provenance metadata is package-managed",
        )
    if any(key not in _ALLOWED_CALLER_METADATA_KEYS for key in raw_metadata):
        raise assessment_error(
            "unsupported_rag_metadata",
            "$.metadata",
            "metadata key is not allowed for RAG scoring requests",
        )
    frozen = freeze_metadata(raw_metadata)
    if not isinstance(frozen, Mapping):
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata must be a mapping",
        )
    output = thaw_json_value(frozen)
    if not isinstance(output, dict):
        raise assessment_error(
            "invalid_rag_metadata",
            "$.metadata",
            "metadata must be a mapping",
        )
    if "evaluation_split" in output:
        output["evaluation_split"] = descriptive_identifier(
            output["evaluation_split"],
            "evaluation_split",
            "$.metadata.evaluation_split",
        )
    output.update(
        {
            "rag_evidence_regime": evidence_regime.value,
            "rag_candidate_visibility": candidate_visibility.value,
            "rag_system_configuration_id": system_configuration_id,
            "rag_system_configuration_fingerprint": system_configuration_fingerprint,
            "rag_retrieval_run_fingerprint": retrieval_run_fingerprint,
            "rag_query_revision_fingerprint": query_revision_fingerprint,
        }
    )
    return output


def build_rag_scoring_request(
    *,
    request_id: str,
    assessment: AssessmentSpec,
    rubric: RubricSpecification,
    query_id: str,
    query_revision_fingerprint: str,
    query_testlet_id: str,
    evidence_regime: RAGEvidenceRegime | str,
    candidate_visibility: RAGCandidateVisibility | str,
    system_configuration_id: str,
    system_configuration_fingerprint: str,
    system_run_id: str,
    response_id: str,
    retrieval_run_fingerprint: str,
    response_content_fingerprint: str,
    occasion_id: str,
    criterion_ids: Iterable[str] = (),
    response_character_count: int,
    response_unit_count: int,
    metadata: Mapping[str, Any] | None = None,
) -> ScoringRequest:
    """Build one provenance-bound criterion-level RAG scoring request.

    The RAG-specific identities are projected onto the shared scoring axes:
    stochastic system run -> respondent, generated answer -> response, query ->
    task, exact query revision -> task revision, and query testlet -> task
    family. System configuration identity, evidence regime, candidate visibility,
    retrieval-run identity, and exact configuration/revision fingerprints are
    package-managed metadata and therefore participate in the canonical request
    fingerprint.

    Raw query, context, answer, and source text are intentionally absent from
    this interface. Caller metadata is intentionally allowlisted to prevent raw
    content from being smuggled into the canonical artifact. A reference-free
    request records its evidence regime; it does not imply world correctness,
    absolute retrieval recall, or deployment validity.
    """
    normalized_regime = enum_value(
        evidence_regime,
        RAGEvidenceRegime,
        "rag_evidence_regime",
        "$.evidence_regime",
    )
    normalized_visibility = enum_value(
        candidate_visibility,
        RAGCandidateVisibility,
        "rag_candidate_visibility",
        "$.candidate_visibility",
    )
    normalized_query_revision = fingerprint(
        query_revision_fingerprint,
        "query_revision_fingerprint",
        "$.query_revision_fingerprint",
    )
    normalized_system_fingerprint = fingerprint(
        system_configuration_fingerprint,
        "system_configuration_fingerprint",
        "$.system_configuration_fingerprint",
    )
    normalized_retrieval_fingerprint = fingerprint(
        retrieval_run_fingerprint,
        "retrieval_run_fingerprint",
        "$.retrieval_run_fingerprint",
    )
    normalized_system_configuration_id = descriptive_identifier(
        system_configuration_id,
        "system_configuration_id",
        "$.system_configuration_id",
    )

    rag_metadata = _rag_metadata(
        metadata=metadata,
        evidence_regime=normalized_regime,
        candidate_visibility=normalized_visibility,
        system_configuration_id=normalized_system_configuration_id,
        system_configuration_fingerprint=normalized_system_fingerprint,
        retrieval_run_fingerprint=normalized_retrieval_fingerprint,
        query_revision_fingerprint=normalized_query_revision,
    )

    return build_scoring_request(
        request_id=request_id,
        assessment=assessment,
        rubric=rubric,
        granularity=ObservationGranularity.CRITERION_LEVEL,
        respondent_id=system_run_id,
        response_id=response_id,
        task_id=query_id,
        task_revision_fingerprint=normalized_query_revision,
        task_family_id=query_testlet_id,
        occasion_id=occasion_id,
        criterion_ids=criterion_ids,
        response_content_fingerprint=response_content_fingerprint,
        response_character_count=response_character_count,
        response_unit_count=response_unit_count,
        metadata=rag_metadata,
    )


__all__ = [
    "RAGCandidateVisibility",
    "RAGEvidenceRegime",
    "RAGPerturbationAnchor",
    "RAGPerturbationConstructBasis",
    "RAGPerturbationDirection",
    "RAGPerturbationKind",
    "build_rag_perturbation_anchor",
    "build_rag_scoring_request",
]
