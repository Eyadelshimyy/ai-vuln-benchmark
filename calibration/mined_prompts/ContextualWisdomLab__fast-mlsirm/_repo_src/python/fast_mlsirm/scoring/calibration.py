"""Governed criterion-level many-facet calibration handoff.

The module projects already governed scoring requests, results, observations,
and engine descriptors into deterministic criterion-specific
``respondents x task revisions x raters`` tensors accepted by the existing
Rust-backed :func:`fast_mlsirm.fit_facets` estimator. Python validates,
preserves provenance, and marshals data only; it implements no psychometric
likelihood, gradient, quadrature, optimization, or uncertainty arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import Any

import numpy as np

from ._contract_safety import artifact_digest, bounded_values, enum_value
from ._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
    assessment_schema_version,
    descriptive_identifier,
    fingerprint,
    strict_boolean,
)
from .execution import (
    EngineDescriptor,
    ObservationGranularity,
    ObservationStatus,
    ScoreObservation,
    ScoringRequest,
    ScoringResult,
)

MAX_SCORING_FACETS_RATINGS = 100_000
# ``fit_facets`` currently consumes a dense persons-by-items-by-raters tensor.
# Structurally missing respondent-task-revision combinations therefore still
# occupy memory, so this guard intentionally bounds the actual allocation.
MAX_SCORING_FACETS_CELLS = 1_000_000
MAX_SCORING_SCORE_CATEGORIES = 64

_RATING_TOKEN = object()
_DESIGN_TOKEN = object()
_BUNDLE_TOKEN = object()


def _score_values(values: Iterable[int]) -> tuple[int, ...]:
    """Return a bounded sorted unique score scale of exact integers."""
    raw = bounded_values(
        values,
        "allowed_scores",
        minimum=2,
        maximum=MAX_SCORING_SCORE_CATEGORIES,
    )
    normalized: list[int] = []
    for index, value in enumerate(raw):
        if type(value) is not int:
            raise assessment_error(
                "invalid_allowed_scores",
                f"$.allowed_scores[{index}]",
                "allowed scores must be exact integers",
            )
        normalized.append(value)
    output = tuple(normalized)
    if output != tuple(sorted(set(output))):
        raise assessment_error(
            "invalid_allowed_scores",
            "$.allowed_scores",
            "allowed scores must be sorted and unique",
        )
    return output


def _score_for_status(
    value: int | None,
    status: ObservationStatus,
) -> int | None:
    """Validate the score/status relationship of one governed rating."""
    if status is ObservationStatus.SCORED:
        if type(value) is not int:
            raise assessment_error(
                "missing_rating_score",
                "$.score_category",
                "scored ratings require an exact integer score",
            )
        return value
    if value is not None:
        raise assessment_error(
            "unexpected_rating_score",
            "$.score_category",
            "terminal ratings cannot carry a score",
        )
    return None


@dataclass(frozen=True)
class ScoringFacetsRatingRecord(CanonicalContract):
    """One criterion rating bound to exact request, task, and engine artifacts."""

    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    request_fingerprint: str
    result_fingerprint: str
    observation_fingerprint: str
    respondent_id: str
    response_id: str
    response_content_fingerprint: str
    task_id: str
    task_revision_fingerprint: str
    task_family_id: str
    occasion_id: str
    criterion_id: str
    engine_id: str
    engine_family_id: str
    engine_fingerprint: str
    status: ObservationStatus
    score_category: int | None
    allowed_scores: tuple[int, ...]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _rating_token: InitVar[object | None] = None

    def __post_init__(self, _rating_token: object | None) -> None:
        """Reject direct construction and normalize package-owned fields."""
        if _rating_token is not _RATING_TOKEN:
            raise assessment_error(
                "unverified_facets_rating",
                "$",
                "use build_scoring_facets_rating_records",
            )
        for field_name in (
            "assessment_fingerprint",
            "rubric_fingerprint",
            "request_fingerprint",
            "result_fingerprint",
            "observation_fingerprint",
            "response_content_fingerprint",
            "task_revision_fingerprint",
            "engine_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                fingerprint(getattr(self, field_name), field_name),
            )
        for field_name in (
            "construct_id",
            "respondent_id",
            "response_id",
            "task_id",
            "task_family_id",
            "occasion_id",
            "criterion_id",
            "engine_id",
            "engine_family_id",
        ):
            object.__setattr__(
                self,
                field_name,
                descriptive_identifier(getattr(self, field_name), field_name),
            )
        normalized_status = enum_value(
            self.status,
            ObservationStatus,
            "observation_status",
            "$.status",
        )
        object.__setattr__(self, "status", normalized_status)
        normalized_score = _score_for_status(self.score_category, normalized_status)
        object.__setattr__(self, "score_category", normalized_score)
        normalized_scale = _score_values(self.allowed_scores)
        if (
            normalized_status is ObservationStatus.SCORED
            and normalized_score not in normalized_scale
        ):
            raise assessment_error(
                "rating_score_out_of_range",
                "$.score_category",
                "scored rating is outside the request score scale",
            )
        object.__setattr__(self, "allowed_scores", normalized_scale)
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical rating content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "request_fingerprint": self.request_fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
            "respondent_id": self.respondent_id,
            "response_id": self.response_id,
            "response_content_fingerprint": self.response_content_fingerprint,
            "task_id": self.task_id,
            "task_revision_fingerprint": self.task_revision_fingerprint,
            "task_family_id": self.task_family_id,
            "occasion_id": self.occasion_id,
            "criterion_id": self.criterion_id,
            "engine_id": self.engine_id,
            "engine_family_id": self.engine_family_id,
            "engine_fingerprint": self.engine_fingerprint,
            "status": self.status.value,
            "score_category": self.score_category,
            "allowed_scores": list(self.allowed_scores),
        }

    @property
    def rating_fingerprint(self) -> str:
        """Return SHA-256 over the current normalized rating content."""
        return artifact_digest(self)

    @property
    def rating_handle(self) -> str:
        """Return a descriptive 128-bit public rating handle."""
        return f"scoring_facets_rating_{self.rating_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content and deterministic rating identities."""
        return {
            **self._content_dict(),
            "rating_handle": self.rating_handle,
            "rating_fingerprint": self.rating_fingerprint,
        }


@dataclass(frozen=True)
class ScoringFacetsDesign(CanonicalContract):
    """One criterion-specific task-revision-indexed Rust facets design.

    ``respondent_id`` forms the person axis, exact task-revision fingerprints
    form the item axis, and the exact engine fingerprint forms the rater axis.
    Logical task and family identifiers remain aligned audit labels. Response
    identifiers and content fingerprints remain cell-level audit provenance.
    """

    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    occasion_id: str
    criterion_id: str
    category_values: tuple[int, ...]
    respondent_ids: tuple[str, ...]
    task_revision_fingerprints: tuple[str, ...]
    task_ids: tuple[str, ...]
    task_family_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    response_respondent_ids: tuple[str, ...]
    response_task_revision_fingerprints: tuple[str, ...]
    response_task_ids: tuple[str, ...]
    response_content_fingerprints: tuple[str, ...]
    rater_engine_ids: tuple[str, ...]
    rater_engine_family_ids: tuple[str, ...]
    rater_engine_fingerprints: tuple[str, ...]
    rating_records: tuple[ScoringFacetsRatingRecord, ...]
    respondent_task_connected: bool
    task_rater_connected: bool
    connected: bool
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated assembler."""
        if _design_token is not _DESIGN_TOKEN:
            raise assessment_error(
                "unverified_facets_design",
                "$",
                "use build_scoring_facets_calibration_bundle",
            )
        revisions = tuple(
            fingerprint(
                value,
                "task_revision_fingerprint",
                f"$.task_revision_fingerprints[{index}]",
            )
            for index, value in enumerate(self.task_revision_fingerprints)
        )
        task_ids = tuple(
            descriptive_identifier(
                value,
                "task_id",
                f"$.task_ids[{index}]",
            )
            for index, value in enumerate(self.task_ids)
        )
        task_family_ids = tuple(
            descriptive_identifier(
                value,
                "task_family_id",
                f"$.task_family_ids[{index}]",
            )
            for index, value in enumerate(self.task_family_ids)
        )
        response_revisions = tuple(
            fingerprint(
                value,
                "task_revision_fingerprint",
                f"$.response_task_revision_fingerprints[{index}]",
            )
            for index, value in enumerate(
                self.response_task_revision_fingerprints
            )
        )
        if len(set(revisions)) != len(revisions):
            raise assessment_error(
                "duplicate_task_revision_axis",
                "$.task_revision_fingerprints",
                "task revision axis must contain unique fingerprints",
            )
        if not (
            len(revisions) == len(task_ids) == len(task_family_ids)
        ):
            raise assessment_error(
                "invalid_task_revision_axis",
                "$.task_revision_fingerprints",
                "task revision, logical task, and family axes must align",
            )
        if not (
            len(self.response_ids)
            == len(self.response_respondent_ids)
            == len(response_revisions)
            == len(self.response_task_ids)
            == len(self.response_content_fingerprints)
        ):
            raise assessment_error(
                "invalid_response_task_revision_axis",
                "$.response_task_revision_fingerprints",
                "every response must retain aligned respondent, task, revision, and content provenance",
            )
        object.__setattr__(self, "task_revision_fingerprints", revisions)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "task_family_ids", task_family_ids)
        object.__setattr__(
            self,
            "response_task_revision_fingerprints",
            response_revisions,
        )
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical sparse design content without dense arrays."""
        task_revisions = [
            {
                "task_revision_fingerprint": revision,
                "task_id": task_id,
                "task_family_id": task_family_id,
            }
            for revision, task_id, task_family_id in zip(
                self.task_revision_fingerprints,
                self.task_ids,
                self.task_family_ids,
                strict=True,
            )
        ]
        respondent_task_responses = [
            {
                "respondent_id": respondent_id,
                "task_revision_fingerprint": task_revision_fingerprint,
                "task_id": task_id,
                "response_id": response_id,
                "response_content_fingerprint": response_content_fingerprint,
            }
            for (
                respondent_id,
                task_revision_fingerprint,
                task_id,
                response_id,
                response_content_fingerprint,
            ) in zip(
                self.response_respondent_ids,
                self.response_task_revision_fingerprints,
                self.response_task_ids,
                self.response_ids,
                self.response_content_fingerprints,
                strict=True,
            )
        ]
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "occasion_id": self.occasion_id,
            "criterion_id": self.criterion_id,
            "category_values": list(self.category_values),
            "respondent_ids": list(self.respondent_ids),
            "task_revision_fingerprints": list(
                self.task_revision_fingerprints
            ),
            "task_ids": list(self.task_ids),
            "task_family_ids": list(self.task_family_ids),
            "task_revisions": task_revisions,
            "response_ids": list(self.response_ids),
            "response_respondent_ids": list(self.response_respondent_ids),
            "response_task_revision_fingerprints": list(
                self.response_task_revision_fingerprints
            ),
            "response_task_ids": list(self.response_task_ids),
            "response_content_fingerprints": list(
                self.response_content_fingerprints
            ),
            "respondent_task_responses": respondent_task_responses,
            "rater_engine_ids": list(self.rater_engine_ids),
            "rater_engine_family_ids": list(self.rater_engine_family_ids),
            "rater_engine_fingerprints": list(self.rater_engine_fingerprints),
            "rating_fingerprints": [
                record.rating_fingerprint for record in self.rating_records
            ],
            "respondent_task_connected": self.respondent_task_connected,
            "task_rater_connected": self.task_rater_connected,
            "connected": self.connected,
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete ordered sparse design."""
        return artifact_digest(self)

    @property
    def design_handle(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"scoring_facets_design_{self.design_fingerprint[:32]}"

    def _indexes(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Return deterministic respondent, revision, and rater index maps."""
        return (
            {value: index for index, value in enumerate(self.respondent_ids)},
            {
                value: index
                for index, value in enumerate(
                    self.task_revision_fingerprints
                )
            },
            {
                value: index
                for index, value in enumerate(self.rater_engine_fingerprints)
            },
        )

    def _dense_array(self, *, zero_based: bool) -> np.ndarray:
        """Materialize one fresh respondent-revision-rater tensor."""
        output = np.full(
            (
                len(self.respondent_ids),
                len(self.task_revision_fingerprints),
                len(self.rater_engine_fingerprints),
            ),
            np.nan,
            dtype=np.float64,
        )
        respondent_index, task_index, rater_index = self._indexes()
        category_index = {
            value: index for index, value in enumerate(self.category_values)
        }
        for record in self.rating_records:
            if record.status is not ObservationStatus.SCORED:
                continue
            score = record.score_category
            if score is None:  # pragma: no cover - guaranteed by rating record
                raise RuntimeError("scored rating category is unavailable")
            output[
                respondent_index[record.respondent_id],
                task_index[record.task_revision_fingerprint],
                rater_index[record.engine_fingerprint],
            ] = category_index[score] if zero_based else score
        return output

    def responses_array(self) -> np.ndarray:
        """Return a zero-based float tensor accepted by ``fit_facets``."""
        return self._dense_array(zero_based=True)

    def original_scores_array(self) -> np.ndarray:
        """Return a float tensor preserving original rubric score labels."""
        return self._dense_array(zero_based=False)

    def response_states(
        self,
    ) -> tuple[tuple[tuple[ObservationStatus | None, ...], ...], ...]:
        """Return exact assigned states and ``None`` for unassigned cells."""
        states: list[list[list[ObservationStatus | None]]] = [
            [
                [None for _rater in self.rater_engine_fingerprints]
                for _revision in self.task_revision_fingerprints
            ]
            for _respondent in self.respondent_ids
        ]
        respondent_index, task_index, rater_index = self._indexes()
        for record in self.rating_records:
            states[respondent_index[record.respondent_id]][
                task_index[record.task_revision_fingerprint]
            ][rater_index[record.engine_fingerprint]] = record.status
        return tuple(
            tuple(tuple(rater_states) for rater_states in task_states)
            for task_states in states
        )

    def to_fit_facets_kwargs(self) -> dict[str, Any]:
        """Return copied arguments accepted by Rust-backed ``fit_facets``."""
        observed_categories = {
            record.score_category
            for record in self.rating_records
            if record.status is ObservationStatus.SCORED
        }
        if observed_categories != set(self.category_values):
            raise assessment_error(
                "unobserved_facets_category",
                "$.category_values",
                "every declared score category must be observed before fitting",
            )
        return {
            "responses": self.responses_array(),
            "n_cat": len(self.category_values),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return sparse design content and deterministic identities."""
        return {
            **self._content_dict(),
            "design_handle": self.design_handle,
            "design_fingerprint": self.design_fingerprint,
        }


@dataclass(frozen=True)
class ScoringFacetsCalibrationBundle(CanonicalContract):
    """Content-addressed collection of criterion-specific facets designs."""

    assessment_fingerprint: str
    rubric_fingerprint: str
    construct_id: str
    occasion_id: str
    category_values: tuple[int, ...]
    designs: tuple[ScoringFacetsDesign, ...]
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _bundle_token: InitVar[object | None] = None

    def __post_init__(self, _bundle_token: object | None) -> None:
        """Reject direct construction outside the validated assembler."""
        if _bundle_token is not _BUNDLE_TOKEN:
            raise assessment_error(
                "unverified_facets_bundle",
                "$",
                "use build_scoring_facets_calibration_bundle",
            )
        object.__setattr__(
            self,
            "schema_version",
            assessment_schema_version(self.schema_version),
        )

    @property
    def criterion_ids(self) -> tuple[str, ...]:
        """Return criterion identifiers in deterministic design order."""
        return tuple(design.criterion_id for design in self.designs)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical bundle content without dense tensors."""
        return {
            "schema_version": self.schema_version,
            "assessment_fingerprint": self.assessment_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "construct_id": self.construct_id,
            "occasion_id": self.occasion_id,
            "category_values": list(self.category_values),
            "design_fingerprints": [
                design.design_fingerprint for design in self.designs
            ],
        }

    @property
    def bundle_fingerprint(self) -> str:
        """Return SHA-256 over the exact ordered design collection."""
        return artifact_digest(self)

    @property
    def bundle_handle(self) -> str:
        """Return a descriptive 128-bit public bundle handle."""
        return f"scoring_facets_bundle_{self.bundle_fingerprint[:32]}"

    def design_by_criterion(self) -> dict[str, ScoringFacetsDesign]:
        """Return a fresh criterion-to-design mapping."""
        return {design.criterion_id: design for design in self.designs}

    def to_dict(self) -> dict[str, Any]:
        """Return bundle content and deterministic identities."""
        return {
            **self._content_dict(),
            "criterion_ids": list(self.criterion_ids),
            "bundle_handle": self.bundle_handle,
            "bundle_fingerprint": self.bundle_fingerprint,
        }


def _validated_result_observations(
    *,
    request: ScoringRequest,
    result: ScoringResult,
) -> tuple[ScoreObservation, ...]:
    """Validate result-owned observation types, identities, and criterion scope."""
    observations = tuple(result.observations)
    for index, observation in enumerate(observations):
        if not isinstance(observation, ScoreObservation):
            raise assessment_error(
                "invalid_score_observation",
                f"$.result.observations[{index}]",
                "result observations must be ScoreObservation values",
            )
        if observation.criterion_id is None:
            raise assessment_error(
                "missing_observation_criterion",
                f"$.result.observations[{index}].criterion_id",
                "criterion-level observations require a criterion identifier",
            )
        if observation.criterion_id not in request.criterion_ids:
            raise assessment_error(
                "calibration_observation_criterion_mismatch",
                f"$.result.observations[{index}].criterion_id",
                "observation criterion is not declared by the scoring request",
            )
    criterion_ids = tuple(observation.criterion_id for observation in observations)
    if len(set(criterion_ids)) != len(criterion_ids):
        raise assessment_error(
            "duplicate_observation_criterion",
            "$.result.observations",
            "criterion-level observations must be unique",
        )
    observation_ids = tuple(observation.observation_id for observation in observations)
    if len(set(observation_ids)) != len(observation_ids):
        raise assessment_error(
            "duplicate_observation_id",
            "$.result.observations",
            "observation identifiers must be unique",
        )
    if result.requested_criterion_ids != request.criterion_ids:
        raise assessment_error(
            "calibration_result_criteria_mismatch",
            "$.result.requested_criterion_ids",
            "result criterion scope does not match the supplied scoring request",
        )
    if tuple(sorted(criterion_ids)) != request.criterion_ids:
        raise assessment_error(
            "calibration_observation_coverage_mismatch",
            "$.result.observations",
            "observations must cover every requested criterion exactly once",
        )
    return observations


def build_scoring_facets_rating_records(
    *,
    request: ScoringRequest,
    result: ScoringResult,
    engine: EngineDescriptor,
) -> tuple[ScoringFacetsRatingRecord, ...]:
    """Project one matched criterion-level execution into rating records."""
    if not isinstance(request, ScoringRequest):
        raise assessment_error(
            "invalid_scoring_request",
            "$.request",
            "request must be a ScoringRequest",
        )
    if not isinstance(result, ScoringResult):
        raise assessment_error(
            "invalid_scoring_result",
            "$.result",
            "result must be a ScoringResult",
        )
    if not isinstance(engine, EngineDescriptor):
        raise assessment_error(
            "invalid_engine_descriptor",
            "$.engine",
            "engine must be an EngineDescriptor",
        )
    if request.granularity is not ObservationGranularity.CRITERION_LEVEL:
        raise assessment_error(
            "unsupported_calibration_granularity",
            "$.request.granularity",
            "facets calibration requires criterion-level observations",
        )
    if result.granularity is not ObservationGranularity.CRITERION_LEVEL:
        raise assessment_error(
            "unsupported_calibration_granularity",
            "$.result.granularity",
            "facets calibration requires criterion-level observations",
        )
    if result.request_fingerprint != request.request_fingerprint:
        raise assessment_error(
            "calibration_request_result_mismatch",
            "$.result.request_fingerprint",
            "result is not bound to the supplied scoring request",
        )
    if result.engine_fingerprint != engine.engine_fingerprint:
        raise assessment_error(
            "calibration_engine_result_mismatch",
            "$.result.engine_fingerprint",
            "result is not bound to the supplied engine descriptor",
        )
    observations = _validated_result_observations(request=request, result=result)
    expected_observation_fields = (
        (
            "request_fingerprint",
            request.request_fingerprint,
            "calibration_observation_request_mismatch",
        ),
        (
            "engine_fingerprint",
            engine.engine_fingerprint,
            "calibration_observation_engine_mismatch",
        ),
        (
            "assessment_fingerprint",
            request.assessment_fingerprint,
            "calibration_observation_assessment_mismatch",
        ),
        (
            "rubric_fingerprint",
            request.rubric_fingerprint,
            "calibration_observation_rubric_mismatch",
        ),
        (
            "construct_id",
            request.construct_id,
            "calibration_observation_construct_mismatch",
        ),
        (
            "granularity",
            request.granularity,
            "calibration_observation_granularity_mismatch",
        ),
    )
    for index, observation in enumerate(observations):
        observation_path = f"$.result.observations[{index}]"
        for field_name, expected_value, error_code in expected_observation_fields:
            if getattr(observation, field_name) != expected_value:
                raise assessment_error(
                    error_code,
                    f"{observation_path}.{field_name}",
                    "observation provenance does not match the supplied execution",
                )

    records = tuple(
        ScoringFacetsRatingRecord(
            assessment_fingerprint=observation.assessment_fingerprint,
            rubric_fingerprint=observation.rubric_fingerprint,
            construct_id=observation.construct_id,
            request_fingerprint=request.request_fingerprint,
            result_fingerprint=result.result_fingerprint,
            observation_fingerprint=observation.observation_fingerprint,
            respondent_id=request.respondent_id,
            response_id=request.response_id,
            response_content_fingerprint=request.response_content_fingerprint,
            task_id=request.task_id,
            task_revision_fingerprint=request.task_revision_fingerprint,
            task_family_id=request.task_family_id,
            occasion_id=request.occasion_id,
            criterion_id=observation.criterion_id,
            engine_id=engine.engine_id,
            engine_family_id=engine.engine_family_id,
            engine_fingerprint=engine.engine_fingerprint,
            status=observation.status,
            score_category=observation.score_category,
            allowed_scores=request.allowed_scores,
            _rating_token=_RATING_TOKEN,
        )
        for observation in observations
    )
    keyed_records = [
        (record.criterion_id, record.rating_fingerprint, record)
        for record in records
    ]
    keyed_records.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in keyed_records)


def _identity_provenance(
    records: tuple[ScoringFacetsRatingRecord, ...],
) -> tuple[
    dict[str, tuple[str, str, str, str, str]],
    dict[tuple[str, str], tuple[str, str]],
    dict[str, tuple[str, str]],
    dict[str, tuple[str, str]],
]:
    """Validate response, respondent-revision, task-revision, and rater identity."""
    response_contracts: dict[str, tuple[str, str, str, str, str]] = {}
    respondent_task_contracts: dict[tuple[str, str], tuple[str, str]] = {}
    task_revision_contracts: dict[str, tuple[str, str]] = {}
    rater_contracts: dict[str, tuple[str, str]] = {}
    for index, record in enumerate(records):
        response_contract = (
            record.respondent_id,
            record.task_id,
            record.task_revision_fingerprint,
            record.task_family_id,
            record.response_content_fingerprint,
        )
        previous_response = response_contracts.get(record.response_id)
        if previous_response is not None and previous_response != response_contract:
            fields = (
                "respondent_id",
                "task_id",
                "task_revision_fingerprint",
                "task_family_id",
                "response_content_fingerprint",
            )
            conflict_field = next(
                field_name
                for field_name, previous, current in zip(
                    fields,
                    previous_response,
                    response_contract,
                    strict=True,
                )
                if previous != current
            )
            raise assessment_error(
                "response_provenance_conflict",
                f"$.records[{index}].{conflict_field}",
                "one response identity has conflicting provenance",
            )
        response_contracts[record.response_id] = response_contract

        revision_contract = (record.task_id, record.task_family_id)
        previous_revision = task_revision_contracts.get(
            record.task_revision_fingerprint
        )
        if previous_revision is not None and previous_revision != revision_contract:
            conflict_field = (
                "task_id"
                if previous_revision[0] != revision_contract[0]
                else "task_family_id"
            )
            raise assessment_error(
                "task_revision_provenance_conflict",
                f"$.records[{index}].{conflict_field}",
                "one task revision has conflicting logical task provenance",
            )
        task_revision_contracts[record.task_revision_fingerprint] = (
            revision_contract
        )

        respondent_task_key = (
            record.respondent_id,
            record.task_revision_fingerprint,
        )
        respondent_task_contract = (
            record.response_id,
            record.response_content_fingerprint,
        )
        previous_cell = respondent_task_contracts.get(respondent_task_key)
        if previous_cell is not None and previous_cell != respondent_task_contract:
            conflict_field = (
                "response_id"
                if previous_cell[0] != respondent_task_contract[0]
                else "response_content_fingerprint"
            )
            raise assessment_error(
                "respondent_task_response_conflict",
                f"$.records[{index}].{conflict_field}",
                "one respondent-task-revision cell has conflicting response provenance",
            )
        respondent_task_contracts[respondent_task_key] = respondent_task_contract

        rater_contract = (record.engine_id, record.engine_family_id)
        previous_rater = rater_contracts.get(record.engine_fingerprint)
        if previous_rater is not None and previous_rater != rater_contract:
            raise assessment_error(
                "rater_provenance_conflict",
                f"$.records[{index}].engine_fingerprint",
                "one engine fingerprint has conflicting rater provenance",
            )
        rater_contracts[record.engine_fingerprint] = rater_contract
    return (
        response_contracts,
        respondent_task_contracts,
        task_revision_contracts,
        rater_contracts,
    )


def _bipartite_connected(
    *,
    left_ids: tuple[str, ...],
    right_ids: tuple[str, ...],
    edges: set[tuple[str, str]],
) -> bool:
    """Return whether one bounded bipartite incidence graph is connected."""
    left_nodes = tuple(f"left:{value}" for value in left_ids)
    right_nodes = tuple(f"right:{value}" for value in right_ids)
    adjacency: dict[str, set[str]] = {
        value: set() for value in (*left_nodes, *right_nodes)
    }
    for left_value, right_value in edges:
        left_node = f"left:{left_value}"
        right_node = f"right:{right_value}"
        adjacency[left_node].add(right_node)
        adjacency[right_node].add(left_node)
    pending = [left_nodes[0]]
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency[node] - visited)
    return len(visited) == len(adjacency)


def _design_connected(
    *,
    respondent_ids: tuple[str, ...],
    task_revision_fingerprints: tuple[str, ...],
    rater_fingerprints: tuple[str, ...],
    records: tuple[ScoringFacetsRatingRecord, ...],
) -> tuple[bool, bool]:
    """Return respondent-revision and revision-rater connectedness."""
    scored = tuple(
        record for record in records if record.status is ObservationStatus.SCORED
    )
    respondent_task_edges = {
        (record.respondent_id, record.task_revision_fingerprint)
        for record in scored
    }
    task_rater_edges = {
        (record.task_revision_fingerprint, record.engine_fingerprint)
        for record in scored
    }
    respondent_task_connected = _bipartite_connected(
        left_ids=respondent_ids,
        right_ids=task_revision_fingerprints,
        edges=respondent_task_edges,
    )
    task_rater_connected = _bipartite_connected(
        left_ids=task_revision_fingerprints,
        right_ids=rater_fingerprints,
        edges=task_rater_edges,
    )
    return respondent_task_connected, task_rater_connected


def _build_criterion_design(
    records: tuple[ScoringFacetsRatingRecord, ...],
    *,
    require_connected: bool,
) -> ScoringFacetsDesign:
    """Build one bounded criterion design indexed by exact task revision."""
    first = records[0]
    (
        response_contracts,
        _,
        task_revision_contracts,
        rater_contracts,
    ) = _identity_provenance(records)
    cells: set[tuple[str, str, str]] = set()
    observed_respondents: set[str] = set()
    observed_tasks: set[str] = set()
    observed_raters: set[str] = set()
    observed_categories: set[int] = set()

    for index, record in enumerate(records):
        cell = (
            record.respondent_id,
            record.task_revision_fingerprint,
            record.engine_fingerprint,
        )
        if cell in cells:
            raise assessment_error(
                "duplicate_facets_rating_cell",
                f"$.records[{index}]",
                "each respondent-task-revision-rater cell may occur once per criterion",
            )
        cells.add(cell)
        if record.status is ObservationStatus.SCORED:
            score = record.score_category
            if score is None:  # pragma: no cover - guaranteed by rating record
                raise RuntimeError("scored rating category is unavailable")
            observed_respondents.add(record.respondent_id)
            observed_tasks.add(record.task_revision_fingerprint)
            observed_raters.add(record.engine_fingerprint)
            observed_categories.add(score)

    respondent_ids = tuple(sorted({record.respondent_id for record in records}))
    task_revision_fingerprints = tuple(sorted(task_revision_contracts))
    rater_fingerprints = tuple(sorted(rater_contracts))
    if len(respondent_ids) < 2:
        raise assessment_error(
            "insufficient_facets_respondents",
            "$.records",
            "facets calibration requires at least two respondents",
        )
    if len(task_revision_fingerprints) < 2:
        raise assessment_error(
            "insufficient_facets_tasks",
            "$.records",
            "facets calibration requires at least two task revisions",
        )
    if len(rater_fingerprints) < 2:
        raise assessment_error(
            "insufficient_facets_raters",
            "$.records",
            "facets calibration requires at least two raters",
        )
    if (
        observed_respondents != set(respondent_ids)
        or observed_tasks != set(task_revision_fingerprints)
        or observed_raters != set(rater_fingerprints)
    ):
        raise assessment_error(
            "unobserved_facets_level",
            "$.records",
            "every respondent, task revision, and rater must have a scored rating",
        )
    if len(observed_categories) < 2:
        raise assessment_error(
            "single_observed_category",
            "$.records",
            "each criterion must observe at least two score categories",
        )

    dense_cell_count = (
        len(respondent_ids)
        * len(task_revision_fingerprints)
        * len(rater_fingerprints)
    )
    if dense_cell_count > MAX_SCORING_FACETS_CELLS:
        raise assessment_error(
            "facets_cell_budget_exceeded",
            "$.records",
            "dense respondent-task-revision-rater allocation exceeds the configured cell budget",
        )

    respondent_task_connected, task_rater_connected = _design_connected(
        respondent_ids=respondent_ids,
        task_revision_fingerprints=task_revision_fingerprints,
        rater_fingerprints=rater_fingerprints,
        records=records,
    )
    if require_connected and not respondent_task_connected:
        raise assessment_error(
            "unidentified_respondent_task_design",
            "$.records",
            "scored respondent-task-revision graph must be connected",
        )
    if require_connected and not task_rater_connected:
        raise assessment_error(
            "disconnected_task_rater_design",
            "$.records",
            "scored task-revision-rater graph must be connected",
        )

    response_ids = tuple(sorted(response_contracts))
    return ScoringFacetsDesign(
        assessment_fingerprint=first.assessment_fingerprint,
        rubric_fingerprint=first.rubric_fingerprint,
        construct_id=first.construct_id,
        occasion_id=first.occasion_id,
        criterion_id=first.criterion_id,
        category_values=first.allowed_scores,
        respondent_ids=respondent_ids,
        task_revision_fingerprints=task_revision_fingerprints,
        task_ids=tuple(
            task_revision_contracts[value][0]
            for value in task_revision_fingerprints
        ),
        task_family_ids=tuple(
            task_revision_contracts[value][1]
            for value in task_revision_fingerprints
        ),
        response_ids=response_ids,
        response_respondent_ids=tuple(
            response_contracts[value][0] for value in response_ids
        ),
        response_task_revision_fingerprints=tuple(
            response_contracts[value][2] for value in response_ids
        ),
        response_task_ids=tuple(
            response_contracts[value][1] for value in response_ids
        ),
        response_content_fingerprints=tuple(
            response_contracts[value][4] for value in response_ids
        ),
        rater_engine_ids=tuple(
            rater_contracts[value][0] for value in rater_fingerprints
        ),
        rater_engine_family_ids=tuple(
            rater_contracts[value][1] for value in rater_fingerprints
        ),
        rater_engine_fingerprints=rater_fingerprints,
        rating_records=records,
        respondent_task_connected=respondent_task_connected,
        task_rater_connected=task_rater_connected,
        connected=respondent_task_connected and task_rater_connected,
        _design_token=_DESIGN_TOKEN,
    )


def build_scoring_facets_calibration_bundle(
    records: Iterable[ScoringFacetsRatingRecord],
    *,
    require_connected: bool = True,
) -> ScoringFacetsCalibrationBundle:
    """Assemble ratings into separate criterion-specific Rust handoffs.

    All ratings must share one assessment, rubric, construct, occasion, and
    ordered score scale. This baseline deliberately does not average analytic
    criteria or claim one cross-criterion general factor.
    """
    connected_required = strict_boolean(require_connected, "require_connected")
    raw = bounded_values(
        records,
        "records",
        minimum=1,
        maximum=MAX_SCORING_FACETS_RATINGS,
    )
    for index, record in enumerate(raw):
        if not isinstance(record, ScoringFacetsRatingRecord):
            raise assessment_error(
                "invalid_facets_rating_record",
                f"$.records[{index}]",
                "records must contain ScoringFacetsRatingRecord values",
            )
    keyed = [
        (record.criterion_id, record.rating_fingerprint, record)
        for record in raw
    ]
    keyed.sort(key=lambda item: (item[0], item[1]))
    ordered = tuple(item[2] for item in keyed)
    if len({item[1] for item in keyed}) != len(keyed):
        raise assessment_error(
            "duplicate_facets_rating_record",
            "$.records",
            "rating records must be unique",
        )

    contract_keys = {
        (
            record.assessment_fingerprint,
            record.rubric_fingerprint,
            record.construct_id,
            record.occasion_id,
            record.allowed_scores,
        )
        for record in ordered
    }
    if len(contract_keys) != 1:
        raise assessment_error(
            "mixed_facets_calibration_contract",
            "$.records",
            "ratings must share assessment, rubric, construct, occasion, and scale",
        )
    _identity_provenance(ordered)

    grouped: dict[str, list[ScoringFacetsRatingRecord]] = {}
    for record in ordered:
        grouped.setdefault(record.criterion_id, []).append(record)
    designs = tuple(
        _build_criterion_design(
            tuple(grouped[criterion_id]),
            require_connected=connected_required,
        )
        for criterion_id in sorted(grouped)
    )
    first = ordered[0]
    return ScoringFacetsCalibrationBundle(
        assessment_fingerprint=first.assessment_fingerprint,
        rubric_fingerprint=first.rubric_fingerprint,
        construct_id=first.construct_id,
        occasion_id=first.occasion_id,
        category_values=first.allowed_scores,
        designs=designs,
        _bundle_token=_BUNDLE_TOKEN,
    )


def fit_scoring_facets_design(
    design: ScoringFacetsDesign,
    *,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    allow_disconnected: bool = False,
):
    """Fit one identified design by delegating arithmetic to ``fit_facets``."""
    if not isinstance(design, ScoringFacetsDesign):
        raise assessment_error(
            "invalid_facets_design",
            "$.design",
            "design must be a ScoringFacetsDesign",
        )
    strict_boolean(allow_disconnected, "allow_disconnected")
    if not design.respondent_task_connected:
        raise assessment_error(
            "unidentified_respondent_task_design",
            "$.design.respondent_task_connected",
            "respondent-task-revision design is not identified",
        )
    if not design.task_rater_connected:
        raise assessment_error(
            "disconnected_task_rater_design",
            "$.design.task_rater_connected",
            "task-revision-rater design is disconnected",
        )
    from fast_mlsirm.facets import fit_facets

    return fit_facets(
        **design.to_fit_facets_kwargs(),
        q_theta=q_theta,
        max_iter=max_iter,
        tol=tol,
    )


def fit_scoring_facets_bundle(
    bundle: ScoringFacetsCalibrationBundle,
    *,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
    allow_disconnected: bool = False,
) -> dict[str, Any]:
    """Fit every criterion design through the existing Rust-backed estimator."""
    if not isinstance(bundle, ScoringFacetsCalibrationBundle):
        raise assessment_error(
            "invalid_facets_bundle",
            "$.bundle",
            "bundle must be a ScoringFacetsCalibrationBundle",
        )
    return {
        design.criterion_id: fit_scoring_facets_design(
            design,
            q_theta=q_theta,
            max_iter=max_iter,
            tol=tol,
            allow_disconnected=allow_disconnected,
        )
        for design in bundle.designs
    }


__all__ = [
    "MAX_SCORING_FACETS_CELLS",
    "MAX_SCORING_FACETS_RATINGS",
    "ScoringFacetsCalibrationBundle",
    "ScoringFacetsDesign",
    "ScoringFacetsRatingRecord",
    "build_scoring_facets_calibration_bundle",
    "build_scoring_facets_rating_records",
    "fit_scoring_facets_bundle",
    "fit_scoring_facets_design",
]
