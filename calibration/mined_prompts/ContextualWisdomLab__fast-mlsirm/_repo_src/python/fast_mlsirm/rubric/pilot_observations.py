"""Provenance-safe pilot observations and many-facet calibration handoff."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np

from ..config import MAX_POLYTOMOUS_CATEGORIES
from .models import (
    SCHEMA_VERSION,
    _FINGERPRINT_PATTERN,
    _bounded_values,
    _enum_value,
    _identifier,
    _integer,
    _schema_version,
    _sha256_hex,
    _text,
)
from .verified_pilot import PilotCandidateRecord

MAX_PILOT_OBSERVATIONS = 100_000
# The handoff materializes both Python tuple tensors and a float64 NumPy tensor.
# Bound the full persons-by-items-by-raters cross-product, not only supplied
# sparse records, so untrusted identifiers cannot amplify a bounded input into
# an unbounded dense allocation.
MAX_FACETS_PILOT_CELLS = 1_000_000
# The MIRT handoff materializes a dense persons-by-items matrix; bound the full
# cross-product for the same amplification reason as the facets design.
MAX_MIRT_PILOT_CELLS = 1_000_000
_OBSERVATION_TOKEN = object()
_DESIGN_TOKEN = object()
_MIRT_DESIGN_TOKEN = object()


class PilotResponseState(str, Enum):
    """Explicit state of one pilot response without failure coercion."""

    OBSERVED = "observed"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PilotObservationError(ValueError):
    """Stable structured rejection for pilot-observation assembly failures."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Store bounded machine-readable error metadata without response values."""
        self.code = _identifier(code, "code")
        if not isinstance(path, str) or not path.startswith("$"):
            raise ValueError("path must be a JSON-style path beginning with '$'")
        self.path = path
        self.message = _text(message, "message", maximum=512)
        super().__init__(f"{self.code} at {self.path}: {self.message}")


def _error(code: str, path: str, message: str) -> PilotObservationError:
    """Construct one redacted observation error."""
    return PilotObservationError(code, path, message)


def _fingerprint(value: Any, name: str) -> str:
    """Normalize a complete lower-hexadecimal SHA-256 fingerprint."""
    normalized = _text(value, name, maximum=64)
    if _FINGERPRINT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be 64 lower hexadecimal characters")
    return normalized


@dataclass(frozen=True)
class PilotObservationRecord:
    """One immutable response bound to an admitted pilot candidate."""

    pilot_study_id: str
    query_testlet_id: str
    generator_family_id: str
    judge_policy_id: str
    occasion_id: str
    respondent_id: str
    item_id: str
    rater_id: str
    pilot_record_fingerprint: str
    response_state: PilotResponseState
    category: int | None = None
    schema_version: str = SCHEMA_VERSION
    _admission_token: InitVar[object | None] = None

    def __post_init__(self, _admission_token: object | None) -> None:
        """Normalize fields and reject records not issued by the public builder."""
        if _admission_token is not _OBSERVATION_TOKEN:
            raise ValueError(
                "PilotObservationRecord must be created by "
                "build_pilot_observation_record"
            )
        for name in (
            "pilot_study_id",
            "query_testlet_id",
            "generator_family_id",
            "judge_policy_id",
            "occasion_id",
            "respondent_id",
            "item_id",
            "rater_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "pilot_record_fingerprint",
            _fingerprint(self.pilot_record_fingerprint, "pilot_record_fingerprint"),
        )
        response_state = _enum_value(
            self.response_state,
            PilotResponseState,
            "response_state",
        )
        object.__setattr__(self, "response_state", response_state)
        if response_state is PilotResponseState.OBSERVED:
            category = _integer(self.category, "category")
            if not 0 <= category < MAX_POLYTOMOUS_CATEGORIES:
                raise ValueError(
                    "category must be between 0 and "
                    f"{MAX_POLYTOMOUS_CATEGORIES - 1}"
                )
            object.__setattr__(self, "category", category)
        elif self.category is not None:
            raise ValueError("category must be None unless response_state='observed'")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical response content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "query_testlet_id": self.query_testlet_id,
            "generator_family_id": self.generator_family_id,
            "judge_policy_id": self.judge_policy_id,
            "occasion_id": self.occasion_id,
            "respondent_id": self.respondent_id,
            "item_id": self.item_id,
            "rater_id": self.rater_id,
            "pilot_record_fingerprint": self.pilot_record_fingerprint,
            "response_state": self.response_state.value,
            "category": self.category,
        }

    @property
    def observation_fingerprint(self) -> str:
        """Return the SHA-256 identity of this normalized observation."""
        return _sha256_hex(self._content_dict())

    @property
    def observation_id(self) -> str:
        """Return a descriptive 128-bit public observation handle."""
        return f"pilot_observation_{self.observation_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return normalized observation content and deterministic identities."""
        return {
            **self._content_dict(),
            "observation_id": self.observation_id,
            "observation_fingerprint": self.observation_fingerprint,
        }


@dataclass(frozen=True)
class PilotItemProvenance:
    """Immutable per-item provenance retained in a calibration design."""

    item_id: str
    pilot_record_fingerprint: str
    query_testlet_id: str
    generator_family_id: str
    judge_policy_id: str
    occasion_id: str

    def __post_init__(self) -> None:
        """Normalize item-level pilot provenance."""
        for name in (
            "item_id",
            "query_testlet_id",
            "generator_family_id",
            "judge_policy_id",
            "occasion_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "pilot_record_fingerprint",
            _fingerprint(self.pilot_record_fingerprint, "pilot_record_fingerprint"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible per-item provenance."""
        return {
            "item_id": self.item_id,
            "pilot_record_fingerprint": self.pilot_record_fingerprint,
            "query_testlet_id": self.query_testlet_id,
            "generator_family_id": self.generator_family_id,
            "judge_policy_id": self.judge_policy_id,
            "occasion_id": self.occasion_id,
        }


@dataclass(frozen=True)
class FacetsPilotDesign:
    """Deterministic persons-by-items-by-raters handoff for ``fit_facets``."""

    pilot_study_id: str
    respondent_ids: tuple[str, ...]
    item_provenance: tuple[PilotItemProvenance, ...]
    rater_ids: tuple[str, ...]
    n_cat: int
    responses: tuple[tuple[tuple[int | None, ...], ...], ...]
    response_states: tuple[tuple[tuple[PilotResponseState, ...], ...], ...]
    schema_version: str = SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated batch assembler."""
        if _design_token is not _DESIGN_TOKEN:
            raise ValueError(
                "FacetsPilotDesign must be created by build_facets_pilot_design"
            )
        object.__setattr__(
            self,
            "pilot_study_id",
            _identifier(self.pilot_study_id, "pilot_study_id"),
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return item identifiers in response-tensor order."""
        return tuple(entry.item_id for entry in self.item_provenance)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical design content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "respondent_ids": list(self.respondent_ids),
            "item_provenance": [entry.to_dict() for entry in self.item_provenance],
            "rater_ids": list(self.rater_ids),
            "n_cat": self.n_cat,
            "responses": [
                [list(rater_values) for rater_values in item_values]
                for item_values in self.responses
            ],
            "response_states": [
                [
                    [state.value for state in rater_states]
                    for rater_states in item_states
                ]
                for item_states in self.response_states
            ],
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete ordered calibration design."""
        return _sha256_hex(self._content_dict())

    @property
    def design_id(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"facets_pilot_design_{self.design_fingerprint[:32]}"

    def responses_array(self) -> np.ndarray:
        """Return a fresh float tensor with non-observed states represented by NaN."""
        return np.asarray(
            [
                [
                    [np.nan if value is None else value for value in rater_values]
                    for rater_values in item_values
                ]
                for item_values in self.responses
            ],
            dtype=np.float64,
        )

    def to_fit_facets_kwargs(self) -> dict[str, Any]:
        """Return copied arguments accepted directly by ``fast_mlsirm.fit_facets``."""
        return {"responses": self.responses_array(), "n_cat": self.n_cat}

    def to_dict(self) -> dict[str, Any]:
        """Return canonical design content and deterministic identities."""
        return {
            **self._content_dict(),
            "design_id": self.design_id,
            "design_fingerprint": self.design_fingerprint,
        }


def build_pilot_observation_record(
    pilot_record: PilotCandidateRecord,
    *,
    respondent_id: str,
    rater_id: str,
    response_state: PilotResponseState | str,
    category: int | None = None,
) -> PilotObservationRecord:
    """Bind one response state to a replay-verified pilot admission record."""
    if not isinstance(pilot_record, PilotCandidateRecord):
        raise TypeError("pilot_record must be a verified PilotCandidateRecord")
    return PilotObservationRecord(
        pilot_study_id=pilot_record.pilot_study_id,
        query_testlet_id=pilot_record.query_testlet_id,
        generator_family_id=pilot_record.generator_family_id,
        judge_policy_id=pilot_record.judge_policy_id,
        occasion_id=pilot_record.occasion_id,
        respondent_id=respondent_id,
        item_id=pilot_record.item_id,
        rater_id=rater_id,
        pilot_record_fingerprint=pilot_record.pilot_record_fingerprint,
        response_state=response_state,
        category=category,
        _admission_token=_OBSERVATION_TOKEN,
    )


def _item_provenance(record: PilotObservationRecord) -> PilotItemProvenance:
    """Extract the immutable item-level provenance represented by one response."""
    return PilotItemProvenance(
        item_id=record.item_id,
        pilot_record_fingerprint=record.pilot_record_fingerprint,
        query_testlet_id=record.query_testlet_id,
        generator_family_id=record.generator_family_id,
        judge_policy_id=record.judge_policy_id,
        occasion_id=record.occasion_id,
    )


def _normalized_n_cat(
    n_cat: int | None,
    observed_categories: tuple[int, ...],
) -> int:
    """Validate or infer the bounded number of ordered response categories."""
    if not observed_categories:
        raise _error(
            "no_observed_response",
            "$.records",
            "at least one observed response is required",
        )
    if n_cat is None:
        inferred = max(observed_categories) + 1
        if inferred < 2:
            raise _error(
                "single_category_design",
                "$.records",
                "observed responses must contain at least two categories",
            )
        return inferred
    normalized = _integer(n_cat, "n_cat")
    if not 2 <= normalized <= MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(
            f"n_cat must be between 2 and {MAX_POLYTOMOUS_CATEGORIES}"
        )
    if max(observed_categories) >= normalized:
        raise _error(
            "category_out_of_range",
            "$.records",
            "an observed category is outside the declared category range",
        )
    return normalized


@dataclass(frozen=True)
class MirtPilotDesign:
    """Deterministic persons-by-items binary handoff for ``fast_mlsirm.fit``.

    Items are assigned to trait dimensions by their ``query_testlet_id`` in
    sorted testlet order, matching this repository's simple-structure model
    specialization: every item generated for one query testlet loads on that
    testlet's dimension. The mapping is fully disclosed through
    ``factor_testlet_ids`` and ``item_factor_ids`` and is part of the
    content-addressed design identity, so a buyer can audit exactly which
    dimension each pilot item was calibrated on.
    """

    pilot_study_id: str
    respondent_ids: tuple[str, ...]
    item_provenance: tuple[PilotItemProvenance, ...]
    factor_testlet_ids: tuple[str, ...]
    item_factor_ids: tuple[int, ...]
    responses: tuple[tuple[int | None, ...], ...]
    response_states: tuple[tuple[PilotResponseState, ...], ...]
    rater_assignments: tuple[tuple[str | None, ...], ...]
    schema_version: str = SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated batch assembler."""
        if _design_token is not _MIRT_DESIGN_TOKEN:
            raise ValueError(
                "MirtPilotDesign must be created by build_mirt_pilot_design"
            )
        object.__setattr__(
            self,
            "pilot_study_id",
            _identifier(self.pilot_study_id, "pilot_study_id"),
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return item identifiers in response-matrix column order."""
        return tuple(entry.item_id for entry in self.item_provenance)

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical design content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "respondent_ids": list(self.respondent_ids),
            "item_provenance": [entry.to_dict() for entry in self.item_provenance],
            "factor_testlet_ids": list(self.factor_testlet_ids),
            "item_factor_ids": list(self.item_factor_ids),
            "responses": [list(item_values) for item_values in self.responses],
            "response_states": [
                [state.value for state in item_states]
                for item_states in self.response_states
            ],
            "rater_assignments": [
                list(item_raters) for item_raters in self.rater_assignments
            ],
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete ordered calibration design."""
        return _sha256_hex(self._content_dict())

    @property
    def design_id(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"mirt_pilot_design_{self.design_fingerprint[:32]}"

    def responses_array(self) -> np.ndarray:
        """Return a fresh float matrix with non-observed states represented by NaN."""
        return np.asarray(
            [
                [np.nan if value is None else value for value in item_values]
                for item_values in self.responses
            ],
            dtype=np.float64,
        )

    def factor_id_array(self) -> np.ndarray:
        """Return a fresh per-item trait-dimension assignment vector."""
        return np.asarray(self.item_factor_ids, dtype=np.int64)

    def to_fit_kwargs(self) -> dict[str, Any]:
        """Return copied arguments accepted directly by ``fast_mlsirm.fit``."""
        return {
            "responses": self.responses_array(),
            "factor_id": self.factor_id_array(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return canonical design content and deterministic identities."""
        return {
            **self._content_dict(),
            "design_id": self.design_id,
            "design_fingerprint": self.design_fingerprint,
        }


def build_facets_pilot_design(
    records: Iterable[PilotObservationRecord],
    *,
    n_cat: int | None = None,
) -> FacetsPilotDesign:
    """Assemble bounded pilot records into a deterministic many-facet tensor.

    Absent cells and explicit ``missing`` records become ``NaN`` for
    ``fit_facets``. ``not_applicable`` and ``insufficient_evidence`` also remain
    ``NaN`` numerically, but their exact states are retained in
    ``response_states`` and in the content-addressed design artifact. The full
    dense persons-by-items-by-raters design is bounded before allocation.
    """
    materialized = _bounded_values(
        records,
        "records",
        minimum=1,
        maximum=MAX_PILOT_OBSERVATIONS,
    )
    for index, record in enumerate(materialized):
        if not isinstance(record, PilotObservationRecord):
            raise TypeError(f"records[{index}] must be a PilotObservationRecord")

    pilot_study_id = materialized[0].pilot_study_id
    item_metadata: dict[str, PilotItemProvenance] = {}
    cells: dict[tuple[str, str, str], PilotObservationRecord] = {}
    observed_categories: list[int] = []
    observed_respondent_ids: set[str] = set()
    observed_item_ids: set[str] = set()
    observed_rater_ids: set[str] = set()
    for index, record in enumerate(materialized):
        if record.pilot_study_id != pilot_study_id:
            raise _error(
                "mixed_pilot_study",
                f"$.records[{index}].pilot_study_id",
                "all observations must belong to one pilot study",
            )
        provenance = _item_provenance(record)
        previous = item_metadata.get(record.item_id)
        if previous is not None and previous != provenance:
            raise _error(
                "item_provenance_conflict",
                f"$.records[{index}].item_id",
                "one item identifier is bound to conflicting pilot provenance",
            )
        item_metadata[record.item_id] = provenance
        cell = (record.respondent_id, record.item_id, record.rater_id)
        if cell in cells:
            raise _error(
                "duplicate_observation_cell",
                f"$.records[{index}]",
                "each respondent-item-rater cell may occur only once",
            )
        cells[cell] = record
        if record.response_state is PilotResponseState.OBSERVED:
            if record.category is None:  # pragma: no cover - guaranteed by record
                raise RuntimeError("observed response category is unavailable")
            observed_categories.append(record.category)
            observed_respondent_ids.add(record.respondent_id)
            observed_item_ids.add(record.item_id)
            observed_rater_ids.add(record.rater_id)

    category_count = _normalized_n_cat(n_cat, tuple(observed_categories))
    respondent_ids = tuple(sorted({record.respondent_id for record in materialized}))
    item_ids = tuple(sorted(item_metadata))
    rater_ids = tuple(sorted({record.rater_id for record in materialized}))

    for respondent_id in respondent_ids:
        if respondent_id not in observed_respondent_ids:
            raise _error(
                "unobserved_respondent",
                "$.records",
                "every respondent must have at least one observed response",
            )
    for item_id in item_ids:
        if item_id not in observed_item_ids:
            raise _error(
                "unobserved_item",
                "$.records",
                "every item must have at least one observed response",
            )
    for rater_id in rater_ids:
        if rater_id not in observed_rater_ids:
            raise _error(
                "unobserved_rater",
                "$.records",
                "every rater must have at least one observed response",
            )

    dense_cell_count = len(respondent_ids) * len(item_ids) * len(rater_ids)
    if dense_cell_count > MAX_FACETS_PILOT_CELLS:
        raise _error(
            "facets_design_cell_budget_exceeded",
            "$.records",
            f"dense facets design requires {dense_cell_count} cells; "
            f"maximum is {MAX_FACETS_PILOT_CELLS}",
        )

    respondent_index = {value: index for index, value in enumerate(respondent_ids)}
    item_index = {value: index for index, value in enumerate(item_ids)}
    rater_index = {value: index for index, value in enumerate(rater_ids)}

    responses: list[list[list[int | None]]] = [
        [[None for _rater in rater_ids] for _item in item_ids]
        for _respondent in respondent_ids
    ]
    states: list[list[list[PilotResponseState]]] = [
        [
            [PilotResponseState.MISSING for _rater in rater_ids]
            for _item in item_ids
        ]
        for _respondent in respondent_ids
    ]
    for record in materialized:
        person_position = respondent_index[record.respondent_id]
        item_position = item_index[record.item_id]
        rater_position = rater_index[record.rater_id]
        states[person_position][item_position][rater_position] = record.response_state
        responses[person_position][item_position][rater_position] = record.category

    return FacetsPilotDesign(
        pilot_study_id=pilot_study_id,
        respondent_ids=respondent_ids,
        item_provenance=tuple(item_metadata[item_id] for item_id in item_ids),
        rater_ids=rater_ids,
        n_cat=category_count,
        responses=tuple(
            tuple(tuple(rater_values) for rater_values in item_values)
            for item_values in responses
        ),
        response_states=tuple(
            tuple(tuple(rater_states) for rater_states in item_states)
            for item_states in states
        ),
        _design_token=_DESIGN_TOKEN,
    )


def build_mirt_pilot_design(
    records: Iterable[PilotObservationRecord],
) -> MirtPilotDesign:
    """Assemble bounded pilot records into a deterministic binary MIRT matrix.

    Absent cells and explicit ``missing`` records become ``NaN`` for
    ``fast_mlsirm.fit``; ``not_applicable`` and ``insufficient_evidence`` also
    remain ``NaN`` numerically while their exact states and per-cell rater
    assignments are retained in the content-addressed design artifact. Each
    respondent-item pair may carry at most one response regardless of rater,
    and observed categories must already be binary; multi-rater or polytomous
    pilot data belongs to ``build_facets_pilot_design`` instead of being
    silently aggregated or dichotomized here. The full dense persons-by-items
    matrix is bounded before allocation.
    """
    materialized = _bounded_values(
        records,
        "records",
        minimum=1,
        maximum=MAX_PILOT_OBSERVATIONS,
    )
    for index, record in enumerate(materialized):
        if not isinstance(record, PilotObservationRecord):
            raise TypeError(f"records[{index}] must be a PilotObservationRecord")

    pilot_study_id = materialized[0].pilot_study_id
    item_metadata: dict[str, PilotItemProvenance] = {}
    cells: dict[tuple[str, str], PilotObservationRecord] = {}
    observed_respondent_ids: set[str] = set()
    observed_item_ids: set[str] = set()
    for index, record in enumerate(materialized):
        if record.pilot_study_id != pilot_study_id:
            raise _error(
                "mixed_pilot_study",
                f"$.records[{index}].pilot_study_id",
                "all observations must belong to one pilot study",
            )
        provenance = _item_provenance(record)
        previous = item_metadata.get(record.item_id)
        if previous is not None and previous != provenance:
            raise _error(
                "item_provenance_conflict",
                f"$.records[{index}].item_id",
                "one item identifier is bound to conflicting pilot provenance",
            )
        item_metadata[record.item_id] = provenance
        cell = (record.respondent_id, record.item_id)
        if cell in cells:
            raise _error(
                "duplicate_person_item_cell",
                f"$.records[{index}]",
                "each respondent-item pair may carry one response; "
                "multi-rater designs require build_facets_pilot_design",
            )
        cells[cell] = record
        if record.response_state is PilotResponseState.OBSERVED:
            if record.category not in (0, 1):
                raise _error(
                    "non_binary_observed_category",
                    f"$.records[{index}].category",
                    "MIRT calibration accepts binary categories 0 and 1; "
                    "polytomous designs require build_facets_pilot_design",
                )
            observed_respondent_ids.add(record.respondent_id)
            observed_item_ids.add(record.item_id)

    if not observed_respondent_ids:
        raise _error(
            "no_observed_response",
            "$.records",
            "at least one observed response is required",
        )
    respondent_ids = tuple(sorted({record.respondent_id for record in materialized}))
    item_ids = tuple(sorted(item_metadata))
    for respondent_id in respondent_ids:
        if respondent_id not in observed_respondent_ids:
            raise _error(
                "unobserved_respondent",
                "$.records",
                "every respondent must have at least one observed response",
            )
    for item_id in item_ids:
        if item_id not in observed_item_ids:
            raise _error(
                "unobserved_item",
                "$.records",
                "every item must have at least one observed response",
            )

    dense_cell_count = len(respondent_ids) * len(item_ids)
    if dense_cell_count > MAX_MIRT_PILOT_CELLS:
        raise _error(
            "mirt_design_cell_budget_exceeded",
            "$.records",
            f"dense MIRT design requires {dense_cell_count} cells; "
            f"maximum is {MAX_MIRT_PILOT_CELLS}",
        )

    factor_testlet_ids = tuple(
        sorted({entry.query_testlet_id for entry in item_metadata.values()})
    )
    testlet_factor_index = {
        value: index for index, value in enumerate(factor_testlet_ids)
    }
    item_factor_ids = tuple(
        testlet_factor_index[item_metadata[item_id].query_testlet_id]
        for item_id in item_ids
    )

    respondent_index = {value: index for index, value in enumerate(respondent_ids)}
    item_index = {value: index for index, value in enumerate(item_ids)}
    responses: list[list[int | None]] = [
        [None for _item in item_ids] for _respondent in respondent_ids
    ]
    states: list[list[PilotResponseState]] = [
        [PilotResponseState.MISSING for _item in item_ids]
        for _respondent in respondent_ids
    ]
    raters: list[list[str | None]] = [
        [None for _item in item_ids] for _respondent in respondent_ids
    ]
    for record in materialized:
        person_position = respondent_index[record.respondent_id]
        item_position = item_index[record.item_id]
        states[person_position][item_position] = record.response_state
        responses[person_position][item_position] = record.category
        raters[person_position][item_position] = record.rater_id

    return MirtPilotDesign(
        pilot_study_id=pilot_study_id,
        respondent_ids=respondent_ids,
        item_provenance=tuple(item_metadata[item_id] for item_id in item_ids),
        factor_testlet_ids=factor_testlet_ids,
        item_factor_ids=item_factor_ids,
        responses=tuple(tuple(item_values) for item_values in responses),
        response_states=tuple(tuple(item_states) for item_states in states),
        rater_assignments=tuple(tuple(item_raters) for item_raters in raters),
        _design_token=_MIRT_DESIGN_TOKEN,
    )
