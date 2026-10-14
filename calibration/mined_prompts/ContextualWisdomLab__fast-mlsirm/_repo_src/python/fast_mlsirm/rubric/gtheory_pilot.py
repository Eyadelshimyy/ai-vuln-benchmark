"""Provenance-safe one-facet G-theory handoff for pilot scores."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from .models import (
    SCHEMA_VERSION,
    _bounded_values,
    _identifier,
    _schema_version,
    _sha256_hex,
)
from .pilot_observations import (
    FacetsPilotDesign,
    PilotItemProvenance,
    PilotObservationError,
    PilotObservationRecord,
    PilotResponseState,
    build_facets_pilot_design,
)

_GTHEORY_PI_DESIGN_TOKEN = object()
MAX_GTHEORY_D_STUDY_ROWS = 64
MAX_GTHEORY_PRIME_SIZE = 1_000_000
_DEFAULT_N_I_PRIME = (5, 10, 15, 20)
_NUMPY_INTEGER_SCALAR_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)
_NUMPY_FLOATING_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _is_exact_numpy_integer_scalar(value_type: type[Any]) -> bool:
    """Return whether ``value_type`` is one package-supported NumPy integer class."""
    return any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES)


def _is_exact_numpy_floating_scalar(value_type: type[Any]) -> bool:
    """Return whether ``value_type`` is one package-supported NumPy float class."""
    return any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES)


def _positive_design_size(value: Any, name: str) -> int:
    """Return one bounded trusted integer without caller protocol dispatch."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif _is_exact_numpy_integer_scalar(value_type):
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if not 1 <= normalized <= MAX_GTHEORY_PRIME_SIZE:
        raise ValueError(f"{name} must be between 1 and {MAX_GTHEORY_PRIME_SIZE}")
    return normalized


def _normalized_n_i_prime(values: Sequence[int] | Iterable[int]) -> tuple[int, ...]:
    """Materialize and validate a bounded one-facet D-study design sequence."""
    materialized = _bounded_values(
        values,
        "n_i_prime",
        minimum=1,
        maximum=MAX_GTHEORY_D_STUDY_ROWS,
    )
    return tuple(
        _positive_design_size(value, f"n_i_prime[{index}]")
        for index, value in enumerate(materialized)
    )


def _finite_cut(value: Any) -> float:
    """Return one finite mastery cut whose identity survives Rust ``f64`` marshalling."""
    value_type = type(value)
    integer_value: int | None = None
    if value_type is int:
        integer_value = value
        try:
            normalized = float(value)
        except OverflowError:
            raise ValueError("cut must be a finite number") from None
    elif _is_exact_numpy_integer_scalar(value_type):
        integer_value = int(value)
        normalized = float(integer_value)
    elif value_type is float:
        normalized = value
    elif _is_exact_numpy_floating_scalar(value_type):
        try:
            normalized = float(value)
        except OverflowError:
            # Mirror the estimator-control contract: a finite wide-longdouble
            # beyond binary64 range must fail as ValueError, not OverflowError.
            raise ValueError("cut must be a finite number") from None
    else:
        raise ValueError("cut must be a finite number")
    if not math.isfinite(normalized):
        raise ValueError("cut must be a finite number")
    if integer_value is not None and int(normalized) != integer_value:
        raise ValueError("cut must be exactly representable as float64")
    if value_type is np.longdouble and np.longdouble(normalized) != value:
        raise ValueError("cut must be exactly representable as float64")
    return normalized


@dataclass(frozen=True)
class GTheoryPiPilotDesign:
    """Content-addressed complete ``p x i`` pilot handoff for G theory.

    The wrapped :class:`FacetsPilotDesign` preserves the full generated-item
    provenance and exact response-state tensor. This narrower contract permits
    exactly one rater and one declared occasion and requires every
    respondent-item cell to be observed. It therefore exposes a complete,
    balanced persons-by-items score matrix without deleting, imputing,
    aggregating, or coercing any pilot response.

    The handoff is an input-governance artifact only. It does not establish that
    persons and items are random facets, that the ANOVA model is appropriate,
    that negative variance components should be clamped, or that any resulting
    generalizability/dependability coefficient supports operational use.
    """

    facets_design: FacetsPilotDesign
    rater_id: str
    occasion_id: str
    schema_version: str = SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction and revalidate every exposed design invariant."""
        if _design_token is not _GTHEORY_PI_DESIGN_TOKEN:
            raise ValueError(
                "GTheoryPiPilotDesign must be created by "
                "build_gtheory_pi_pilot_design"
            )
        if not isinstance(self.facets_design, FacetsPilotDesign):
            raise TypeError("facets_design must be a validated FacetsPilotDesign")
        object.__setattr__(self, "rater_id", _identifier(self.rater_id, "rater_id"))
        object.__setattr__(
            self,
            "occasion_id",
            _identifier(self.occasion_id, "occasion_id"),
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.schema_version != self.facets_design.schema_version:
            raise ValueError("schema_version must match the wrapped facets design")
        if self.facets_design.rater_ids != (self.rater_id,):
            raise PilotObservationError(
                "gtheory_pi_rater_confounded",
                "$.records",
                "one-facet G theory requires exactly one observed rater",
            )
        occasion_ids = tuple(
            sorted({entry.occasion_id for entry in self.facets_design.item_provenance})
        )
        if occasion_ids != (self.occasion_id,):
            raise PilotObservationError(
                "gtheory_pi_occasion_confounded",
                "$.records",
                "one-facet G theory requires exactly one declared occasion",
            )
        if len(self.facets_design.respondent_ids) < 2:
            raise PilotObservationError(
                "gtheory_pi_insufficient_respondents",
                "$.records",
                "one-facet G theory requires at least two respondents",
            )
        if len(self.facets_design.item_ids) < 2:
            raise PilotObservationError(
                "gtheory_pi_insufficient_items",
                "$.records",
                "one-facet G theory requires at least two items",
            )
        if any(
            state is not PilotResponseState.OBSERVED
            for person_states in self.facets_design.response_states
            for item_states in person_states
            for state in item_states
        ):
            raise PilotObservationError(
                "gtheory_pi_incomplete_design",
                "$.records",
                "every respondent-item cell must be explicitly observed",
            )

    @property
    def pilot_study_id(self) -> str:
        """Return the pilot-study identifier retained by the facets design."""
        return self.facets_design.pilot_study_id

    @property
    def respondent_ids(self) -> tuple[str, ...]:
        """Return respondent identifiers in score-matrix row order."""
        return self.facets_design.respondent_ids

    @property
    def item_provenance(self) -> tuple[PilotItemProvenance, ...]:
        """Return immutable item provenance in score-matrix column order."""
        return self.facets_design.item_provenance

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return item identifiers in score-matrix column order."""
        return self.facets_design.item_ids

    @property
    def scores(self) -> tuple[tuple[int, ...], ...]:
        """Return the complete immutable persons-by-items integer score matrix."""
        return tuple(
            tuple(int(item_values[0]) for item_values in person_values)
            for person_values in self.facets_design.responses
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical design content without outer derived identities."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "rater_id": self.rater_id,
            "occasion_id": self.occasion_id,
            "facets_design": self.facets_design.to_dict(),
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete one-facet G-theory handoff."""
        return _sha256_hex(self._content_dict())

    @property
    def design_id(self) -> str:
        """Return a descriptive 128-bit public G-theory-design handle."""
        return f"gtheory_pi_pilot_design_{self.design_fingerprint[:32]}"

    def scores_array(self) -> np.ndarray:
        """Return a fresh float64 persons-by-items matrix for Rust G theory."""
        return np.asarray(self.scores, dtype=np.float64)

    def to_gtheory_pi_kwargs(
        self,
        n_i_prime: Sequence[int] | Iterable[int] = _DEFAULT_N_I_PRIME,
    ) -> dict[str, Any]:
        """Return copied arguments accepted directly by ``gtheory_pi``."""
        return {
            "data": self.scores_array(),
            "n_i_prime": _normalized_n_i_prime(n_i_prime),
        }

    def to_phi_lambda_kwargs(
        self,
        cut: float,
        n_i_prime: Sequence[int] | Iterable[int] = _DEFAULT_N_I_PRIME,
    ) -> dict[str, Any]:
        """Return copied arguments accepted directly by ``phi_lambda``."""
        return {
            "data": self.scores_array(),
            "cut": _finite_cut(cut),
            "n_i_prime": _normalized_n_i_prime(n_i_prime),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content and deterministic G-theory identities."""
        return {
            **self._content_dict(),
            "design_id": self.design_id,
            "design_fingerprint": self.design_fingerprint,
        }


def build_gtheory_pi_pilot_design(
    records: Iterable[PilotObservationRecord],
) -> GTheoryPiPilotDesign:
    """Build a complete one-rater, one-occasion ``p x i`` pilot handoff.

    The many-facet pilot assembler remains the fail-closed source of truth for
    provenance, category, duplicate-cell, observed-support, and dense-allocation
    validation. The resulting design is narrowed to the exact complete balanced
    contract required by the existing Rust-backed :func:`gtheory_pi` and
    :func:`phi_lambda` APIs.
    """
    facets_design = build_facets_pilot_design(records)
    if len(facets_design.rater_ids) != 1:
        raise PilotObservationError(
            "gtheory_pi_rater_confounded",
            "$.records",
            "one-facet G theory requires exactly one observed rater",
        )
    occasion_ids = tuple(
        sorted({entry.occasion_id for entry in facets_design.item_provenance})
    )
    if len(occasion_ids) != 1:
        raise PilotObservationError(
            "gtheory_pi_occasion_confounded",
            "$.records",
            "one-facet G theory requires exactly one declared occasion",
        )
    return GTheoryPiPilotDesign(
        facets_design=facets_design,
        rater_id=facets_design.rater_ids[0],
        occasion_id=occasion_ids[0],
        schema_version=facets_design.schema_version,
        _design_token=_GTHEORY_PI_DESIGN_TOKEN,
    )
