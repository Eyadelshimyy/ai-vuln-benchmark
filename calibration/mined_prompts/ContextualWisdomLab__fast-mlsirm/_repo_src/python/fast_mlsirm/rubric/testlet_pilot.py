"""Provenance-safe binary testlet calibration handoff for pilot items."""

from __future__ import annotations

from collections import Counter
from dataclasses import InitVar, dataclass
import math
import sys
from typing import Any, Iterable

import numpy as np

from ..config import MAX_MAX_ITER
from .models import SCHEMA_VERSION, _schema_version, _sha256_hex
from .pilot_observations import (
    MirtPilotDesign,
    PilotItemProvenance,
    PilotObservationRecord,
    PilotResponseState,
    build_mirt_pilot_design,
)

_TESTLET_DESIGN_TOKEN = object()
_SUPPORTED_MODELS = ("rasch", "2pl")
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
    """Return whether ``value_type`` is one supported NumPy integer class."""
    return any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES)


def _is_exact_numpy_floating_scalar(value_type: type[Any]) -> bool:
    """Return whether ``value_type`` is one supported NumPy floating class."""
    return any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES)


def _normalized_integer(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    """Return one bounded trusted integer without caller protocol dispatch."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif _is_exact_numpy_integer_scalar(value_type):
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _normalized_float(value: Any, name: str) -> float:
    """Return one finite non-negative trusted numeric scalar."""
    value_type = type(value)
    if value_type is int or value_type is float:
        normalized = float(value)
    elif _is_exact_numpy_integer_scalar(value_type) or _is_exact_numpy_floating_scalar(
        value_type
    ):
        normalized = float(value)
    else:
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _normalized_bool(value: Any, name: str) -> bool:
    """Return one strict trusted boolean setting."""
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class TestletPilotDesign:
    """Content-addressed pilot handoff for the Rust-backed testlet model.

    The wrapped binary design maps each governed ``query_testlet_id`` to one
    integer testlet identifier. At least one testlet must contain two or more
    items, so an artifact labelled as a testlet design cannot silently encode
    only singleton item groups. Exact missingness and rater provenance remain
    available in the immutable nested design.

    This object is a calibration-input contract only. It does not establish
    connectedness, convergence, adequate fit, local dependence, scoreability,
    fairness, or validity.
    """

    __test__ = False

    binary_design: MirtPilotDesign
    schema_version: str = SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated public builder."""
        if _design_token is not _TESTLET_DESIGN_TOKEN:
            raise ValueError(
                "TestletPilotDesign must be created by build_testlet_pilot_design"
            )
        if not isinstance(self.binary_design, MirtPilotDesign):
            raise TypeError("binary_design must be a validated MirtPilotDesign")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.schema_version != self.binary_design.schema_version:
            raise ValueError("schema_version must match the wrapped binary design")
        testlet_counts = Counter(self.binary_design.item_factor_ids)
        if max(testlet_counts.values()) < 2:
            raise ValueError(
                "testlet design requires at least one query testlet with two or more items"
            )

    @property
    def pilot_study_id(self) -> str:
        """Return the pilot-study identifier retained by the binary design."""
        return self.binary_design.pilot_study_id

    @property
    def respondent_ids(self) -> tuple[str, ...]:
        """Return respondent identifiers in response-matrix row order."""
        return self.binary_design.respondent_ids

    @property
    def item_provenance(self) -> tuple[PilotItemProvenance, ...]:
        """Return immutable item provenance in response-matrix column order."""
        return self.binary_design.item_provenance

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return item identifiers in response-matrix column order."""
        return self.binary_design.item_ids

    @property
    def query_testlet_ids(self) -> tuple[str, ...]:
        """Return descriptive query-testlet identifiers in numeric-id order."""
        return self.binary_design.factor_testlet_ids

    @property
    def item_testlet_ids(self) -> tuple[int, ...]:
        """Return each item's zero-based testlet assignment."""
        return self.binary_design.item_factor_ids

    @property
    def responses(self) -> tuple[tuple[int | None, ...], ...]:
        """Return the immutable binary response matrix with explicit gaps."""
        return self.binary_design.responses

    @property
    def response_states(self) -> tuple[tuple[PilotResponseState, ...], ...]:
        """Return exact response states retained independently of numeric NaN."""
        return self.binary_design.response_states

    @property
    def rater_assignments(self) -> tuple[tuple[str | None, ...], ...]:
        """Return the provenance-only rater assignment for every matrix cell."""
        return self.binary_design.rater_assignments

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical design content without outer derived identities."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "query_testlet_ids": list(self.query_testlet_ids),
            "item_testlet_ids": list(self.item_testlet_ids),
            "binary_design": self.binary_design.to_dict(),
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete testlet handoff design."""
        return _sha256_hex(self._content_dict())

    @property
    def design_id(self) -> str:
        """Return a descriptive 128-bit public testlet-design handle."""
        return f"testlet_pilot_design_{self.design_fingerprint[:32]}"

    def responses_array(self) -> np.ndarray:
        """Return a fresh float matrix with non-observed states represented by NaN."""
        return self.binary_design.responses_array()

    def testlet_id_array(self) -> np.ndarray:
        """Return a fresh per-item integer testlet assignment vector."""
        return self.binary_design.factor_id_array()

    def to_fit_testlet_kwargs(
        self,
        *,
        model: str = "rasch",
        max_iter: int = 500,
        tol: float = 1e-6,
        q_gamma: int = 21,
        estimate_sigma: bool = True,
        init_sigma2: float = 0.5,
        require_convergence: bool = False,
    ) -> dict[str, Any]:
        """Return copied arguments accepted directly by ``fit_testlet``.

        Numerical settings remain caller-tunable, but this boundary validates
        them before producing an execution payload. The estimator itself keeps
        responsibility for Rust-core availability, convergence, fit, and the
        estimated testlet variances.
        """
        if type(model) is not str:
            raise ValueError("model must be 'rasch' or '2pl'")
        normalized_model = model.casefold()
        if normalized_model not in _SUPPORTED_MODELS:
            raise ValueError("model must be 'rasch' or '2pl'")
        normalized_max_iter = _normalized_integer(
            max_iter,
            "max_iter",
            minimum=1,
            maximum=MAX_MAX_ITER,
        )
        # #1929: no node-count cap; the Rust core generates any n >= 1 rule
        # on demand (Golub & Welsch, 1969) and guards allocation overflow.
        normalized_q_gamma = _normalized_integer(
            q_gamma,
            "q_gamma",
            minimum=1,
            maximum=sys.maxsize,
        )
        return {
            "responses": self.responses_array(),
            "testlet_id": self.testlet_id_array(),
            "model": normalized_model,
            "max_iter": normalized_max_iter,
            "tol": _normalized_float(tol, "tol"),
            "q_gamma": normalized_q_gamma,
            "estimate_sigma": _normalized_bool(estimate_sigma, "estimate_sigma"),
            "init_sigma2": _normalized_float(init_sigma2, "init_sigma2"),
            "require_convergence": _normalized_bool(
                require_convergence,
                "require_convergence",
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content and deterministic testlet identities."""
        return {
            **self._content_dict(),
            "design_id": self.design_id,
            "design_fingerprint": self.design_fingerprint,
        }


def build_testlet_pilot_design(
    records: Iterable[PilotObservationRecord],
) -> TestletPilotDesign:
    """Build a deterministic binary testlet handoff from pilot observations.

    The existing binary MIRT assembler remains the fail-closed source of truth
    for response, provenance, missingness, duplicate-cell, category, observed
    support, and dense-allocation validation. This builder reinterprets its
    disclosed ``query_testlet_id`` mapping only as the testlet grouping expected
    by :func:`fast_mlsirm.fit_testlet`; it performs no psychometric arithmetic.
    """
    binary_design = build_mirt_pilot_design(records)
    return TestletPilotDesign(
        binary_design=binary_design,
        schema_version=binary_design.schema_version,
        _design_token=_TESTLET_DESIGN_TOKEN,
    )
