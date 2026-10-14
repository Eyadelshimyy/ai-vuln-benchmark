"""Versioned binary-response value objects and matrix marshalling.

This module owns response-state semantics at the measurement boundary. It does
not estimate IRT parameters or perform psychometric arithmetic; numerical model
work remains Rust-owned.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any

import numpy as np

BINARY_RESPONSE_CONTRACT_ID = "fast_mlsirm_binary_response/v1"
MAX_BINARY_RESPONSE_CELLS = 1_000_000
MAX_BINARY_RESPONSE_REFERENCE_CHARS = 256
_CELL_TOKEN = object()
_MATRIX_TOKEN = object()


class BinaryResponseState(str, Enum):
    """State of one dichotomous measurement observation."""

    OBSERVED = "observed"
    MISSING = "missing"
    NOT_OBSERVED = "not_observed"
    ABSTAINED = "abstained"
    INVALID = "invalid"
    OMITTED = "omitted"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ADJUDICATED = "adjudicated"


class BinaryResponseContractError(ValueError):
    """Stable fail-closed error for binary-response contract violations."""

    def __init__(self, code: str, path: str, message: str) -> None:
        """Retain bounded machine-readable rejection metadata."""
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


def _error(code: str, path: str, message: str) -> BinaryResponseContractError:
    return BinaryResponseContractError(code, path, message)


def _state(value: BinaryResponseState | str) -> BinaryResponseState:
    """Normalize a public response-state value without accepting arbitrary objects."""
    if type(value) is BinaryResponseState:
        return value
    if type(value) is not str:
        raise TypeError("state must be a BinaryResponseState or exact string")
    try:
        return BinaryResponseState(value)
    except ValueError as exc:
        raise _error(
            "invalid_response_state",
            "$.state",
            "state must identify a supported binary-response state",
        ) from exc


def _reference(value: str, path: str) -> str:
    """Validate one opaque reference without invoking caller string protocols."""
    if type(value) is not str:
        raise TypeError(f"{path} must be a string")
    if (
        not value
        or len(value) > MAX_BINARY_RESPONSE_REFERENCE_CHARS
        or value != value.strip()
        or any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise _error(
            "invalid_reference",
            path,
            "reference must be 1..256 Unicode scalar values without boundary "
            "whitespace or control characters",
        )
    return value


def _binary_value(value: int | None) -> int:
    """Admit only exact built-in integer categories zero and one."""
    if type(value) is not int or value not in (0, 1):
        raise _error(
            "invalid_binary_value",
            "$.value",
            "binary response value must be exact integer 0 or 1",
        )
    return value


@dataclass(frozen=True)
class BinaryResponseCell:
    """Immutable dichotomous response plus provenance-bearing state."""

    state: BinaryResponseState
    value: int | None
    observation_ref: str
    adjudication_ref: str | None = None
    _admission_token: InitVar[object | None] = None

    def __post_init__(self, _admission_token: object | None) -> None:
        """Prevent construction that bypasses the public invariant checker."""
        if _admission_token is not _CELL_TOKEN:
            raise ValueError("BinaryResponseCell must be created by build_binary_response_cell")

    def to_dict(self) -> dict[str, Any]:
        """Return the version-neutral cell payload for contract serialization."""
        return {
            "state": self.state.value,
            "value": self.value,
            "observation_ref": self.observation_ref,
            "adjudication_ref": self.adjudication_ref,
        }


def build_binary_response_cell(
    *,
    state: BinaryResponseState | str,
    value: int | None,
    observation_ref: str,
    adjudication_ref: str | None = None,
) -> BinaryResponseCell:
    """Build one binary response while preserving nonresponse and adjudication states."""
    normalized_state = _state(state)
    normalized_observation_ref = _reference(observation_ref, "$.observation_ref")

    if normalized_state in (BinaryResponseState.OBSERVED, BinaryResponseState.ADJUDICATED):
        normalized_value: int | None = _binary_value(value)
        if normalized_state is BinaryResponseState.ADJUDICATED:
            if adjudication_ref is None:
                raise _error(
                    "missing_adjudication_ref",
                    "$.adjudication_ref",
                    "adjudicated responses require an adjudication reference",
                )
            normalized_adjudication_ref = _reference(
                adjudication_ref,
                "$.adjudication_ref",
            )
        else:
            if adjudication_ref is not None:
                raise _error(
                    "unexpected_adjudication_ref",
                    "$.adjudication_ref",
                    "observed responses cannot carry adjudication provenance",
                )
            normalized_adjudication_ref = None
    else:
        if value is not None:
            raise _error(
                "nonresponse_has_value",
                "$.value",
                "nonresponse states must not be encoded as category 0 or 1",
            )
        if adjudication_ref is not None:
            raise _error(
                "unexpected_adjudication_ref",
                "$.adjudication_ref",
                "nonresponse states cannot carry adjudication provenance",
            )
        normalized_value = None
        normalized_adjudication_ref = None

    return BinaryResponseCell(
        state=normalized_state,
        value=normalized_value,
        observation_ref=normalized_observation_ref,
        adjudication_ref=normalized_adjudication_ref,
        _admission_token=_CELL_TOKEN,
    )


@dataclass(frozen=True)
class BinaryResponseMatrix:
    """Rectangular aggregate of versioned dichotomous response cells."""

    rows: tuple[tuple[BinaryResponseCell, ...], ...]
    _admission_token: InitVar[object | None] = None

    def __post_init__(self, _admission_token: object | None) -> None:
        """Prevent aggregate construction outside the bounded matrix builder."""
        if _admission_token is not _MATRIX_TOKEN:
            raise ValueError("BinaryResponseMatrix must be created by build_binary_response_matrix")

    @property
    def contract_id(self) -> str:
        """Return the immutable Published Language identity for this aggregate."""
        return BINARY_RESPONSE_CONTRACT_ID

    @property
    def values(self) -> tuple[tuple[int | None, ...], ...]:
        """Return binary values separately from their observation states."""
        return tuple(tuple(cell.value for cell in row) for row in self.rows)

    @property
    def states(self) -> tuple[tuple[BinaryResponseState, ...], ...]:
        """Return exact response states in matrix order."""
        return tuple(tuple(cell.state for cell in row) for row in self.rows)

    @property
    def observation_refs(self) -> tuple[tuple[str, ...], ...]:
        """Return exact observation provenance references in matrix order."""
        return tuple(tuple(cell.observation_ref for cell in row) for row in self.rows)

    @property
    def adjudication_refs(self) -> tuple[tuple[str | None, ...], ...]:
        """Return adjudication references independently of binary values."""
        return tuple(tuple(cell.adjudication_ref for cell in row) for row in self.rows)

    def responses_array(self) -> np.ndarray:
        """Marshal values to float64, representing only absent values as NaN."""
        return np.asarray(
            [
                [np.nan if cell.value is None else cell.value for cell in row]
                for row in self.rows
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible Published Language payload."""
        return {
            "contract_id": BINARY_RESPONSE_CONTRACT_ID,
            "rows": [[cell.to_dict() for cell in row] for row in self.rows],
        }


def build_binary_response_matrix(
    rows: tuple[tuple[BinaryResponseCell, ...], ...],
) -> BinaryResponseMatrix:
    """Assemble a bounded rectangular binary-response aggregate without coercion."""
    if not isinstance(rows, tuple):
        raise TypeError("rows must be a tuple")
    if not rows:
        raise _error(
            "empty_response_matrix",
            "$.rows",
            "binary response matrix must contain at least one row",
        )

    for row_index, row in enumerate(rows):
        if not isinstance(row, tuple):
            raise TypeError(f"rows[{row_index}] must be a tuple")

    width = len(rows[0])
    if width == 0:
        raise _error(
            "empty_response_row",
            "$.rows[0]",
            "binary response rows must contain at least one cell",
        )
    for row_index, row in enumerate(rows[1:], start=1):
        if len(row) != width:
            raise _error(
                "nonrectangular_response_matrix",
                f"$.rows[{row_index}]",
                "all binary response rows must have the same width",
            )

    cell_count = len(rows) * width
    if cell_count > MAX_BINARY_RESPONSE_CELLS:
        raise _error(
            "response_cell_budget_exceeded",
            "$.rows",
            f"binary response matrix requires {cell_count} cells; maximum is {MAX_BINARY_RESPONSE_CELLS}",
        )

    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            if type(cell) is not BinaryResponseCell:
                raise TypeError(
                    f"rows[{row_index}][{column_index}] must be a BinaryResponseCell"
                )

    return BinaryResponseMatrix(rows=rows, _admission_token=_MATRIX_TOKEN)
