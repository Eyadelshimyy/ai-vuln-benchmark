"""Contracts for the binary-response measurement bounded context."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.measurement.binary_response import (
    BINARY_RESPONSE_CONTRACT_ID,
    BinaryResponseCell,
    BinaryResponseContractError,
    BinaryResponseMatrix,
    BinaryResponseState,
    build_binary_response_cell,
    build_binary_response_matrix,
)


class _ForgedBinaryResponseCell(BinaryResponseCell):
    """Bypass the base post-init seal to exercise exact aggregate admission."""

    def __post_init__(self, _admission_token: object | None) -> None:
        pass


class _ForgedBoundaryString(str):
    """Lie about trimming so subtype callbacks cannot redefine wire identity."""

    def strip(self, chars: str | None = None) -> str:
        return self


def _cell(
    state: BinaryResponseState | str = BinaryResponseState.OBSERVED,
    value: int | None = 1,
    *,
    observation_ref: str = "observation_alpha",
    adjudication_ref: str | None = None,
) -> BinaryResponseCell:
    """Build one value object through the public admission boundary."""
    return build_binary_response_cell(
        state=state,
        value=value,
        observation_ref=observation_ref,
        adjudication_ref=adjudication_ref,
    )


def test_nonresponse_states_are_distinct_from_binary_values() -> None:
    """Every nonresponse state remains explicit and cannot acquire category zero."""
    states = (
        BinaryResponseState.MISSING,
        BinaryResponseState.NOT_OBSERVED,
        BinaryResponseState.ABSTAINED,
        BinaryResponseState.INVALID,
        BinaryResponseState.OMITTED,
        BinaryResponseState.NOT_APPLICABLE,
        BinaryResponseState.INSUFFICIENT_EVIDENCE,
    )

    assert len({state.value for state in states}) == len(states)
    for state in states:
        cell = _cell(state=state, value=None)
        assert cell.state is state
        assert cell.value is None
        assert cell.to_dict()["state"] == state.value
        with pytest.raises(BinaryResponseContractError) as caught:
            _cell(state=state, value=0)
        assert caught.value.code == "nonresponse_has_value"

    with pytest.raises(BinaryResponseContractError) as caught:
        _cell(
            state=BinaryResponseState.OMITTED,
            value=None,
            adjudication_ref="adjudication_alpha",
        )
    assert caught.value.code == "unexpected_adjudication_ref"


def test_observed_cells_admit_only_exact_zero_or_one() -> None:
    """The observed channel is dichotomous and never thresholds other carriers."""
    assert _cell(value=0).value == 0
    assert _cell(value=1).value == 1
    assert _cell(state="observed", value=1).state is BinaryResponseState.OBSERVED

    for invalid in (None, True, False, 0.0, 1.0, 2, -1):
        with pytest.raises(BinaryResponseContractError) as caught:
            _cell(value=invalid)  # type: ignore[arg-type]
        assert caught.value.code == "invalid_binary_value"


def test_adjudicated_cells_keep_binary_value_and_provenance_separate() -> None:
    """An adjudicated value is scoreable only when its adjudication provenance exists."""
    cell = _cell(
        state=BinaryResponseState.ADJUDICATED,
        value=1,
        adjudication_ref="adjudication_alpha",
    )
    assert cell.value == 1
    assert cell.state is BinaryResponseState.ADJUDICATED
    assert cell.adjudication_ref == "adjudication_alpha"

    with pytest.raises(BinaryResponseContractError) as caught:
        _cell(state=BinaryResponseState.ADJUDICATED, value=1)
    assert caught.value.code == "missing_adjudication_ref"

    with pytest.raises(BinaryResponseContractError) as caught:
        _cell(value=1, adjudication_ref="adjudication_alpha")
    assert caught.value.code == "unexpected_adjudication_ref"

    with pytest.raises(BinaryResponseContractError) as caught:
        _cell(
            state=BinaryResponseState.ADJUDICATED,
            value=2,
            adjudication_ref="adjudication_alpha",
        )
    assert caught.value.code == "invalid_binary_value"


def test_cell_references_fail_closed_without_silent_normalization() -> None:
    """Opaque evidence references reject ambiguous boundary text instead of stripping it."""
    for invalid in (
        "",
        " observation_alpha",
        "observation_alpha ",
        "line\nbreak",
        "\ud800",
        "x" * 257,
    ):
        with pytest.raises(BinaryResponseContractError) as caught:
            _cell(observation_ref=invalid)
        assert caught.value.code == "invalid_reference"

    with pytest.raises(TypeError, match="observation_ref must be a string"):
        _cell(observation_ref=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="observation_ref must be a string"):
        _cell(observation_ref=_ForgedBoundaryString(" observation_alpha"))

    with pytest.raises(BinaryResponseContractError) as caught:
        build_binary_response_cell(
            state="unknown",
            value=None,
            observation_ref="observation_alpha",
        )
    assert caught.value.code == "invalid_response_state"

    with pytest.raises(TypeError, match="state must be"):
        build_binary_response_cell(
            state=object(),  # type: ignore[arg-type]
            value=None,
            observation_ref="observation_alpha",
        )

    with pytest.raises(ValueError, match="build_binary_response_cell"):
        BinaryResponseCell(
            state=BinaryResponseState.OBSERVED,
            value=1,
            observation_ref="observation_alpha",
        )


def test_matrix_preserves_values_states_and_provenance_without_dichotomizing() -> None:
    """Marshalling maps only absent values to NaN while retaining state evidence."""
    matrix = build_binary_response_matrix(
        (
            (
                _cell(value=1, observation_ref="observation_alpha"),
                _cell(
                    state=BinaryResponseState.ADJUDICATED,
                    value=0,
                    observation_ref="observation_beta",
                    adjudication_ref="adjudication_beta",
                ),
            ),
            (
                _cell(value=0, observation_ref="observation_gamma"),
                _cell(
                    state=BinaryResponseState.ABSTAINED,
                    value=None,
                    observation_ref="observation_delta",
                ),
            ),
        )
    )

    assert matrix.contract_id == BINARY_RESPONSE_CONTRACT_ID
    assert matrix.values == ((1, 0), (0, None))
    assert matrix.states == (
        (BinaryResponseState.OBSERVED, BinaryResponseState.ADJUDICATED),
        (BinaryResponseState.OBSERVED, BinaryResponseState.ABSTAINED),
    )
    assert matrix.adjudication_refs == (
        (None, "adjudication_beta"),
        (None, None),
    )
    assert matrix.observation_refs[1][1] == "observation_delta"

    numeric = matrix.responses_array()
    assert numeric.dtype == np.float64
    assert numeric.shape == (2, 2)
    assert numeric[0, 0] == 1.0
    assert numeric[0, 1] == 0.0
    assert numeric[1, 0] == 0.0
    assert np.isnan(numeric[1, 1])
    numeric[0, 0] = 0.0
    assert matrix.responses_array()[0, 0] == 1.0

    payload = matrix.to_dict()
    assert payload["contract_id"] == BINARY_RESPONSE_CONTRACT_ID
    assert payload["rows"][0][1]["state"] == "adjudicated"
    assert payload["rows"][1][1]["value"] is None


def test_matrix_rejects_shape_type_and_resource_violations() -> None:
    """The aggregate is sealed, rectangular, non-empty, typed, and allocation-bounded."""
    valid = _cell()

    with pytest.raises(ValueError, match="build_binary_response_matrix"):
        BinaryResponseMatrix(rows=((valid,),))
    with pytest.raises(BinaryResponseContractError) as caught:
        build_binary_response_matrix(())
    assert caught.value.code == "empty_response_matrix"
    with pytest.raises(TypeError, match="rows must be a tuple"):
        build_binary_response_matrix([(valid,)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"rows\[0\] must be a tuple"):
        build_binary_response_matrix(([valid],))  # type: ignore[list-item]
    with pytest.raises(BinaryResponseContractError) as caught:
        build_binary_response_matrix(((),))
    assert caught.value.code == "empty_response_row"
    with pytest.raises(TypeError, match=r"rows\[0\]\[0\]"):
        build_binary_response_matrix(((object(),),))  # type: ignore[arg-type]
    with pytest.raises(BinaryResponseContractError) as caught:
        build_binary_response_matrix(((valid,), (valid, valid)))
    assert caught.value.code == "nonrectangular_response_matrix"

    forged = _ForgedBinaryResponseCell(
        state=BinaryResponseState.OBSERVED,
        value=2,
        observation_ref="observation_forged",
    )
    with pytest.raises(TypeError, match=r"rows\[0\]\[0\]"):
        build_binary_response_matrix(((forged,),))

    oversized_row = tuple(valid for _ in range(1_001))
    oversized_rows = tuple(oversized_row for _ in range(1_000))
    with pytest.raises(BinaryResponseContractError) as caught:
        build_binary_response_matrix(oversized_rows)
    assert caught.value.code == "response_cell_budget_exceeded"
