"""Resource-bound regressions for cognitive-diagnosis evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.cdm as cdm
from fast_mlsirm import fit_cdm
from fast_mlsirm._cdm_response_safety import _reject_untrusted_response_container


MAX_CDM_EVIDENCE_CELLS = 20_000_000


def _unexpected_asarray(*args, **kwargs):
    """Fail if NumPy materialization runs before the CDM resource gate."""

    raise AssertionError("NumPy materialization executed before CDM resource admission")


def _q_matrix() -> np.ndarray:
    """Return a minimal valid item-by-attribute design."""

    return np.array([[1], [1]], dtype=np.int8)


def test_oversized_exact_response_array_fails_before_numpy_materialization(monkeypatch):
    """A huge broadcast response view is rejected without a dense copy."""

    responses = np.broadcast_to(
        np.array([[0, 1]], dtype=np.int8),
        (MAX_CDM_EVIDENCE_CELLS // 2 + 1, 2),
    )
    monkeypatch.setattr(cdm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 20000000-cell CDM evidence budget",
    ):
        fit_cdm(responses, _q_matrix())


def test_oversized_nested_numpy_response_leaf_fails_before_materialization(monkeypatch):
    """Nested exact NumPy leaves count toward the logical response budget."""

    leaf = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (MAX_CDM_EVIDENCE_CELLS + 1,),
    )
    monkeypatch.setattr(cdm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 20000000-cell CDM evidence budget",
    ):
        fit_cdm([leaf], _q_matrix())


def test_oversized_nested_numpy_q_leaf_fails_before_materialization(monkeypatch):
    """Q-matrix leaves are bounded before NumPy can stack the sequence."""

    q_leaf = np.broadcast_to(
        np.array([1], dtype=np.int8),
        (MAX_CDM_EVIDENCE_CELLS + 1,),
    )
    monkeypatch.setattr(cdm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"q_matrix exceeds the 20000000-cell CDM evidence budget",
    ):
        cdm._validate_q_matrix_input([q_leaf], "q_matrix", 1)


def test_cdm_resource_preflight_preserves_small_shared_sequence_dag():
    """Logical repeated rows remain valid while true cycles stay fail-closed."""

    shared_row = [0, 1]
    _reject_untrusted_response_container([shared_row, shared_row])


def test_shared_nested_dag_hits_logical_budget_without_exponential_retraversal():
    """Shared nested containers are accounted by multiplicity without rewalking them."""

    evidence: object = [0]
    for _ in range(25):
        evidence = [evidence, evidence]

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 20000000-cell CDM evidence budget",
    ):
        _reject_untrusted_response_container(evidence)


def test_boolean_round_trip_does_not_require_equal_nan(monkeypatch):
    """Boolean response admission remains compatible with the declared NumPy floor."""

    original_array_equal = np.array_equal

    def guarded_array_equal(left, right, *, equal_nan=False):
        if equal_nan and np.asarray(left).dtype.kind == "b":
            raise AssertionError("equal_nan must not be used for boolean arrays")
        return original_array_equal(left, right, equal_nan=equal_nan)

    monkeypatch.setattr(cdm.np, "array_equal", guarded_array_equal)
    observed = cdm._response_array(np.array([[True, False]], dtype=np.bool_))

    assert observed.dtype == np.float64
    assert observed.tolist() == [[1.0, 0.0]]
