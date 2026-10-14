"""Coverage for DETECT and DIMTEST dimensionality analysis (detect.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import fast_mlsirm.detect as detect_module
from fast_mlsirm.detect import DetectResult, DimtestResult, detect_analysis, dimtest


def _binary(seed=0, n_persons=30, n_items=6):
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(float)


def test_detect_happy_path_integer_cluster():
    res = detect_analysis(_binary(), np.array([0, 0, 0, 1, 1, 1]))
    assert isinstance(res, DetectResult)
    assert res.n_pairs == 15
    assert res.pair_i.shape == res.pair_j.shape


def test_detect_happy_path_unsigned_cluster():
    res = detect_analysis(_binary(), np.array([0, 0, 0, 1, 1, 1], dtype=np.uint64))
    assert isinstance(res, DetectResult)


def test_detect_happy_path_valid_float_cluster():
    res = detect_analysis(_binary(), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    assert isinstance(res, DetectResult)


def test_detect_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            detect_analysis(_binary(), np.array([0, 0, 0, 1, 1, 1]))


def test_detect_rejects_non_2d():
    with pytest.raises(ValueError):
        detect_analysis(np.zeros(6), np.array([0, 1]))


def test_detect_rejects_too_small():
    with pytest.raises(ValueError):
        detect_analysis(np.zeros((1, 6)), np.zeros(6, dtype=np.int64))
    with pytest.raises(ValueError):
        detect_analysis(np.zeros((5, 1)), np.zeros(1, dtype=np.int64))


def test_detect_rejects_missing_values():
    y = _binary()
    y[0, 0] = np.nan
    with pytest.raises(ValueError):
        detect_analysis(y, np.array([0, 0, 0, 1, 1, 1]))


def test_detect_rejects_cluster_length_mismatch():
    with pytest.raises(ValueError):
        detect_analysis(_binary(), np.array([0, 1, 0]))


def test_detect_rejects_non_integer_float_cluster():
    with pytest.raises(ValueError):
        detect_analysis(_binary(), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.5]))


def test_detect_rejects_float_cluster_out_of_i64_range():
    with pytest.raises(ValueError):
        detect_analysis(
            _binary(), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0**63])
        )


def test_detect_rejects_unsigned_cluster_over_i64():
    bad = np.array([0, 0, 0, 1, 1, 2**63], dtype=np.uint64)
    with pytest.raises(ValueError):
        detect_analysis(_binary(), bad)


def test_detect_rejects_self_referential_response_list():
    responses = []
    responses.append(responses)
    with pytest.raises(ValueError):
        detect_analysis(responses, np.array([0, 0, 0, 1, 1, 1]))


def test_detect_rejects_indirectly_cyclic_cluster_list():
    inner = []
    outer = [inner]
    inner.append(outer)
    with pytest.raises(ValueError):
        detect_analysis(_binary(), outer)


def test_detect_preserves_shared_acyclic_response_rows():
    responses = _binary().tolist()
    responses[1] = responses[0]

    result = detect_analysis(responses, np.array([0, 0, 0, 1, 1, 1]))

    assert isinstance(result, DetectResult)
    assert result.n_pairs == 15


def test_detect_bounds_compressed_shared_dag_before_numpy_materialization(monkeypatch):
    leaf = [0.0, 1.0]
    responses = leaf
    for _ in range(6):
        responses = [responses, responses]

    monkeypatch.setattr(detect_module, "_MAX_DETECT_SEQUENCE_CELLS", 64, raising=False)

    def _materialization_must_not_run(*args, **kwargs):
        raise AssertionError("NumPy materialization ran before shared-DAG resource admission")

    monkeypatch.setattr(detect_module.np, "asarray", _materialization_must_not_run)

    with pytest.raises(ValueError, match="numeric array"):
        detect_module._trusted_numeric_array(
            responses,
            numeric_error="responses must be a numeric array",
            complex_error="responses must be real-valued",
        )


def test_dimtest_happy_path():
    y = _binary(n_persons=300, n_items=15)
    res = dimtest(y, at1=np.array([0, 1, 2, 3]), at2=np.array([4, 5, 6, 7]))
    assert isinstance(res, DimtestResult)
    assert res.groups_used >= 0


def test_dimtest_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            dimtest(_binary(), np.array([0, 1]), np.array([2, 3]))


def test_dimtest_rejects_complex_and_non_numeric():
    with pytest.raises(ValueError):
        dimtest(np.array([[1j, 0j]]), np.array([0]), np.array([1]))
    with pytest.raises(ValueError):
        dimtest(np.array([["a", "b"]]), np.array([0]), np.array([1]))


def test_dimtest_rejects_bad_response_shape_or_values():
    with pytest.raises(ValueError):
        dimtest(np.zeros(6), np.array([0]), np.array([1]))
    with pytest.raises(ValueError):
        dimtest(np.zeros((0, 4)), np.array([0]), np.array([1]))
    y = _binary()
    y[0, 0] = 2.0
    with pytest.raises(ValueError):
        dimtest(y, np.array([0, 1]), np.array([2, 3]))


def test_dimtest_index_set_validation():
    y = _binary(n_persons=40, n_items=8)
    with pytest.raises(ValueError):
        dimtest(y, np.array([0j, 1j]), np.array([2, 3]))
    with pytest.raises(ValueError):
        dimtest(y, np.array(["a", "b"]), np.array([2, 3]))
    with pytest.raises(ValueError):
        dimtest(y, np.array([]), np.array([2, 3]))
    with pytest.raises(ValueError):
        dimtest(y, np.array([0.5, 1.5]), np.array([2, 3]))
    with pytest.raises(ValueError):
        dimtest(y, np.array([0, 1, 2, 999]), np.array([3, 4, 5, 6]))
