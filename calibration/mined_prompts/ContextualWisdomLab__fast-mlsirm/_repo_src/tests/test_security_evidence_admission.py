"""Trust-boundary regressions for answer-copying evidence admission."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.security import gbt, k_index, k_variants, wollack_omega


class _HostileArrayProvider:
    """Array provider that records forbidden protocol execution."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def __array__(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        raise AssertionError("caller __array__ executed")


class _ArraySubclass(np.ndarray):
    """Caller-owned ndarray identity that must not be trusted implicitly."""


def _provider(payload):
    return _HostileArrayProvider(payload)


def _core_forbidden():
    raise AssertionError("compiled-core discovery reached invalid evidence")


def _responses():
    return np.array(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int8,
    )


def _wollack_probs():
    return np.full((4, 3), 1.0 / 3.0)


@pytest.mark.parametrize(
    ("make_call", "payload"),
    [
        (
            lambda bad: wollack_omega(
                bad,
                np.array([0, 1, 2, 0]),
                _wollack_probs(),
                3,
            ),
            np.array([0, 1, 2, 0]),
        ),
        (
            lambda bad: wollack_omega(
                np.array([0, 1, 2, 0]),
                np.array([0, 1, 0, 2]),
                bad,
                3,
            ),
            _wollack_probs(),
        ),
        (lambda bad: k_index(bad, 0, 1), _responses()),
        (
            lambda bad: gbt(bad, np.array([0.25, 0.5, 0.75, 0.5])),
            np.array([1, 0, 1, 0]),
        ),
        (
            lambda bad: gbt(np.array([1, 0, 1, 0]), bad),
            np.array([0.25, 0.5, 0.75, 0.5]),
        ),
        (lambda bad: k_variants(bad, 0, 1), _responses()),
    ],
)
def test_answer_copying_rejects_arbitrary_array_providers_before_callbacks(
    make_call,
    payload,
):
    bad = _provider(payload)
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_core_forbidden):
        with pytest.raises(ValueError):
            make_call(bad)
    assert bad.calls == 0


@pytest.mark.parametrize(
    ("make_call", "payload"),
    [
        (
            lambda bad: wollack_omega(
                bad,
                np.array([0, 1, 2, 0]),
                _wollack_probs(),
                3,
            ),
            np.array([0, 1, 2, 0]),
        ),
        (lambda bad: k_index(bad, 0, 1), _responses()),
        (
            lambda bad: gbt(bad, np.array([0.25, 0.5, 0.75, 0.5])),
            np.array([1, 0, 1, 0]),
        ),
        (lambda bad: k_variants(bad, 0, 1), _responses()),
    ],
)
def test_answer_copying_rejects_ndarray_subclasses(make_call, payload):
    bad = payload.view(_ArraySubclass)
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_core_forbidden):
        with pytest.raises(ValueError):
            make_call(bad)


class _FakeCore:
    def py_wollack_omega(self, copier, source, probs, n_options):
        assert copier == [0, 1, 2, 0]
        assert source == [0, 1, 0, 2]
        assert probs.shape == (12,)
        assert n_options == 3
        return {
            "observed_matches": 2,
            "expected_matches": 4.0 / 3.0,
            "variance": 1.0,
            "omega": 2.0 / 3.0,
            "p_value": 0.25,
        }

    def py_k_index(self, flat, n_persons, n_items, copier, source):
        assert flat.shape == (12,)
        assert (n_persons, n_items, copier, source) == (3, 4, 0, 1)
        return {
            "wc": 2,
            "ws": 4,
            "m": 2,
            "subgroup": [0],
            "emp_agg": [2],
            "p": 0.5,
            "k_index": 0.25,
        }

    def py_gbt(self, matches, probs):
        assert matches.tolist() == [1.0, 0.0, 1.0, 0.0]
        assert probs.tolist() == [0.25, 0.5, 0.75, 0.5]
        return {
            "observed_matches": 2,
            "match_dist": [0.1, 0.2, 0.4, 0.2, 0.1],
            "p_value": 0.7,
        }

    def py_k_variants(self, flat, n_persons, n_items, copier, source):
        assert flat.shape == (12,)
        assert (n_persons, n_items, copier, source) == (3, 4, 0, 1)
        return {
            "wc": 2,
            "ws": 4,
            "m": 2,
            "mm": 2,
            "pr": [0.0, 0.5],
            "pj": [0.0, 0.5],
            "p1": 0.5,
            "p2": 0.5,
            "s1": 2.0,
            "s2": 2.0,
            "k1": 0.25,
            "k2": 0.25,
            "s1_index": 0.25,
            "s2_index": 0.25,
        }


def test_answer_copying_preserves_trusted_builtin_and_numpy_scalar_sequences():
    copier = [np.int8(0), np.int16(1), np.uint8(2), 0]
    source = (np.int8(0), np.int16(1), np.uint8(0), 2)
    probs = [
        [np.float32(1.0 / 3.0), np.float64(1.0 / 3.0), 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ]
    responses = [
        [np.int8(1), np.uint8(0), 1.0, np.float32(0.0)],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
    ]
    matches = (np.int8(1), np.uint8(0), 1.0, np.float32(0.0))
    match_probs = [np.float32(0.25), np.float64(0.5), 0.75, 0.5]

    with patch("fast_mlsirm.fitstats._core_module", return_value=_FakeCore()):
        assert wollack_omega(copier, source, probs, np.int16(3)).observed_matches == 2
        assert k_index(responses, np.int8(0), np.int16(1)).ws == 4
        assert gbt(matches, match_probs).observed_matches == 2
        assert k_variants(responses, np.int8(0), np.int16(1)).ws == 4
