"""Compatibility regression for Boolean Wollack probability evidence."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from fast_mlsirm.security import wollack_omega


class _BoolProbabilityCore:
    """Fake Rust boundary that records normalized Boolean probabilities."""

    def py_wollack_omega(self, copier, source, probs, n_options):
        assert copier == [0, 1, 2, 0]
        assert source == [0, 1, 0, 2]
        assert n_options == 3
        assert probs.tolist() == [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        return {
            "observed_matches": 2,
            "expected_matches": 1.0,
            "variance": 1.0,
            "omega": 1.0,
            "p_value": 0.25,
        }


def test_wollack_preserves_numpy_bool_scalars_in_builtin_probability_sequences():
    """``allow_bool`` must cover concrete NumPy Boolean scalar leaves too."""
    probs = [
        [np.bool_(True), np.bool_(False), np.bool_(False)],
        [np.bool_(False), np.bool_(True), np.bool_(False)],
        [np.bool_(True), np.bool_(False), np.bool_(False)],
        [np.bool_(False), np.bool_(False), np.bool_(True)],
    ]
    with patch("fast_mlsirm.fitstats._core_module", return_value=_BoolProbabilityCore()):
        result = wollack_omega([0, 1, 2, 0], [0, 1, 0, 2], probs, 3)
    assert result.observed_matches == 2
