"""Container/scalar callback regressions for answer-copying evidence."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fast_mlsirm.security import gbt, k_index


class _HostileFloat(float):
    calls = 0

    def __float__(self):
        type(self).calls += 1
        raise AssertionError("caller __float__ executed")


class _HostileList(list):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("caller __iter__ executed")

    def __len__(self):
        type(self).calls += 1
        raise AssertionError("caller __len__ executed")

    def __getitem__(self, index):
        type(self).calls += 1
        raise AssertionError("caller __getitem__ executed")


def _core_forbidden():
    raise AssertionError("compiled-core discovery reached invalid evidence")


def test_gbt_rejects_numeric_subclasses_before_conversion_callbacks():
    _HostileFloat.calls = 0
    matches = [_HostileFloat(1.0), 0.0, 1.0, 0.0]
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_core_forbidden):
        with pytest.raises(ValueError, match="matches must be an integer or float array"):
            gbt(matches, [0.25, 0.5, 0.75, 0.5])
    assert _HostileFloat.calls == 0


def test_k_index_rejects_container_subclasses_before_sequence_callbacks():
    _HostileList.calls = 0
    responses = _HostileList([[1, 0, 1], [0, 0, 0], [1, 1, 0]])
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_core_forbidden):
        with pytest.raises(ValueError, match="responses must be an integer or float array"):
            k_index(responses, 0, 1)
    assert _HostileList.calls == 0


def test_k_index_rejects_cyclic_builtin_sequences_before_numpy_materialization():
    responses = []
    responses.append(responses)
    with patch("fast_mlsirm.fitstats._core_module", side_effect=_core_forbidden):
        with pytest.raises(ValueError, match="responses must be an integer or float array"):
            k_index(responses, 0, 1)


def test_k_index_preserves_acyclic_shared_builtin_rows():
    row = [1, 0, 1]
    responses = [row, row, [0, 0, 0]]
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError, match="requires the compiled Rust core"):
            k_index(responses, 0, 1)
