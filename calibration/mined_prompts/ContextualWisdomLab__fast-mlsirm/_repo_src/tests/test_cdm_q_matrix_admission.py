"""Callback-safety regressions for cognitive-diagnosis Q-matrix evidence."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import fit_cdm, fit_gdina


class _HostileInt(int):
    """Caller-defined numeric identity that must not be coerced by validation."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller integer conversion executed")

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller float conversion executed")


class _HostileList(list):
    """Caller-defined container identity whose iteration must remain untouched."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("caller container iteration executed")


def _unexpected_core_discovery():
    """Fail if invalid Q-matrix evidence reaches compiled-core discovery."""

    raise AssertionError("compiled core must not be discovered for invalid Q evidence")


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_q_matrix_numeric_subclasses_without_callbacks(monkeypatch, fit):
    """Caller numeric subclasses cannot synthesize Q-matrix cells."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileInt.reset()
    q_matrix = [[_HostileInt(1)], [1]]
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)

    with pytest.raises(ValueError, match="q_matrix must be a trusted NumPy array or built-in sequence"):
        fit(responses, q_matrix)

    assert _HostileInt.calls == 0


@pytest.mark.parametrize("fit", [fit_cdm, fit_gdina])
def test_cdm_fits_reject_q_matrix_container_subclasses_without_callbacks(monkeypatch, fit):
    """Caller container subclasses cannot execute iteration while Q evidence is admitted."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    _HostileList.reset()
    q_matrix = _HostileList([[1], [1]])
    responses = np.array([[0, 1], [1, 0]], dtype=np.int8)

    with pytest.raises(ValueError, match="q_matrix must be a trusted NumPy array or built-in sequence"):
        fit(responses, q_matrix)

    assert _HostileList.calls == 0


def test_cdm_reload_repairs_q_matrix_guard_before_validation():
    """Direct module reload cannot reactivate callback-bearing Q materialization."""

    script = r'''
import importlib
import numpy as np
import fast_mlsirm.cdm as cdm
import fast_mlsirm.fitstats as fitstats


class ArraySentinel:
    calls = 0

    def __array__(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("caller Q-matrix array protocol executed")


def missing_core():
    return None


fitstats._core_module = missing_core
reloaded = importlib.reload(cdm)
responses = np.array([[0, 1], [1, 0]], dtype=np.int8)

try:
    reloaded.fit_cdm(responses, ArraySentinel())
except ValueError as exc:
    assert "q_matrix must be a trusted NumPy array or built-in sequence" in str(exc)
else:
    raise AssertionError("callback-bearing Q-matrix provider was accepted after reload")
assert ArraySentinel.calls == 0

try:
    reloaded.fit_cdm(responses, [[np.bool_(True)], [np.int8(1)]])
except RuntimeError as exc:
    assert "requires the compiled Rust core" in str(exc)
else:
    raise AssertionError("trusted Q-matrix evidence did not reach native dispatch")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
