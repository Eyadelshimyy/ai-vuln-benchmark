"""Compatibility regressions for logistic DIF controls at the Rust boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm as fast_mlsirm
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import dif as dif_module
from fast_mlsirm.dif import logistic_dif, logistic_dif_purified


def _unexpected_core_discovery():
    """Fail if a rejected ``max_iter`` control reaches native-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


class _CaptureCore:
    """Capture normalized controls while returning Rust-shaped DIF evidence."""

    def __init__(self) -> None:
        self.logistic_controls: tuple[Any, ...] | None = None
        self.purified_controls: tuple[Any, ...] | None = None

    @staticmethod
    def _result() -> dict[str, Any]:
        return {
            "item": [0],
            "chi2_uniform": [0.0],
            "p_uniform": [1.0],
            "chi2_nonuniform": [0.0],
            "p_nonuniform": [1.0],
            "chi2_total": [0.0],
            "p_total": [1.0],
            "delta_r2": [0.0],
            "delta_r2_uniform": [0.0],
            "jg_class": ["A"],
            "flagged_bh": [False],
            "converged": [False],
        }

    def logistic_dif(self, yy, gg, n_persons, n_items, exclude, fdr_q, max_iter):
        self.logistic_controls = (exclude, fdr_q, max_iter)
        return self._result()

    def logistic_dif_purified(
        self,
        yy,
        gg,
        n_persons,
        n_items,
        exclude,
        fdr_q,
        max_iter,
        max_rounds,
        min_anchor_items,
    ):
        self.purified_controls = (
            exclude,
            fdr_q,
            max_iter,
            max_rounds,
            min_anchor_items,
        )
        return self._result() | {
            "anchor": [True],
            "n_anchor": 1,
            "rounds": 1,
            "purify_converged": True,
            "purify_termination_reason": "stable_flag_set",
        }


def test_logistic_dif_preserves_numpy_boolean_and_minimum_iteration_native_domain(monkeypatch):
    core = _CaptureCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    responses = np.array([[0], [1]], dtype=np.int8)
    group = np.array([0, 1], dtype=np.int8)

    logistic_dif(
        responses,
        group,
        exclude_studied_item=np.bool_(True),
        fdr_q=np.float32(0.05),
        max_iter=np.int16(1),
    )

    assert core.logistic_controls is not None
    exclude, fdr_q, max_iter = core.logistic_controls
    assert type(exclude) is bool and exclude is True
    assert type(fdr_q) is float
    assert type(max_iter) is int and max_iter == 1


def test_purified_logistic_preserves_minimum_iteration_native_domain(monkeypatch):
    core = _CaptureCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    responses = np.array([[0], [1]], dtype=np.int8)
    group = np.array([0, 1], dtype=np.int8)

    logistic_dif_purified(
        responses,
        group,
        max_iter=np.uint8(1),
        max_rounds=np.uint8(1),
        min_anchor_items=np.uint8(0),
    )

    assert core.purified_controls is not None
    exclude, fdr_q, max_iter, max_rounds, min_anchor_items = core.purified_controls
    assert type(exclude) is bool
    assert type(fdr_q) is float
    assert type(max_iter) is int and max_iter == 1
    assert type(max_rounds) is int and max_rounds == 1
    assert type(min_anchor_items) is int and min_anchor_items == 0


def test_logistic_dif_rejects_zero_max_iter_before_native_discovery(monkeypatch):
    """Zero iterations fail at the Python boundary, matching the Rust ``logistic_sweep`` domain."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        logistic_dif(
            np.array([[0], [1]], dtype=np.int8),
            np.array([0, 1], dtype=np.int8),
            max_iter=0,
        )


def test_purified_logistic_dif_rejects_zero_max_iter_before_native_discovery(monkeypatch):
    """Zero iterations fail at the Python boundary, matching the Rust ``logistic_sweep`` domain."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        logistic_dif_purified(
            np.array([[0], [1]], dtype=np.int8),
            np.array([0, 1], dtype=np.int8),
            max_iter=0,
        )


def test_package_aliases_share_hardened_dif_functions():
    assert fast_mlsirm.logistic_dif is dif_module.logistic_dif
    assert fast_mlsirm.mantel_haenszel_dif_purified is dif_module.mantel_haenszel_dif_purified
    assert fast_mlsirm.logistic_dif_purified is dif_module.logistic_dif_purified
