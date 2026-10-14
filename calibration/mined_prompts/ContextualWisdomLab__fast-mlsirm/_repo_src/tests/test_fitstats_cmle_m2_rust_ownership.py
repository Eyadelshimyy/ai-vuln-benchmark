"""Fail-first ownership contracts for public conditional-Rasch M2."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats


def _complete_rasch_fixture() -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    """Return a full-pattern Rasch fixture with every raw-score category."""
    patterns = np.array(
        [
            [(pattern >> item) & 1 for item in range(5)]
            for pattern in range(1 << 5)
        ],
        dtype=np.float64,
    )
    responses = np.tile(patterns, (20, 1))
    factor_id = np.zeros(responses.shape[1], dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(responses.shape[1], dtype=np.float64),
        b=np.zeros(responses.shape[1], dtype=np.float64),
        zeta=np.zeros((responses.shape[1], 1), dtype=np.float64),
        tau=-30.0,
        theta=np.zeros((responses.shape[0], 1), dtype=np.float64),
        xi=np.zeros((responses.shape[0], 1), dtype=np.float64),
    )
    return responses, factor_id, params


def test_public_cmle_m2_fails_closed_without_rust_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public conditional-Rasch path must not execute Python M2 arithmetic."""
    responses, factor_id, params = _complete_rasch_fixture()
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fitstats.m2_cmle_rasch(responses, params.b)

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        fitstats.m2(responses, factor_id, params, "MIRT", estimator="cmle")


def test_public_cmle_m2_delegates_every_result_field_to_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python may validate and marshal but must not recompute the CMLE M2 result."""
    responses, factor_id, params = _complete_rasch_fixture()
    payload = {
        "m2": 12.5,
        "df": 6.0,
        "p_value": 0.051,
        "rmsea2": 0.025,
        "rmsea2_ci_lower": 0.001,
        "rmsea2_ci_upper": 0.049,
        "srmsr": 0.031,
        "null_m2": 44.0,
        "null_df": 10.0,
        "cfi": 0.97,
        "tli": 0.95,
        "n_moments": 15,
        "n_parameters": 9,
        "n_complete": responses.shape[0],
    }

    class RecordingCore:
        """Return a sentinel Rust payload and retain the delegated inputs."""

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def m2_cmle_rasch_stat(self, *args: object, **kwargs: object) -> dict[str, object]:
            """Record one native call and return the sentinel result."""
            self.calls.append((args, kwargs))
            return dict(payload)

    core = RecordingCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    direct = fitstats.m2_cmle_rasch(responses, params.b)
    routed = fitstats.m2(
        responses,
        factor_id,
        params,
        "MIRT",
        estimator="cmle",
    )

    assert len(core.calls) == 2
    for result in (direct, routed):
        for field_name, expected in payload.items():
            assert getattr(result, field_name) == expected
        assert result.estimator == "cmle"
        assert "conditional Rasch M2" in result.inference_note
