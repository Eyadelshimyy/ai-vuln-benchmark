"""Fail-first ownership contracts for public S-X² and person-fit statistics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats_module
from fast_mlsirm.fitstats import person_fit, s_x2


def _fixture() -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    """Return a small valid dichotomous fixture for fit-stat ownership tests."""
    y = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    factor_id = np.zeros(y.shape[1], dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(y.shape[1], dtype=np.float64),
        b=np.linspace(-0.5, 0.5, y.shape[1]),
        zeta=np.zeros((y.shape[1], 1), dtype=np.float64),
        tau=-30.0,
        theta=np.linspace(-1.0, 1.0, y.shape[0])[:, None],
        xi=np.zeros((y.shape[0], 1), dtype=np.float64),
    )
    return y, factor_id, params


def test_sx2_missing_core_fails_before_python_reference_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary public S-X² must not silently select the Python numerical reference."""
    y, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: None)
    monkeypatch.setattr(
        fitstats_module,
        "_icc_grid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Python S-X2 reference grid executed")
        ),
    )

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        s_x2(y, factor_id, params, "MIRT", q_theta=7, q_xi=7)


def test_person_fit_missing_core_fails_before_python_numerics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary public person-fit must fail closed before NumPy probability arithmetic."""
    y, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: None)
    monkeypatch.setattr(
        fitstats_module.np,
        "exp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Python person-fit probability arithmetic executed")
        ),
    )

    with pytest.raises(RuntimeError, match="fit statistics require the compiled Rust core"):
        person_fit(y, factor_id, params, "MIRT")


def test_sx2_prior_mean_still_dispatches_to_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero trait prior must use the existing Rust S-X² entrypoint, not Python fallback."""
    y, factor_id, params = _fixture()
    prior_mean = np.array([0.35], dtype=np.float64)

    class RecordingCore:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def s_x2_stat(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            n_items = y.shape[1]
            return {
                "statistic": np.zeros(n_items),
                "g2_statistic": np.zeros(n_items),
                "df": np.ones(n_items),
                "p_value": np.ones(n_items),
                "g2_p_value": np.ones(n_items),
                "flagged_bh": np.zeros(n_items, dtype=bool),
                "n_score_groups": np.ones(n_items, dtype=np.int64),
                "rms_residual": np.zeros(n_items),
            }

    core = RecordingCore()
    monkeypatch.setattr(fitstats_module, "_core_module", lambda: core)
    monkeypatch.setattr(
        fitstats_module,
        "_icc_grid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Python S-X2 reference grid executed")
        ),
    )

    result = s_x2(
        y,
        factor_id,
        params,
        "MIRT",
        q_theta=7,
        q_xi=7,
        prior_mean=prior_mean,
    )

    assert result.statistic.shape == (y.shape[1],)
    assert len(core.calls) == 1
    args, _kwargs = core.calls[0]
    vector_args = [
        np.asarray(value)
        for value in args
        if isinstance(value, np.ndarray) and np.asarray(value).shape == prior_mean.shape
    ]
    assert any(np.array_equal(value, prior_mean) for value in vector_args)
