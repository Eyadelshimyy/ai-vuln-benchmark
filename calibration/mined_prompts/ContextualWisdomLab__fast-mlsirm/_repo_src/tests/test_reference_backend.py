"""Contract tests for the explicit non-production fitting reference."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from fast_mlsirm import FitConfig, fit
from fast_mlsirm.backend import _reference_backend_scope
from fast_mlsirm.reference import fit_reference


class _HostileFitConfig(FitConfig):
    """FitConfig subclass that detects truth-value dispatch during admission."""

    def __bool__(self) -> bool:
        raise AssertionError("public fit dispatched caller-defined truthiness")


def test_reference_rejects_rust_only_plain_unidimensional_mmle() -> None:
    """The reference label must not disguise a Rust-only legacy MMLE path."""
    with pytest.raises(RuntimeError, match="NumPy reference is unavailable"):
        fit_reference(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0, 0], dtype=np.int64),
            FitConfig(model="ULS2PLM", estimator="mmle", max_iter=1),
        )


def test_public_fit_has_no_reference_authority_keyword() -> None:
    """Only the named reference API may activate the NumPy fitting authority."""
    assert "_allow_reference_backend" not in inspect.signature(fit).parameters

    with pytest.raises(TypeError, match="_allow_reference_backend"):
        fit(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0, 0], dtype=np.int64),
            FitConfig(backend="numpy", max_iter=1, n_restarts=1),
            _allow_reference_backend=True,
        )


def test_public_fit_rejects_config_subclass_before_truthiness_callback() -> None:
    """Production config admission must not invoke caller-defined truthiness."""
    with pytest.raises(ValueError, match="config must be a FitConfig or None"):
        fit(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([0, 0], dtype=np.int64),
            _HostileFitConfig(),
        )


def test_public_fit_validates_config_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid semantic controls must fail before compiled-core discovery."""

    def _unexpected_core_discovery() -> None:
        """Fail if the invalid configuration reaches native discovery."""
        raise AssertionError("native core discovery preceded config validation")

    monkeypatch.setattr("fast_mlsirm.backend._load_core", _unexpected_core_discovery)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        fit(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([0, 0], dtype=np.int64),
            FitConfig(max_iter=0),
        )


def test_public_fit_rejects_numpy_inside_reference_scope() -> None:
    """Importable reference scope cannot authorize NumPy through public fit."""
    with _reference_backend_scope():
        with pytest.raises(ValueError, match="production backend"):
            fit(
                np.array(
                    [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array([0, 0, 0, 0], dtype=np.int64),
                FitConfig(
                    model="MLS2PLM",
                    latent_dim=1,
                    max_iter=1,
                    n_restarts=1,
                    backend="numpy",
                ),
            )


def test_public_plain_mmle_auto_reports_resolved_rust_backend() -> None:
    """Successful production auto resolution records the concrete Rust owner."""
    pytest.importorskip("fast_mlsirm._core")
    responses = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = fit(
        responses,
        np.zeros(4, dtype=np.int64),
        FitConfig(
            model="ULS2PLM",
            estimator="mmle",
            max_iter=2,
            n_restarts=1,
            backend="auto",
        ),
    )

    assert result.backend == "rust"


def test_reference_rejects_invalid_config_before_dataclass_replace() -> None:
    """The reference API owns a stable validation error for invalid configs."""
    with pytest.raises(ValueError, match="config must be a FitConfig or None"):
        fit_reference(
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            np.array([0, 0], dtype=np.int64),
            object(),  # type: ignore[arg-type]
        )


def test_named_reference_fit_owns_numpy_scope() -> None:
    """The named reference API can run NumPy without a compiled Rust module."""
    result = fit_reference(
        np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
        np.array([0, 0, 0, 0], dtype=np.int64),
        FitConfig(
            model="MLS2PLM",
            latent_dim=1,
            max_iter=1,
            n_restarts=1,
            backend="auto",
        ),
    )

    assert result.backend == "numpy"
