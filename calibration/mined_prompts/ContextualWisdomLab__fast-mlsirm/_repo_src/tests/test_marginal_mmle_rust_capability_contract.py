"""ABI compatibility contract for Rust marginal MMLE dispatch."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm.config import FitConfig


fit_module = importlib.import_module("fast_mlsirm.fit")
marginal_module = importlib.import_module("fast_mlsirm.estimators.marginal")
_MISSING = object()


@pytest.mark.parametrize(
    "capability",
    [
        pytest.param(_MISSING, id="absent"),
        pytest.param(None, id="none"),
        pytest.param(False, id="boolean"),
        pytest.param("1", id="string"),
        pytest.param(0, id="stale"),
        pytest.param(2, id="unsupported-future"),
    ],
)
def test_public_spatial_mmle_rejects_incompatible_rust_marginal_capability(
    monkeypatch: pytest.MonkeyPatch,
    capability: object,
) -> None:
    """Absent, malformed, stale, and unsupported ABIs fail before dispatch."""
    responses = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    factor_id = np.zeros(responses.shape[1], dtype=np.int64)
    stale_calls: list[tuple[object, ...]] = []
    numpy_calls: list[tuple[object, ...]] = []

    def stale_fit_marginal(
        y: object,
        observed: object,
        factors: object,
        n_persons: object,
        n_items: object,
        n_dims: object,
        latent_dim: object,
        model: object,
        eps_distance: object,
    ) -> None:
        stale_calls.append(
            (
                y,
                observed,
                factors,
                n_persons,
                n_items,
                n_dims,
                latent_dim,
                model,
                eps_distance,
            )
        )

    def reject_numpy_reference(
        *args: object, **_kwargs: object
    ) -> dict[str, object]:
        numpy_calls.append(args)
        raise AssertionError(
            "public marginal MMLE entered NumPy production arithmetic"
        )

    core_attributes: dict[str, object] = {
        "fit_marginal": stale_fit_marginal,
    }
    if capability is not _MISSING:
        core_attributes["MARGINAL_CAPABILITY_VERSION"] = capability
    stale_core = SimpleNamespace(**core_attributes)
    monkeypatch.setattr(fast_mlsirm, "_core", stale_core, raising=False)
    monkeypatch.setattr(fit_module, "resolve_backend", lambda _backend: "rust")
    monkeypatch.setattr(
        marginal_module, "fit_marginal_numpy", reject_numpy_reference
    )

    with pytest.raises(
        RuntimeError,
        match=r"compiled Rust core marginal ABI capability",
    ):
        fit_module.fit(
            responses,
            factor_id,
            FitConfig(
                estimator="mmle",
                model="MLS2PLM",
                backend="rust",
                latent_dim=1,
                max_iter=1,
                n_restarts=1,
            ),
        )

    assert stale_calls == []
    assert numpy_calls == []


def test_public_spatial_mmle_rejects_versioned_callable_with_stale_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A version-1 callable that rejects current keywords must not leak TypeError."""
    responses = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    factor_id = np.zeros(responses.shape[1], dtype=np.int64)
    stale_calls: list[tuple[object, ...]] = []
    numpy_calls: list[tuple[object, ...]] = []

    def stale_keyword_fit_marginal(*args: object, **_kwargs: object) -> None:
        stale_calls.append(args)
        raise TypeError(
            "fit_marginal() got an unexpected keyword argument 'pop_kind'"
        )

    def reject_numpy_reference(
        *args: object, **_kwargs: object
    ) -> dict[str, object]:
        numpy_calls.append(args)
        raise AssertionError(
            "public marginal MMLE entered NumPy production arithmetic"
        )

    stale_core = SimpleNamespace(
        MARGINAL_CAPABILITY_VERSION=fit_module._MARGINAL_CAPABILITY_VERSION,
        fit_marginal=stale_keyword_fit_marginal,
    )
    monkeypatch.setattr(fast_mlsirm, "_core", stale_core, raising=False)
    monkeypatch.setattr(fit_module, "resolve_backend", lambda _backend: "rust")
    monkeypatch.setattr(
        marginal_module, "fit_marginal_numpy", reject_numpy_reference
    )

    with pytest.raises(
        RuntimeError,
        match=r"compiled Rust core marginal ABI capability",
    ):
        fit_module.fit(
            responses,
            factor_id,
            FitConfig(
                estimator="mmle",
                model="MLS2PLM",
                backend="rust",
                latent_dim=1,
                max_iter=1,
                n_restarts=1,
            ),
        )

    assert stale_calls
    assert numpy_calls == []


def test_compiled_rust_core_exports_current_marginal_capability() -> None:
    """The native module publishes the exact Python-supported ABI version."""
    core = pytest.importorskip("fast_mlsirm._core")
    capability = core.MARGINAL_CAPABILITY_VERSION
    assert type(capability) is int
    assert capability == fit_module._MARGINAL_CAPABILITY_VERSION
