"""Trust-boundary regressions for dichotomous CAT administration evidence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm.cat as cat_module
from fast_mlsirm.cat import estimate_ability_eap
from fast_mlsirm.types import MLSIRMParams


class _ArrayProvider:
    def __init__(self, payload: np.ndarray) -> None:
        self.payload = payload
        self.calls = 0

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.payload, dtype=dtype)


class _CoreSentinel:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"compiled CAT core must not be reached for invalid evidence: {name}")


class _CoreCapture:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def cat_ability_eap(self, **kwargs: object) -> tuple[list[float], list[float], list[bool]]:
        self.kwargs = kwargs
        return [0.0], [1.0], [True]

    def cat_ability_standard_error(self, **kwargs: object) -> list[float]:
        self.kwargs = kwargs
        return [0.5]


def _bank() -> MLSIRMParams:
    return MLSIRMParams(
        theta=np.zeros((1, 1), dtype=np.float64),
        alpha=np.zeros(2, dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((2, 1), dtype=np.float64),
        tau=0.0,
    )


def _forbid_dense_conversion(
    monkeypatch: pytest.MonkeyPatch,
    *protected: np.ndarray,
) -> None:
    """Fail if CAT asks NumPy to convert one of the protected evidence arrays."""
    original_asarray = np.asarray
    protected_ids = {id(value) for value in protected}

    def guarded_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
        if id(value) in protected_ids:
            raise AssertionError("invalid CAT evidence must fail before dense NumPy conversion")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(cat_module.np, "asarray", guarded_asarray)


def test_cat_rejects_administered_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ArrayProvider(np.array([0], dtype=np.int64))
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="administered item indices"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            provider,  # type: ignore[arg-type]
            [1],
        )

    assert provider.calls == 0


def test_cat_rejects_response_array_provider_before_callback_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ArrayProvider(np.array([1.0], dtype=np.float64))
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            [0],
            provider,  # type: ignore[arg-type]
        )

    assert provider.calls == 0


def test_cat_preserves_trusted_numpy_scalar_sequence_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _CoreCapture()
    monkeypatch.setattr(fast_mlsirm, "_core", core, raising=False)

    result = estimate_ability_eap(
        _bank(),
        np.zeros(2, dtype=np.int64),
        [np.int16(0), np.float32(1.0)],
        (np.bool_(True), np.uint8(0)),
    )

    assert result.method == "eap"
    assert core.kwargs is not None
    administered = core.kwargs["administered"]
    responses = core.kwargs["responses"]
    assert type(administered) is np.ndarray
    assert type(responses) is np.ndarray
    np.testing.assert_array_equal(administered, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(responses, np.array([1.0, 0.0], dtype=np.float64))


def test_cat_rejects_impossible_administration_length_before_dense_conversion_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administered = np.broadcast_to(np.array([0], dtype=np.int64), (3,))
    responses = np.broadcast_to(np.array([1.0], dtype=np.float64), (3,))
    _forbid_dense_conversion(monkeypatch, administered, responses)
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="administration length cannot exceed item bank size"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            administered,
            responses,
        )


def test_cat_rejects_impossible_administration_before_unsupported_response_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-bank administration must fail before response protocols or dense scans."""
    administered = np.broadcast_to(np.array([0], dtype=np.int64), (3,))
    responses = _ArrayProvider(np.array([1.0], dtype=np.float64))
    _forbid_dense_conversion(monkeypatch, administered)
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="administration length cannot exceed item bank size"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            administered,
            responses,  # type: ignore[arg-type]
        )

    assert responses.calls == 0


def test_cat_rejects_length_mismatch_before_dense_conversion_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administered = np.array([0, 1], dtype=np.int64)
    responses = np.array([1.0], dtype=np.float64)
    _forbid_dense_conversion(monkeypatch, administered, responses)
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="administered and responses must be 1D arrays of equal length"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            administered,
            responses,
        )


def test_cat_rejects_overrank_administration_before_dense_conversion_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administered = np.array([[0]], dtype=np.int64)
    responses = np.array([[1.0]], dtype=np.float64)
    _forbid_dense_conversion(monkeypatch, administered, responses)
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(ValueError, match="administered and responses must be 1D arrays of equal length"):
        estimate_ability_eap(
            _bank(),
            np.zeros(2, dtype=np.int64),
            administered,
            responses,
        )


@pytest.mark.parametrize(
    "standard_error",
    [cat_module.ability_standard_error, fast_mlsirm.ability_standard_error],
    ids=["module", "package"],
)
@pytest.mark.parametrize(
    "administered",
    [
        np.array([0, 0, 1], dtype=np.int64),
        np.array([[0, 1], [1, 0]], dtype=np.int64),
    ],
    ids=["duplicates-over-bank-length", "multidimensional-dedup"],
)
def test_standard_error_preserves_historical_deduplication_semantics(
    monkeypatch: pytest.MonkeyPatch,
    standard_error: Callable[..., np.ndarray],
    administered: np.ndarray,
) -> None:
    """SE accepts duplicate/over-rank evidence because its mask is set-valued."""
    core = _CoreCapture()
    monkeypatch.setattr(fast_mlsirm, "_core", core, raising=False)

    result = standard_error(
        _bank(),
        np.zeros(2, dtype=np.int64),
        np.zeros(1, dtype=np.float64),
        administered=administered,
    )

    assert core.kwargs is not None
    np.testing.assert_array_equal(core.kwargs["administered"], np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(result, np.array([0.5], dtype=np.float64))
