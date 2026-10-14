"""Callback-free scientific-evidence regressions for Rasch CML."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.rasch_cml as rasch_cml


class _HostileArrayProvider:
    """Array provider whose callback must not execute during package admission."""

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("caller array protocol executed")


def _unexpected_core() -> object:
    """Fail if invalid evidence reaches compiled-core discovery."""
    raise AssertionError("compiled core was discovered")


def test_fit_rasch_cml_rejects_response_provider_before_protocol_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted response providers fail before NumPy or Rust discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)

    with pytest.raises(ValueError, match="responses must be complete 0/1"):
        rasch_cml.fit_rasch_cml(_HostileArrayProvider(), max_iter=100, tol=1e-8)


def test_andersen_rejects_group_provider_before_protocol_or_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted group providers cannot synthesize the Andersen split."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core)
    responses = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int8)

    with pytest.raises(ValueError, match="group labels must be finite non-negative integers"):
        rasch_cml.andersen_lr_test(responses, _HostileArrayProvider(), max_iter=100, tol=1e-8)


def test_rasch_cml_preserves_trusted_builtin_and_numpy_scalar_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical inert array-like evidence still reaches Rust as canonical arrays."""
    captured: dict[str, np.ndarray] = {}

    class _Core:
        def fit_rasch_cml(
            self,
            responses: np.ndarray,
            n_persons: int,
            n_items: int,
            max_iter: int,
            tol: float,
        ) -> dict[str, object]:
            assert type(responses) is np.ndarray
            assert responses.dtype == np.int64
            assert (n_persons, n_items) == (4, 2)
            assert type(max_iter) is int
            assert type(tol) is float
            captured["fit"] = responses.copy()
            return {
                "beta": [0.0, 0.0],
                "se": [1.0, 1.0],
                "loglik": -1.0,
                "n_iter": 1,
                "converged": True,
                "n_used": 4,
            }

        def andersen_lr_test(
            self,
            responses: np.ndarray,
            group: np.ndarray,
            n_groups: int,
            n_persons: int,
            n_items: int,
            max_iter: int,
            tol: float,
        ) -> dict[str, object]:
            assert type(responses) is np.ndarray
            assert responses.dtype == np.int64
            assert type(group) is np.ndarray
            assert group.dtype == np.int64
            assert n_groups == 2
            assert (n_persons, n_items) == (4, 2)
            captured["group"] = group.copy()
            return {
                "lr": 0.0,
                "df": 1,
                "p_value": 1.0,
                "n_used": [2, 2],
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())

    responses = [
        [np.bool_(False), np.uint8(1)],
        [np.int8(1), np.float32(0.0)],
        [0, 1],
        [1.0, 0.0],
    ]
    group = [np.int16(0), np.uint8(0), 1, np.int32(1)]

    assert rasch_cml.fit_rasch_cml(responses, max_iter=100, tol=1e-8)["converged"] is True
    assert rasch_cml.andersen_lr_test(responses, group, max_iter=100, tol=1e-8)["converged"] is True
    assert captured["fit"].tolist() == [0, 1, 1, 0, 0, 1, 1, 0]
    assert captured["group"].tolist() == [0, 0, 1, 1]


def _assert_andersen_dense_groups(
    monkeypatch: pytest.MonkeyPatch,
    group: object,
    expected: list[int],
) -> None:
    """Require external group identities to survive until deterministic densification."""
    responses = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int8)

    class _Core:
        def andersen_lr_test(
            self,
            responses: np.ndarray,
            dense_group: np.ndarray,
            n_groups: int,
            n_persons: int,
            n_items: int,
            max_iter: int,
            tol: float,
        ) -> dict[str, object]:
            assert type(dense_group) is np.ndarray
            assert dense_group.dtype == np.int64
            assert dense_group.tolist() == expected
            assert n_groups == len(set(expected))
            return {
                "lr": 0.0,
                "df": max(1, (n_groups - 1) * (n_items - 1)),
                "p_value": 1.0,
                "n_used": [1] * n_groups,
                "converged": True,
            }

    monkeypatch.setattr(fitstats, "_core_module", lambda: _Core())
    assert rasch_cml.andersen_lr_test(responses, group, max_iter=100, tol=1e-8)["converged"] is True


def test_andersen_preserves_integer_identity_beyond_float64_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct exact integer labels must not collapse through float64."""
    group = np.array([0, 2**53, 2**53 + 1, 0], dtype=np.uint64)
    _assert_andersen_dense_groups(monkeypatch, group, [0, 1, 2, 0])


def test_andersen_preserves_uint64_identity_without_signed_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsigned external labels stay ordered by exact value before dense mapping."""
    group = np.array([0, np.iinfo(np.uint64).max, 1, 0], dtype=np.uint64)
    _assert_andersen_dense_groups(monkeypatch, group, [0, 2, 1, 0])
