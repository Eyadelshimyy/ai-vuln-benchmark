"""Fail-first ordering contracts for public G-theory semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


def _unexpected_core_discovery(name: str) -> object:
    """Fail if invalid G-theory input reaches native capability discovery."""
    raise AssertionError(f"invalid G-theory input reached Rust discovery: {name}")


class _UnexpectedArrayMaterialization:
    """Fail if invalid semantic controls reach caller-owned array conversion."""

    calls = 0

    def __array__(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("invalid G-theory control reached data materialization")


class _HostileArrayProvider:
    """Fail if scientific evidence admission executes caller array callbacks."""

    calls = 0

    def __array__(self, *args, **kwargs):
        type(self).calls += 1
        raise AssertionError("G-theory evidence admission executed caller __array__")


def _pi_data() -> np.ndarray:
    """Return a minimal two-dimensional score matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def _pio_data() -> np.ndarray:
    """Return a minimal three-dimensional score tensor."""
    return np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )


def test_gtheory_pi_rejects_invalid_size_before_core_discovery(monkeypatch) -> None:
    """Invalid one-facet D-study sizes fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_pi_data(), n_i_prime=[0])


def test_gtheory_pio_rejects_invalid_pair_before_core_discovery(monkeypatch) -> None:
    """Invalid two-facet D-study pairs fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match=r"n_prime entries must be pairs of positive integers",
    ):
        gtheory.gtheory_pio(_pio_data(), n_prime=[(2, 0)])


def test_phi_lambda_rejects_invalid_cut_before_core_discovery(monkeypatch) -> None:
    """Invalid mastery cuts fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_pi_data(), np.inf, n_i_prime=[2])


def test_phi_lambda_rejects_invalid_size_before_core_discovery(monkeypatch) -> None:
    """Invalid Phi(lambda) D-study sizes fail before native capability lookup."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.phi_lambda(_pi_data(), 0.5, n_i_prime=[0])


def test_gtheory_pi_rejects_invalid_size_before_data_materialization() -> None:
    """Invalid one-facet D-study sizes must fail before caller array callbacks."""
    _UnexpectedArrayMaterialization.calls = 0

    with pytest.raises(ValueError, match=r"n_i_prime entries must be positive integers"):
        gtheory.gtheory_pi(_UnexpectedArrayMaterialization(), n_i_prime=[0])

    assert _UnexpectedArrayMaterialization.calls == 0


def test_gtheory_pio_rejects_invalid_pair_before_data_materialization() -> None:
    """Invalid two-facet D-study pairs must fail before caller array callbacks."""
    _UnexpectedArrayMaterialization.calls = 0

    with pytest.raises(
        ValueError,
        match=r"n_prime entries must be pairs of positive integers",
    ):
        gtheory.gtheory_pio(_UnexpectedArrayMaterialization(), n_prime=[(2, 0)])

    assert _UnexpectedArrayMaterialization.calls == 0


def test_phi_lambda_rejects_invalid_cut_before_data_materialization() -> None:
    """Invalid mastery cuts must fail before caller array callbacks."""
    _UnexpectedArrayMaterialization.calls = 0

    with pytest.raises(ValueError, match=r"cut must be a finite real scalar"):
        gtheory.phi_lambda(_UnexpectedArrayMaterialization(), np.inf, n_i_prime=[2])

    assert _UnexpectedArrayMaterialization.calls == 0


@pytest.mark.parametrize(
    ("entrypoint", "kwargs"),
    [
        (gtheory.gtheory_pi, {"n_i_prime": [2]}),
        (gtheory.gtheory_pio, {"n_prime": [(2, 2)]}),
        (gtheory.phi_lambda, {"cut": 0.5, "n_i_prime": [2]}),
    ],
)
def test_gtheory_rejects_arbitrary_array_provider_before_callbacks_and_core(
    monkeypatch,
    entrypoint,
    kwargs,
) -> None:
    """G-study evidence must be inert before NumPy or Rust receives it."""
    _HostileArrayProvider.calls = 0
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"data must be a real numeric array"):
        entrypoint(_HostileArrayProvider(), **kwargs)

    assert _HostileArrayProvider.calls == 0


@pytest.mark.parametrize(
    ("entrypoint", "data", "kwargs"),
    [
        (
            gtheory.gtheory_pi,
            np.array([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128),
            {"n_i_prime": [2]},
        ),
        (
            gtheory.gtheory_pio,
            np.array(
                [
                    [[0.0 + 1.0j, 1.0], [1.0, 0.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                dtype=np.complex128,
            ),
            {"n_prime": [(2, 2)]},
        ),
    ],
)
def test_gtheory_rejects_complex_evidence_before_lossy_narrowing_and_core(
    monkeypatch,
    entrypoint,
    data,
    kwargs,
) -> None:
    """Imaginary score evidence must not be projected onto a real G-study."""
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"data must be real-valued"):
        entrypoint(data, **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        [[np.int8(0), np.float32(1.0)], [np.uint8(1), np.float64(0.0)]],
        ((np.bool_(False), np.int16(1)), (np.float32(1.0), np.uint16(0))),
    ],
)
def test_gtheory_real_data_preserves_builtin_sequence_numpy_scalars(data) -> None:
    """Trusted built-in score trees keep their concrete NumPy scalar compatibility."""
    result = gtheory._validated_real_data(data)

    assert type(result) is np.ndarray
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_gtheory_real_data_preserves_builtin_sequence_of_exact_ndarrays() -> None:
    """Historical built-in containers of exact inert NumPy rows remain supported."""
    rows = [
        np.array([0, 1], dtype=np.int16),
        np.array([1.0, 0.0], dtype=np.float32),
    ]

    result = gtheory._validated_real_data(rows)

    assert type(result) is np.ndarray
    assert result.dtype == np.float64
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, np.array([[0.0, 1.0], [1.0, 0.0]]))
