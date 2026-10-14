"""Trust-boundary regressions for factor/reliability/MAP wrappers."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.factor as factor


def _data() -> np.ndarray:
    rng = np.random.default_rng(1201)
    latent = rng.standard_normal((160, 1))
    loadings = np.array([[0.75, 0.65, 0.55, 0.45]])
    return latent @ loadings + rng.standard_normal((160, 4)) * 0.45


def _corr() -> np.ndarray:
    return np.corrcoef(_data().T)


class _ArrayBomb:
    """Caller-owned array protocol that must not run for rejected controls."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("caller array protocol executed")


class _HostileInt(int):
    """An integer subclass whose coercion must never be invoked."""

    def __int__(self) -> int:
        raise AssertionError("caller integer coercion executed")

    def __index__(self) -> int:
        raise AssertionError("caller index coercion executed")


class _FloatBomb:
    """Object-array element whose numeric conversion must never run."""

    def __float__(self) -> float:
        raise AssertionError("caller element conversion executed")


@pytest.mark.parametrize(
    "call",
    [
        lambda matrix: factor.minres_fa(matrix, 1),
        factor.omega_total_1f,
        factor.glb_fa,
        factor.velicer_map,
    ],
)
def test_correlation_wrappers_reject_complex_before_real_narrowing(call):
    corr = _corr().astype(np.complex128)
    corr[0, 1] += 0.25j
    corr[1, 0] -= 0.25j

    with pytest.raises(ValueError, match="real-valued"):
        call(corr)


@pytest.mark.parametrize(
    "call",
    [
        lambda data: factor.minres_fa_from_data(data, 1),
        factor.omega_total_1f_from_data,
        factor.glb_fa_from_data,
        factor.velicer_map_from_data,
    ],
)
def test_data_wrappers_reject_complex_before_real_narrowing(call):
    data = _data().astype(np.complex128)
    data[0, 0] += 1.0j

    with pytest.raises(ValueError, match="real-valued"):
        call(data)


@pytest.mark.parametrize(
    "call",
    [
        lambda matrix: factor.minres_fa(matrix, 1),
        factor.omega_total_1f,
        factor.glb_fa,
        factor.velicer_map,
    ],
)
def test_correlation_wrappers_reject_object_storage_without_element_coercion(call):
    corr = _corr().astype(object)
    corr[0, 0] = _FloatBomb()

    with pytest.raises(ValueError, match="numeric"):
        call(corr)


@pytest.mark.parametrize(
    "call",
    [
        lambda data: factor.minres_fa_from_data(data, 1),
        factor.omega_total_1f_from_data,
        factor.glb_fa_from_data,
        factor.velicer_map_from_data,
    ],
)
def test_data_wrappers_reject_object_storage_without_element_coercion(call):
    data = _data().astype(object)
    data[0, 0] = _FloatBomb()

    with pytest.raises(ValueError, match="numeric"):
        call(data)


@pytest.mark.parametrize(
    "call",
    [
        lambda payload, control: factor.minres_fa(payload, control),
        lambda payload, control: factor.minres_fa_from_data(payload, control),
        lambda payload, control: factor.velicer_map(payload, max_m=control),
        lambda payload, control: factor.velicer_map_from_data(payload, max_m=control),
    ],
)
def test_rejected_count_subclass_precedes_caller_data_work(call):
    payload = _ArrayBomb()

    with pytest.raises(TypeError, match="integer"):
        call(payload, _HostileInt(1))

    assert payload.calls == 0


def test_concrete_numpy_integer_controls_remain_supported():
    fa = factor.minres_fa(_corr(), np.int64(1))
    mapped = factor.velicer_map(_corr(), max_m=np.int32(2))

    assert fa.loadings.shape == (4, 1)
    assert mapped.f2.shape[0] >= 1
