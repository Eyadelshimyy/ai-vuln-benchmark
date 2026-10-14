"""Regression coverage for lossless Oakes uncertainty input admission."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from fast_mlsirm import _core
import fast_mlsirm.inference as inference
from fast_mlsirm.inference import oakes_standard_errors


def _result():
    """Return the smallest converged marginal-fit record accepted by the wrapper."""
    return SimpleNamespace(
        model="MLS2PLM",
        optimizer="mmle_marginal_em/rust",
        convergence_status="converged",
        population={},
        params=SimpleNamespace(
            alpha=np.array([0.0], dtype=np.float64),
            b=np.array([0.0], dtype=np.float64),
            zeta=np.zeros((1, 1), dtype=np.float64),
            tau=1.0,
        ),
    )


def _unexpected_oakes_dispatch(*args, **kwargs):
    """Fail if an invalid input reaches Rust-owned Oakes arithmetic."""
    raise AssertionError("invalid Oakes input reached native uncertainty arithmetic")


class _HostileArrayProvider:
    """Record any attempt to execute a caller-owned NumPy array protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, dtype=None):
        self.calls += 1
        raise AssertionError("caller __array__ protocol executed")


class _HostileFloatProvider:
    """Record any attempt to execute a caller-owned numeric conversion protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def __float__(self) -> float:
        self.calls += 1
        raise AssertionError("caller __float__ protocol executed")


class _HostileTruthProvider:
    """Record any attempt to execute a caller-owned truth-value protocol."""

    def __init__(self) -> None:
        self.calls = 0

    def __bool__(self) -> bool:
        self.calls += 1
        raise AssertionError("caller __bool__ protocol executed")


class _MaskMeta(type):
    """Metaclass that records unsafe class-attribute admission reads."""

    calls = 0

    def __getattribute__(cls, name: str):
        if name in {"__module__", "__mro__", "__bases__"}:
            _MaskMeta.calls += 1
            raise AssertionError("caller metaclass attribute callback executed")
        return super().__getattribute__(name)


class _HostileMaskCell(metaclass=_MaskMeta):
    """Mask cell whose class metadata must never be inspected during admission."""


class _HostileNumericCell(metaclass=_MaskMeta):
    """Scientific numeric cell whose class metadata must never be inspected."""


def test_oakes_rejects_control_before_caller_evidence(monkeypatch):
    """An invalid callback-bearing step must fail before response/factor work."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = _HostileArrayProvider()
    factor_id = _HostileArrayProvider()
    h = _HostileFloatProvider()

    with pytest.raises(ValueError, match="h must be > 0 and finite"):
        oakes_standard_errors(_result(), responses, factor_id, h=h)

    assert h.calls == 0
    assert responses.calls == 0
    assert factor_id.calls == 0


@pytest.mark.parametrize("field", ["responses", "factor_id"])
def test_oakes_rejects_top_level_array_providers_before_callbacks(monkeypatch, field):
    """Scientific evidence must be sealed before any caller array protocol runs."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    hostile = _HostileArrayProvider()
    responses = hostile if field == "responses" else np.array([[0.0], [1.0]])
    factor_id = hostile if field == "factor_id" else np.array([0], dtype=np.int64)

    with pytest.raises(ValueError):
        oakes_standard_errors(_result(), responses, factor_id)

    assert hostile.calls == 0


def test_oakes_rejects_nested_numeric_provider_before_conversion(monkeypatch):
    """Built-in scientific containers cannot smuggle conversion callbacks."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    hostile = _HostileFloatProvider()

    with pytest.raises(ValueError):
        oakes_standard_errors(_result(), [[0.0], [hostile]], [0])

    assert hostile.calls == 0


@pytest.mark.parametrize("field", ["responses", "factor_id"])
def test_oakes_rejects_numeric_metaclass_before_class_attribute_callbacks(monkeypatch, field):
    """Numeric admission must use type identity without caller class metadata reads."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    _MaskMeta.calls = 0
    hostile = _HostileNumericCell()
    responses = [[0.0], [hostile]] if field == "responses" else [[0.0], [1.0]]
    factor_id = [hostile] if field == "factor_id" else [0]

    with pytest.raises(ValueError):
        oakes_standard_errors(_result(), responses, factor_id)

    assert _MaskMeta.calls == 0


def test_oakes_rejects_mask_truth_provider_before_truth_coercion(monkeypatch):
    """Observation-mask cells must not execute caller truth protocols."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    hostile = _HostileTruthProvider()

    with pytest.raises(ValueError):
        oakes_standard_errors(
            _result(),
            [[0.0], [1.0]],
            [0],
            mask=[[True], [hostile]],
        )

    assert hostile.calls == 0


def test_oakes_rejects_mask_metaclass_before_class_attribute_callbacks(monkeypatch):
    """Mask admission must use type identity without inspecting caller class metadata."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    _MaskMeta.calls = 0

    with pytest.raises(ValueError, match="mask must contain concrete numeric or Boolean values"):
        oakes_standard_errors(
            _result(),
            [[0.0], [1.0]],
            [0],
            mask=[[True], [_HostileMaskCell()]],
        )

    assert _MaskMeta.calls == 0


@pytest.mark.parametrize("mask_scalar", [np.longlong(1), np.ulonglong(1)])
def test_oakes_preserves_typecode_integer_mask_scalar_compatibility(monkeypatch, mask_scalar):
    """Built-in masks admit every concrete NumPy integer scalar supported by typecodes."""
    captured: dict[str, np.ndarray] = {}

    def fake_oakes(*args, **kwargs):
        captured["observed"] = np.asarray(args[1])
        return {"ok": True}

    monkeypatch.setattr(_core, "oakes_standard_errors", fake_oakes)

    result = oakes_standard_errors(
        _result(),
        [[0.0], [1.0]],
        [0],
        mask=[[mask_scalar], [np.bool_(True)]],
    )

    assert result == {"ok": True}
    assert np.array_equal(captured["observed"], np.array([True, True]))


def test_oakes_bounds_response_cells_before_native_arithmetic(monkeypatch):
    """Logical response size must be bounded before dense/native work."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    monkeypatch.setattr(inference, "_MAX_OAKES_RESPONSE_CELLS", 2, raising=False)

    with pytest.raises(ValueError, match="responses resource limit exceeded"):
        oakes_standard_errors(_result(), [[0.0], [1.0], [0.0]], [0])


def test_oakes_rejects_complex_responses_before_native_arithmetic(monkeypatch):
    """Imaginary response components must not be discarded before Oakes SEs."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0 + 1.0j], [1.0 + 0.0j]])

    with pytest.raises(ValueError, match="responses must be real-valued"):
        oakes_standard_errors(_result(), responses, np.array([0], dtype=np.int64))


def test_oakes_rejects_complex_factor_id_before_native_arithmetic(monkeypatch):
    """Imaginary factor assignments must not be discarded before Oakes SEs."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0], [1.0]])
    factor_id = np.array([0.0 + 1.0j])

    with pytest.raises(ValueError, match="factor_id must be real-valued integers"):
        oakes_standard_errors(_result(), responses, factor_id)


@pytest.mark.parametrize("factor_value", [2**63, np.iinfo(np.uint64).max])
def test_oakes_rejects_factor_id_signed64_overflow_before_native_arithmetic(
    monkeypatch, factor_value
):
    """Unsigned factor labels must not wrap while narrowing to Rust int64 IDs."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0], [1.0]])
    factor_id = np.array([factor_value], dtype=np.uint64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="factor_id must fit signed 64-bit integers"):
            oakes_standard_errors(_result(), responses, factor_id)

    assert not [warning for warning in caught if issubclass(warning.category, RuntimeWarning)]


def test_oakes_preserves_signed64_max_until_structural_dimension_validation(monkeypatch):
    """The valid signed-int64 boundary must not be mistaken for narrowing overflow."""
    monkeypatch.setattr(_core, "oakes_standard_errors", _unexpected_oakes_dispatch)
    responses = np.array([[0.0], [1.0]])
    factor_id = np.array([np.iinfo(np.int64).max], dtype=np.int64)

    with pytest.raises(ValueError, match="factor_id implies more dimensions than items"):
        oakes_standard_errors(_result(), responses, factor_id)


def test_oakes_preserves_real_response_missingness_and_factor_assignment(monkeypatch):
    """Binary values, NaN/-1 missingness, and integer factors retain their contract."""
    captured: dict[str, np.ndarray] = {}

    def fake_oakes(*args, **kwargs):
        captured["responses"] = np.asarray(args[0])
        captured["observed"] = np.asarray(args[1])
        captured["factors"] = np.asarray(args[2])
        return {"ok": True}

    monkeypatch.setattr(_core, "oakes_standard_errors", fake_oakes)
    responses = np.array([[0.0], [np.nan], [-1.0], [1.0]], dtype=np.float32)

    result = oakes_standard_errors(_result(), responses, np.array([0], dtype=np.int32))

    assert result == {"ok": True}
    assert np.array_equal(captured["responses"], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.array_equal(captured["observed"], np.array([True, False, False, True]))
    assert np.array_equal(captured["factors"], np.array([0], dtype=np.int64))


def test_oakes_preserves_built_in_container_and_numpy_scalar_compatibility(monkeypatch):
    """Inert built-in evidence remains supported and is canonically marshalled."""
    captured: dict[str, np.ndarray] = {}

    def fake_oakes(*args, **kwargs):
        captured["responses"] = np.asarray(args[0])
        captured["observed"] = np.asarray(args[1])
        captured["factors"] = np.asarray(args[2])
        return {"ok": True}

    monkeypatch.setattr(_core, "oakes_standard_errors", fake_oakes)
    responses = [
        [np.float32(0.0)],
        [np.float64(np.nan)],
        [np.int8(-1)],
        [np.uint8(1)],
    ]
    mask = [[np.bool_(True)], [np.bool_(True)], [np.bool_(True)], [np.bool_(True)]]

    result = oakes_standard_errors(_result(), responses, (np.int32(0),), mask=mask)

    assert result == {"ok": True}
    assert np.array_equal(captured["responses"], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.array_equal(captured["observed"], np.array([True, False, False, True]))
    assert np.array_equal(captured["factors"], np.array([0], dtype=np.int64))
