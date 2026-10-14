"""Trust-boundary regressions for Owen-approximate EPV selection."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import exposure


class _UnexpectedCore:
    """Fail if rejected caller evidence reaches native EPV dispatch."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"native core reached before EPV admission: {name}")


class _HostileFloat(float):
    """Float subclass whose conversion callback must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller __float__ executed during EPV admission")


class _HostileElement:
    """Object-array cell whose numeric conversion must never execute."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller element __float__ executed during EPV admission")


@pytest.mark.parametrize("field", ["mu", "sig2"])
def test_epv_rejects_hostile_posterior_scalar_before_callback_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Posterior controls require package-trusted scalar identity."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileFloat.callbacks = 0
    values: dict[str, object] = {"mu": 0.0, "sig2": 1.0}
    values[field] = _HostileFloat(0.5)

    with pytest.raises(ValueError, match=rf"{field} must be a real scalar"):
        exposure.epv_select(
            np.array([1.0]),
            np.array([0.0]),
            administered=np.array([False]),
            mu=values["mu"],
            sig2=values["sig2"],
        )

    assert _HostileFloat.callbacks == 0


@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_epv_rejects_complex_item_evidence_before_lossy_cast_or_native(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Imaginary item evidence must never be projected onto the real axis."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    values: dict[str, object] = {
        "a": np.array([1.0]),
        "b": np.array([0.0]),
        "c": np.array([0.0]),
    }
    values[field] = np.array([1.0 + 0.25j])

    with pytest.raises(ValueError, match=rf"{field} must be a real numeric array"):
        exposure.epv_select(
            values["a"],
            values["b"],
            values["c"],
            administered=np.array([False]),
            mu=0.0,
            sig2=1.0,
        )


def test_epv_rejects_object_item_storage_without_element_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Object item storage must fail before caller numeric callbacks execute."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)
    _HostileElement.callbacks = 0
    hostile = np.array([_HostileElement()], dtype=object)

    with pytest.raises(ValueError, match="a must be a real numeric array"):
        exposure.epv_select(
            hostile,
            np.array([0.0]),
            administered=np.array([False]),
            mu=0.0,
            sig2=1.0,
        )

    assert _HostileElement.callbacks == 0


def test_epv_rejects_text_item_storage_before_numeric_coercion_or_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text that happens to parse as a number is not item evidence."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)

    with pytest.raises(ValueError, match="a must be a real numeric array"):
        exposure.epv_select(
            np.array(["1.0"]),
            np.array([0.0]),
            administered=np.array([False]),
            mu=0.0,
            sig2=1.0,
        )


def test_epv_rejects_non_boolean_administered_storage_before_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection-mask identity must already be Boolean storage."""

    monkeypatch.setattr(fast_mlsirm, "_core", _UnexpectedCore(), raising=False)

    with pytest.raises(ValueError, match="administered must be a boolean array"):
        exposure.epv_select(
            np.array([1.0]),
            np.array([0.0]),
            administered=np.array([0], dtype=np.int8),
            mu=0.0,
            sig2=1.0,
        )


def test_valid_epv_inputs_preserve_normalized_native_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted NumPy inputs reach Rust as normalized package payloads."""

    captured: dict[str, tuple[object, ...]] = {}

    class _Core:
        def py_epv_select(self, *args: object) -> dict[str, object]:
            captured["args"] = args
            return {
                "selected": 0,
                "epv": np.array([0.5]),
                "predictive": np.array([0.6]),
            }

    monkeypatch.setattr(fast_mlsirm, "_core", _Core(), raising=False)

    result = exposure.epv_select(
        np.array([1], dtype=np.int16),
        np.array([0.0], dtype=np.float32),
        None,
        administered=np.array([False], dtype=np.bool_),
        mu=np.float32(0.0),
        sig2=np.float64(1.0),
    )

    args = captured["args"]
    for array in args[:3]:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
    assert isinstance(args[3], np.ndarray)
    assert args[3].dtype == np.bool_
    assert args[3].flags.c_contiguous
    assert type(args[4]) is float
    assert type(args[5]) is float
    assert result["selected"] == 0
