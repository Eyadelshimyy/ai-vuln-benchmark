"""Fail-first ATA integer-control callback-safety contracts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.ata as ata
from fast_mlsirm.types import MLSIRMParams


class _HostileInt(int):
    """Python integer subclass whose conversion callback must never execute."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise RuntimeError("ATA_HOSTILE_INT_CALLBACK")


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass whose conversion callback must never execute."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise RuntimeError("ATA_HOSTILE_NUMPY_INT_CALLBACK")


class _HostileIntegerMeta(type):
    """Metaclass proving integer admission cannot dispatch caller equality."""

    calls: list[str] = []
    __hash__ = type.__hash__

    def __eq__(cls, other: object) -> bool:
        cls.calls.append("type-__eq__")
        raise RuntimeError("ATA_HOSTILE_TYPE_EQUALITY_CALLBACK")


class _HostileMetaInt(int, metaclass=_HostileIntegerMeta):
    """Integer subclass whose type equality callback must never execute."""


class _HostileMetaNumpyInt(np.int64, metaclass=_HostileIntegerMeta):
    """NumPy integer subclass whose type equality callback must never execute."""


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small valid calibrated bank for public ATA preflight tests."""
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)),
        b=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def _invoke(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> tuple[int, BaseException | None]:
    """Invoke ATA while counting whether rejected controls reach information work."""
    bank, factor_id = _bank()
    information_calls = 0

    def counted_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", counted_information)
    kwargs: dict[str, object] = {
        "bank": bank,
        "factor_id": factor_id,
        "target_thetas": np.array([0.0], dtype=np.float64),
        "target_info": np.array([100.0], dtype=np.float64),
        "length": 2,
        "model": "MIRT",
        "content": np.array(["A", "A", "B", "B"], dtype=object),
        "seed": 0,
    }
    mutate(kwargs)

    failure: BaseException | None = None
    try:
        ata.assemble_to_target(**kwargs)
    except BaseException as exc:  # noqa: BLE001 - callback escape is the regression target.
        failure = exc
    return information_calls, failure


@pytest.mark.parametrize(
    ("field", "value_factory", "expected_message"),
    [
        ("length", lambda: _HostileInt(2), "length must be an integer"),
        ("seed", lambda: _HostileInt(0), "seed must be an integer"),
        ("seed", lambda: _HostileNumpyInt(0), "seed must be an integer"),
        ("exposure_max", lambda: _HostileInt(1), "exposure_max must be an integer"),
    ],
)
def test_scalar_integer_subclasses_fail_before_conversion_or_information(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value_factory: Callable[[], object],
    expected_message: str,
) -> None:
    """Scalar integer subclasses must not run conversion callbacks or scoring."""
    _HostileInt.calls = 0
    _HostileNumpyInt.calls = 0
    value = value_factory()

    calls, failure = _invoke(monkeypatch, lambda kwargs: kwargs.__setitem__(field, value))

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == expected_message
    assert _HostileInt.calls == 0
    assert _HostileNumpyInt.calls == 0


@pytest.mark.parametrize("value", [_HostileMetaInt(0), _HostileMetaNumpyInt(0)])
def test_scalar_integer_subclasses_fail_before_type_equality_or_information(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    """Exact-type admission must not invoke caller-controlled metaclass equality."""
    _HostileIntegerMeta.calls = []

    calls, failure = _invoke(monkeypatch, lambda kwargs: kwargs.__setitem__("seed", value))

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == "seed must be an integer"
    assert _HostileIntegerMeta.calls == []


@pytest.mark.parametrize(
    ("mutate", "expected_message"),
    [
        (
            lambda kwargs: kwargs.__setitem__("min_per_content", {"A": _HostileInt(1)}),
            "content constraint counts must be integers",
        ),
        (
            lambda kwargs: kwargs.__setitem__("exposure_counts", {0: _HostileInt(0)}),
            "exposure_counts keys and values must be integers",
        ),
        (
            lambda kwargs: kwargs.__setitem__("exclude", [_HostileInt(1)]),
            "exclude must contain integer item indices",
        ),
    ],
)
def test_container_integer_subclasses_fail_before_conversion_or_information(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
    expected_message: str,
) -> None:
    """Container integer subclasses must fail closed before callbacks or scoring."""
    _HostileInt.calls = 0

    calls, failure = _invoke(monkeypatch, mutate)

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == expected_message
    assert _HostileInt.calls == 0
