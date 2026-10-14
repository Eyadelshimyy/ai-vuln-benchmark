"""Fail-first public-boundary contracts for ATA semantic controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ata as ata
from fast_mlsirm.types import MLSIRMParams


class _HostileString:
    """Object whose representation callbacks must never run during validation."""

    def __init__(self) -> None:
        self.str_calls = 0
        self.repr_calls = 0

    def __str__(self) -> str:
        self.str_calls += 1
        raise RuntimeError("ATA_CONSTRAINT_STR_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("ATA_CONSTRAINT_REPR_SENTINEL")


class _HostileInteger:
    """Object whose integer-conversion hooks must never run during validation."""

    def __init__(self) -> None:
        self.int_calls = 0
        self.index_calls = 0

    def __int__(self) -> int:
        self.int_calls += 1
        raise RuntimeError("ATA_CONSTRAINT_INT_SENTINEL")

    def __index__(self) -> int:
        self.index_calls += 1
        raise RuntimeError("ATA_CONSTRAINT_INDEX_SENTINEL")


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small valid calibrated bank for ATA boundary tests."""
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)),
        b=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def _base_call(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> tuple[int, Exception | None]:
    """Call ATA with a counted information boundary and return observed failure."""
    bank, factor_id = _bank()
    information_calls = 0

    def counted_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", counted_information)
    kwargs: dict[str, object] = {
        "content": np.array(["A", "A", "B", "B"], dtype=object),
        "seed": 0,
    }
    kwargs.update(overrides)

    failure: Exception | None = None
    try:
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            np.array([100.0], dtype=np.float64),
            length=2,
            model="MIRT",
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - the test records the public boundary.
        failure = exc
    return information_calls, failure


def test_content_constraint_key_rejects_hostile_string_before_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content-map keys must be validated, not coerced through ``str`` callbacks."""
    hostile = _HostileString()

    calls, failure = _base_call(monkeypatch, min_per_content={hostile: 1})

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == "content constraint keys must be strings"
    assert hostile.str_calls == 0
    assert hostile.repr_calls == 0


def test_content_constraint_count_rejects_hostile_integer_before_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content-map counts must be exact integers without invoking conversion hooks."""
    hostile = _HostileInteger()

    calls, failure = _base_call(monkeypatch, max_per_content={"A": hostile})

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == "content constraint counts must be integers"
    assert hostile.int_calls == 0
    assert hostile.index_calls == 0


def test_exposure_count_key_rejects_hostile_integer_before_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exposure-map identities must not invoke arbitrary integer conversion callbacks."""
    hostile = _HostileInteger()

    calls, failure = _base_call(monkeypatch, exposure_counts={hostile: 1})

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == "exposure_counts keys and values must be integers"
    assert hostile.int_calls == 0
    assert hostile.index_calls == 0


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", 1.5), ("exposure_max", True), ("exposure_max", 1.5)])
def test_scalar_controls_require_exact_integers_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    """Boolean/fractional scalar controls must not be silently coerced to integers."""
    calls, failure = _base_call(monkeypatch, **{field: value})

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == f"{field} must be an integer"
