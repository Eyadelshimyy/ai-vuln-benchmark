"""Fail-first validation contracts for public IRT-linking controls."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.linking import IrtLinkResult, irt_link


class _HostileMethod:
    """Method control whose representation callbacks must never execute."""

    def __init__(self) -> None:
        self.str_calls = 0
        self.repr_calls = 0

    def __str__(self) -> str:
        self.str_calls += 1
        raise RuntimeError("METHOD_STR_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("METHOD_REPR_SENTINEL")


class _HostileString(str):
    """String subclass that must not gain control of normalization callbacks."""

    def lower(self) -> str:
        raise RuntimeError("METHOD_LOWER_SENTINEL")

    def __str__(self) -> str:
        raise RuntimeError("METHOD_STR_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("METHOD_REPR_SENTINEL")


class _HostileQuadrature(int):
    """Integer subclass whose coercion callback must never execute."""

    def __new__(cls, value: int):
        instance = super().__new__(cls, value)
        instance.int_calls = 0
        return instance

    def __int__(self) -> int:
        self.int_calls += 1
        raise RuntimeError("Q_THETA_INT_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("Q_THETA_REPR_SENTINEL")


class _HostileNumpyQuadrature(np.int64):
    """NumPy-integer subclass whose coercion callback must never execute."""

    def __int__(self) -> int:
        raise RuntimeError("Q_THETA_NUMPY_INT_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("Q_THETA_NUMPY_REPR_SENTINEL")


def _anchors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tiny, finite common-item parameter vectors."""
    return (
        np.array([1.0, 1.2, 0.8], dtype=np.float64),
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        np.array([1.1, 1.0, 0.9], dtype=np.float64),
        np.array([-0.9, 0.1, 1.1], dtype=np.float64),
    )


def _result_payload() -> dict[str, object]:
    """Return the minimum successful Rust-shaped IRT-link result payload."""
    return {
        "slope": 1.0,
        "intercept": 0.0,
        "criterion": 0.0,
        "n_iter": 0,
        "converged": True,
        "termination_reason": "closed_form",
        "max_iter": 0,
        "final_objective_span": 0.0,
        "objective_tolerance": 0.0,
        "final_parameter_span": 0.0,
        "parameter_tolerance": 0.0,
    }


def test_irt_link_rejects_hostile_method_before_core_loader(monkeypatch) -> None:
    """An alien method fails closed without callbacks or native-loader access."""
    loader_calls = 0

    def forbidden_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("CORE_LOADER_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_loader)
    hostile = _HostileMethod()

    with pytest.raises(ValueError, match="method"):
        irt_link(*_anchors(), method=hostile, q_theta=7)

    assert loader_calls == 0
    assert hostile.str_calls == 0
    assert hostile.repr_calls == 0


def test_irt_link_rejects_string_subclass_before_normalization(monkeypatch) -> None:
    """A string subclass cannot execute overridden normalization callbacks."""
    loader_calls = 0

    def forbidden_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("CORE_LOADER_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_loader)

    with pytest.raises(ValueError, match="method"):
        irt_link(*_anchors(), method=_HostileString("stocking_lord"), q_theta=7)

    assert loader_calls == 0


def test_irt_link_rejects_unsupported_builtin_method_before_core(monkeypatch) -> None:
    """Unsupported trusted text fails before native dispatch or loader work."""
    loader_calls = 0

    def forbidden_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("CORE_LOADER_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_loader)

    with pytest.raises(ValueError, match="method"):
        irt_link(*_anchors(), method="unsupported", q_theta=7)

    assert loader_calls == 0


@pytest.mark.parametrize(
    "q_theta_type",
    [_HostileQuadrature, _HostileNumpyQuadrature],
)
def test_irt_link_rejects_integer_subclasses_before_core_loader(
    monkeypatch,
    q_theta_type: type,
) -> None:
    """Quadrature subclasses fail closed before native-loader or coercion hooks."""
    loader_calls = 0

    def forbidden_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("CORE_LOADER_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_loader)
    q_theta = q_theta_type(7)

    with pytest.raises(ValueError, match="q_theta"):
        irt_link(*_anchors(), method="stocking_lord", q_theta=q_theta)

    assert loader_calls == 0
    if isinstance(q_theta, _HostileQuadrature):
        assert q_theta.int_calls == 0


def test_irt_link_rejects_hostile_type_hash_before_core_loader(monkeypatch) -> None:
    """Quadrature admission must not hash a caller-controlled scalar type."""

    calls: list[str] = []
    loader_calls = 0

    class HostileMeta(type):
        def __hash__(cls) -> int:
            calls.append("type-__hash__")
            raise AssertionError("type hash callback executed")

    class HostileNumpyQuadrature(np.int64, metaclass=HostileMeta):
        pass

    def forbidden_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("CORE_LOADER_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_loader)

    with pytest.raises(ValueError, match="q_theta"):
        irt_link(*_anchors(), method="stocking_lord", q_theta=HostileNumpyQuadrature(7))

    assert calls == []
    assert loader_calls == 0


def test_irt_link_preserves_supported_alias_without_restringifying(monkeypatch) -> None:
    """A trusted Rust-supported alias crosses the boundary unchanged."""
    calls: list[str] = []

    class Core:
        """Stub core that captures the validated linking method."""

        def irt_link(self, *args, method):
            calls.append(method)
            return _result_payload()

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    result = irt_link(*_anchors(), method="SL", q_theta=7)

    assert isinstance(result, IrtLinkResult)
    assert calls == ["SL"]
    assert result.method == "SL"


def test_irt_link_preserves_genuine_numpy_quadrature_scalar(monkeypatch) -> None:
    """A genuine NumPy integer retains the established quadrature contract."""
    calls: list[str] = []

    class Core:
        """Stub core that records successful dispatch after trusted coercion."""

        def irt_link(self, *args, method):
            calls.append(method)
            return _result_payload()

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())

    result = irt_link(*_anchors(), method="mean_mean", q_theta=np.int64(7))

    assert isinstance(result, IrtLinkResult)
    assert calls == ["mean_mean"]
