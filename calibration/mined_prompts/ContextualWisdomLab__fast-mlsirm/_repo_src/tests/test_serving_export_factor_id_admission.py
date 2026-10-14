"""Serving-export regressions for factor, item, and dimension identity admission."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pytest

from fast_mlsirm import serving
from fast_mlsirm.types import FitResult, MLSIRMParams


class _HostileInteger:
    """Integer-like object whose coercion must not run during factor admission."""

    def __init__(self) -> None:
        self.calls = 0

    def __int__(self) -> int:
        self.calls += 1
        raise AssertionError("caller-controlled integer conversion executed")

    def __index__(self) -> int:
        self.calls += 1
        raise AssertionError("caller-controlled index conversion executed")


class _HostileItemCodes(list[str]):
    """List subclass whose container callbacks must never run during export."""

    def __init__(self, values: list[str]) -> None:
        super().__init__(values)
        self.calls = 0

    def _trip(self) -> NoReturn:
        self.calls += 1
        raise AssertionError("caller-controlled item-code container callback executed")

    def __len__(self) -> int:
        return self._trip()

    def __iter__(self):
        return self._trip()

    def __getitem__(self, index):
        return self._trip()


class _HostileItemCode(str):
    """String subclass that must not survive serving-artifact admission."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def _trip(self) -> NoReturn:
        self.calls += 1
        raise AssertionError("caller-controlled item-code scalar callback executed")

    def __str__(self) -> str:
        return self._trip()

    def __hash__(self) -> int:
        return self._trip()

    def __eq__(self, other):
        return self._trip()


class _HostileDimNames(list[str]):
    """List subclass whose callbacks must not run during dimension-name admission."""

    def __init__(self, values: list[str]) -> None:
        super().__init__(values)
        self.calls = 0

    def _trip(self) -> NoReturn:
        self.calls += 1
        raise AssertionError("caller-controlled dimension-name container callback executed")

    def __len__(self) -> int:
        return self._trip()

    def __iter__(self):
        return self._trip()

    def __getitem__(self, index):
        return self._trip()


class _HostileDimName(str):
    """String subclass that must not enter the frozen dimension-name artifact."""

    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def _trip(self) -> NoReturn:
        self.calls += 1
        raise AssertionError("caller-controlled dimension-name scalar callback executed")

    def __str__(self) -> str:
        return self._trip()

    def __hash__(self) -> int:
        return self._trip()

    def __eq__(self, other):
        return self._trip()


def _result() -> FitResult:
    """Return a minimal converged two-item result for export-boundary tests."""
    params = MLSIRMParams(
        theta=np.zeros((1, 2), dtype=np.float64),
        alpha=np.zeros(2, dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((2, 1), dtype=np.float64),
        tau=0.0,
    )
    return FitResult(
        params=params,
        model="MLS2PLM",
        optimizer="em",
        backend="rust",
        rust_device="cpu",
        objective=0.0,
        loglik_trace=[0.0],
        objective_trace=[],
        convergence_status="converged",
        n_iter=1,
    )


def _core_must_not_be_discovered() -> None:
    raise AssertionError("invalid serving evidence reached compiled-core discovery")


@pytest.mark.parametrize(
    "factor_id",
    [
        np.array([0, np.uint64(2**63)], dtype=np.uint64),
        np.array([0.0, 0.5], dtype=np.float64),
        np.array([0.0 + 0.0j, 1.0 + 0.5j], dtype=np.complex128),
        np.array([-1, 0], dtype=np.int64),
    ],
)
def test_export_rejects_lossy_factor_identity_before_native_discovery(
    monkeypatch, factor_id
):
    """Factor identities must not wrap, truncate, project, or go negative."""
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="factor_id"):
        serving.export_serving_bundle(_result(), ["q0", "q1"], factor_id)


def test_export_rejects_object_factor_identity_without_integer_callback(monkeypatch):
    """Object-backed factor evidence must fail before per-cell conversion."""
    hostile = _HostileInteger()
    factor_id = np.array([0, hostile], dtype=object)
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="factor_id"):
        serving.export_serving_bundle(_result(), ["q0", "q1"], factor_id)

    assert hostile.calls == 0


def test_export_preserves_trusted_integer_valued_factor_sequence(monkeypatch):
    """Ordinary trusted integer-valued factor evidence keeps its Rust payload."""
    seen: dict[str, object] = {}

    def _eapsum_tables(alpha, b, zeta, tau, factor_id, *args, **kwargs):
        seen["factor_id"] = factor_id
        return []

    monkeypatch.setattr(
        serving,
        "_core_module",
        lambda: SimpleNamespace(eapsum_tables=_eapsum_tables),
    )

    bundle = serving.export_serving_bundle(
        _result(),
        ["q0", "q1"],
        (np.uint8(0), 1.0),
    )

    factor_id = seen["factor_id"]
    assert isinstance(factor_id, np.ndarray)
    assert factor_id.dtype == np.int64
    assert factor_id.tolist() == [0, 1]
    assert bundle["n_dims"] == 2
    assert [item["factor_id"] for item in bundle["items"]] == [0, 1]


def test_export_rejects_callback_bearing_item_code_container_before_native(
    monkeypatch,
):
    """Item-code container protocols must not run while artifact identity is admitted."""
    item_codes = _HostileItemCodes(["q0", "q1"])
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="item_codes"):
        serving.export_serving_bundle(_result(), item_codes, (0, 1))

    assert item_codes.calls == 0


def test_export_rejects_callback_bearing_item_code_scalar_before_native(monkeypatch):
    """Caller-defined string identities must not enter the frozen serving bundle."""
    item_code = _HostileItemCode("q1")
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="item_codes"):
        serving.export_serving_bundle(_result(), ["q0", item_code], (0, 1))

    assert item_code.calls == 0


def test_export_preserves_trusted_item_code_tuple(monkeypatch):
    """Ordinary built-in tuple item identities retain historical export compatibility."""
    monkeypatch.setattr(
        serving,
        "_core_module",
        lambda: SimpleNamespace(eapsum_tables=lambda *args, **kwargs: []),
    )

    bundle = serving.export_serving_bundle(_result(), ("q0", "q1"), (0, 1))

    assert [item["code"] for item in bundle["items"]] == ["q0", "q1"]


def test_export_rejects_callback_bearing_dimension_name_container_before_native(
    monkeypatch,
):
    """Dimension-name container protocols must not run before compiled-core work."""
    dim_names = _HostileDimNames(["primary", "secondary"])
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="dim_names"):
        serving.export_serving_bundle(
            _result(), ["q0", "q1"], (0, 1), dim_names=dim_names
        )

    assert dim_names.calls == 0


def test_export_rejects_callback_bearing_dimension_name_scalar_before_native(
    monkeypatch,
):
    """Caller-defined dimension strings must not enter the frozen serving bundle."""
    dim_name = _HostileDimName("secondary")
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="dim_names"):
        serving.export_serving_bundle(
            _result(), ["q0", "q1"], (0, 1), dim_names=["primary", dim_name]
        )

    assert dim_name.calls == 0


def test_export_rejects_wrong_dimension_name_count_before_native(monkeypatch):
    """Dimension-label cardinality is validated before Rust EAPsum generation."""
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match="dim_names length"):
        serving.export_serving_bundle(
            _result(), ["q0", "q1"], (0, 1), dim_names=["primary"]
        )


@pytest.mark.parametrize("control_name", ["q_theta", "q_xi"])
def test_export_rejects_hostile_quadrature_integer_before_native_discovery(
    monkeypatch, control_name
):
    """Quadrature integer protocols must not execute before package admission."""
    hostile = _HostileInteger()
    monkeypatch.setattr(serving, "_core_module", _core_must_not_be_discovered)

    with pytest.raises(ValueError, match=control_name):
        serving.export_serving_bundle(
            _result(),
            ["q0", "q1"],
            (0, 1),
            **{control_name: hostile},
        )

    assert hostile.calls == 0


def test_export_normalizes_trusted_numpy_quadrature_controls_for_json(
    monkeypatch, tmp_path
):
    """Trusted NumPy integer controls must produce a self-serializable bundle."""
    seen: dict[str, object] = {}

    def _eapsum_tables(alpha, b, zeta, tau, factor_id, *args, **kwargs):
        seen["q_theta"] = kwargs["q_theta"]
        seen["q_xi"] = kwargs["q_xi"]
        return []

    monkeypatch.setattr(
        serving,
        "_core_module",
        lambda: SimpleNamespace(eapsum_tables=_eapsum_tables),
    )
    path = tmp_path / "bundle.json"

    bundle = serving.export_serving_bundle(
        _result(),
        ["q0", "q1"],
        (0, 1),
        path=path,
        q_theta=np.int64(21),
        q_xi=np.uint8(11),
    )

    assert type(bundle["quadrature"]["q_theta"]) is int
    assert type(bundle["quadrature"]["q_xi"]) is int
    assert type(seen["q_theta"]) is int
    assert type(seen["q_xi"]) is int
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["quadrature"] == {"q_theta": 21, "q_xi": 11}
