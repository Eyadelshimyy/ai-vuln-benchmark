"""Callback-safety regressions for enterprise due-diligence gate controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "enterprise_due_diligence_gate.py"
SOURCE_COMMIT = "a" * 40


def _load_module() -> ModuleType:
    """Load the enterprise gate script as an isolated module."""
    spec = importlib.util.spec_from_file_location("enterprise_due_diligence_gate_callbacks", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_module()


def test_build_gate_manifest_rejects_gate_name_subclass_without_callbacks() -> None:
    """Gate-name admission must reject a string subclass before text methods run."""
    callbacks: list[str] = []

    class HostileGateName(str):
        def strip(self, *args: object, **kwargs: object) -> str:
            callbacks.append("strip")
            return super().strip(*args, **kwargs)

    with pytest.raises(ValueError, match="gate_name must be an exact built-in string"):
        GATE.build_gate_manifest(
            source_commit=SOURCE_COMMIT,
            gate_name=HostileGateName(GATE.CANONICAL_GATE_NAME),
        )

    assert callbacks == []


def test_build_gate_manifest_rejects_currency_subclass_without_callbacks() -> None:
    """Currency admission must reject a string subclass before text methods run."""
    callbacks: list[str] = []

    class HostileCurrency(str):
        def strip(self, *args: object, **kwargs: object) -> str:
            callbacks.append("strip")
            return super().strip(*args, **kwargs)

    with pytest.raises(ValueError, match="currency_code must be an exact built-in string"):
        GATE.build_gate_manifest(
            source_commit=SOURCE_COMMIT,
            currency_code=HostileCurrency("KRW"),
        )

    assert callbacks == []


def test_build_gate_manifest_rejects_scenario_int_subclass_without_callbacks() -> None:
    """Scenario admission must reject an integer subclass before comparison runs."""
    callbacks: list[str] = []

    class HostileScenario(int):
        def __le__(self, other: object) -> bool:
            callbacks.append("le")
            return super().__le__(other)

    with pytest.raises(ValueError, match="scenario_amount must be a positive integer"):
        GATE.build_gate_manifest(
            source_commit=SOURCE_COMMIT,
            scenario_amount=HostileScenario(25),
        )

    assert callbacks == []
