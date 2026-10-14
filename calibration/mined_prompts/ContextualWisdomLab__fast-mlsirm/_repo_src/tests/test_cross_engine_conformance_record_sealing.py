"""Regression tests for sealed cross-engine conformance package records."""

from __future__ import annotations

import pytest

from fast_mlsirm.cross_engine_conformance import (
    ComparisonEngine,
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceEvidence,
    ConformanceExecutionStatus,
    ConformanceInventory,
    ConformanceLayer,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SOURCE_COMMIT = "0" * 40


class _HostileMixin:
    """Record any field access that occurs before exact-record rejection."""

    callbacks = 0
    guarded_fields: frozenset[str] = frozenset()

    def __getattribute__(self, name: str):  # type: ignore[no-untyped-def]
        """Fail if package validation reads subclass-controlled fields."""
        guarded = object.__getattribute__(self, "guarded_fields")
        if name in guarded:
            type(self).callbacks += 1
            raise AssertionError("subclass field callback executed")
        return super().__getattribute__(name)


class _HostileEngine(_HostileMixin, ComparisonEngine):
    """Comparison-engine subclass that must never reach field normalization."""

    guarded_fields = frozenset(
        {"engine_id", "engine_version", "source_reference", "license_classification"}
    )


class _HostileEvidence(_HostileMixin, ConformanceEvidence):
    """Evidence subclass that must never reach nested-record normalization."""

    guarded_fields = frozenset({"evidence_id", "engine", "layer", "execution_status"})


class _HostileCapability(_HostileMixin, ConformanceCapability):
    """Capability subclass that must never reach semantic field normalization."""

    guarded_fields = frozenset({"capability_id", "public_entrypoint", "evidence"})


class _HostileInventory(_HostileMixin, ConformanceInventory):
    """Inventory subclass that must never reach provenance normalization."""

    guarded_fields = frozenset({"package_version", "source_commit", "capabilities"})


def _engine() -> ComparisonEngine:
    """Return one valid exact comparison-engine record."""
    return ComparisonEngine(
        engine_id="mirt_engine",
        engine_version="1.44.0",
        source_reference="doi:10.18637/jss.v048.i06",
        license_classification="gpl_3_reviewed",
    )


def _evidence() -> ConformanceEvidence:
    """Return one valid exact evidence record."""
    return ConformanceEvidence(
        evidence_id="rasch_probability_check",
        engine=_engine(),
        layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
        execution_status=ConformanceExecutionStatus.PASSED,
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256=_SHA_A,
        fixture_sha256=_SHA_B,
        environment_sha256=_SHA_C,
        artifact_sha256=_SHA_D,
    )


def _capability() -> ConformanceCapability:
    """Return one valid exact capability record."""
    return ConformanceCapability(
        capability_id="rasch_probability",
        public_entrypoint="fast_mlsirm.rasch.probability",
        estimand="Dichotomous Rasch response probability",
        likelihood_family="bernoulli_logit",
        parameterization="difficulty with unit discrimination",
        identification="latent location fixed by the compared fixture",
        comparison_scope="fixed-parameter equation conformance",
        coverage_status=ConformanceCoverageStatus.COVERED,
        evidence=(_evidence(),),
    )


def test_engine_subclass_is_rejected_before_field_callbacks() -> None:
    """Engine construction rejects subclass identity before reading its fields."""
    _HostileEngine.callbacks = 0
    with pytest.raises(ValueError, match="ComparisonEngine must be an exact"):
        _HostileEngine(
            engine_id="mirt_engine",
            engine_version="1.44.0",
            source_reference="doi:10.18637/jss.v048.i06",
            license_classification="gpl_3_reviewed",
        )
    assert _HostileEngine.callbacks == 0


def test_evidence_subclass_is_rejected_before_field_callbacks() -> None:
    """Evidence construction rejects subclass identity before reading its fields."""
    _HostileEvidence.callbacks = 0
    with pytest.raises(ValueError, match="ConformanceEvidence must be an exact"):
        _HostileEvidence(
            evidence_id="rasch_probability_check",
            engine=_engine(),
            layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
            execution_status=ConformanceExecutionStatus.PASSED,
            parameter_mapping_version="1.0.0",
            parameter_mapping_sha256=_SHA_A,
            fixture_sha256=_SHA_B,
            environment_sha256=_SHA_C,
            artifact_sha256=_SHA_D,
        )
    assert _HostileEvidence.callbacks == 0


def test_capability_subclass_is_rejected_before_field_callbacks() -> None:
    """Capability construction rejects subclass identity before field access."""
    _HostileCapability.callbacks = 0
    with pytest.raises(ValueError, match="ConformanceCapability must be an exact"):
        _HostileCapability(
            capability_id="rasch_probability",
            public_entrypoint="fast_mlsirm.rasch.probability",
            estimand="Dichotomous Rasch response probability",
            likelihood_family="bernoulli_logit",
            parameterization="difficulty with unit discrimination",
            identification="latent location fixed by the compared fixture",
            comparison_scope="fixed-parameter equation conformance",
            coverage_status=ConformanceCoverageStatus.COVERED,
            evidence=(_evidence(),),
        )
    assert _HostileCapability.callbacks == 0


def test_inventory_subclass_is_rejected_before_field_callbacks() -> None:
    """Inventory construction rejects subclass identity before field access."""
    _HostileInventory.callbacks = 0
    with pytest.raises(ValueError, match="ConformanceInventory must be an exact"):
        _HostileInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(_capability(),),
        )
    assert _HostileInventory.callbacks == 0
