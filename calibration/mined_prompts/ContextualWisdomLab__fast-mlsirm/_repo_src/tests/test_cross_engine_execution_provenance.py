"""Execution/provenance consistency regressions for cross-engine conformance."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fast_mlsirm.cross_engine_conformance import (
    ComparisonEngine,
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceEvidence,
    ConformanceExecutionStatus,
    ConformanceInventory,
    ConformanceLayer,
    ConformanceRunProvenance,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SOURCE_COMMIT = "0" * 40


class _HostileText(str):
    """String subtype that records unsafe text callback dispatch."""

    callbacks = 0

    def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Reject normalization before exact built-in text admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile strip callback executed")


def _engine() -> ComparisonEngine:
    """Return one isolated comparison-engine identity."""
    return ComparisonEngine(
        engine_id="mirt_engine",
        engine_version="1.44.0",
        source_reference="doi:10.18637/jss.v048.i06",
        license_classification="gpl_3_reviewed",
    )


def _evidence(
    status: ConformanceExecutionStatus,
) -> ConformanceEvidence:
    """Return one fixed-parameter evidence row with status-consistent artifact state."""
    executed = status in {
        ConformanceExecutionStatus.PASSED,
        ConformanceExecutionStatus.FAILED,
        ConformanceExecutionStatus.INDETERMINATE,
    }
    return ConformanceEvidence(
        evidence_id="rasch_probability_check",
        engine=_engine(),
        layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
        execution_status=status,
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256=_SHA_A,
        fixture_sha256=_SHA_B,
        environment_sha256=_SHA_C,
        artifact_sha256=_SHA_D if executed else None,
        limitation=None,
    )


def _capability(
    status: ConformanceExecutionStatus,
) -> ConformanceCapability:
    """Return a capability whose coverage state matches execution state."""
    executed = status in {
        ConformanceExecutionStatus.PASSED,
        ConformanceExecutionStatus.FAILED,
        ConformanceExecutionStatus.INDETERMINATE,
    }
    return ConformanceCapability(
        capability_id="rasch_probability",
        public_entrypoint="fast_mlsirm.rasch.probability",
        estimand="Dichotomous Rasch response probability",
        likelihood_family="bernoulli_logit",
        parameterization="difficulty with unit discrimination",
        identification="latent location fixed by the compared fixture",
        comparison_scope="fixed-parameter equation conformance",
        coverage_status=(
            ConformanceCoverageStatus.COVERED
            if executed
            else ConformanceCoverageStatus.PLANNED
        ),
        evidence=(_evidence(status),),
    )


def _provenance() -> ConformanceRunProvenance:
    """Return one fully content-addressed conformance-run record."""
    return ConformanceRunProvenance(
        harness_commit=_SOURCE_COMMIT,
        environment_sha256=_SHA_C,
        environment_kind="environment_lock",
        operating_system="linux",
        architecture="x86_64",
        rng_algorithm="pcg64_dxsm",
        rng_seeds=(17, 23),
        mapping_schema_version="1.0.0",
        mapping_sha256=_SHA_D,
        model_configuration_sha256=_SHA_A,
        convergence_controls_sha256=_SHA_B,
        tolerance_sha256=_SHA_A,
        tolerance_rationale="fixed-parameter double-precision comparison",
        raw_output_sha256=_SHA_B,
        normalized_output_sha256=_SHA_C,
        license_classification="synthetic_or_open",
        redistribution_status="redistributable",
    )


@pytest.mark.parametrize(
    "status",
    [
        ConformanceExecutionStatus.PASSED,
        ConformanceExecutionStatus.FAILED,
        ConformanceExecutionStatus.INDETERMINATE,
    ],
)
def test_executed_evidence_requires_run_provenance(
    status: ConformanceExecutionStatus,
) -> None:
    """An executed verdict cannot exist without reproducible run provenance."""
    with pytest.raises(
        ValueError,
        match="run_provenance is required for executed evidence",
    ):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(_capability(status),),
        )


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        (
            "raw_output_sha256",
            "raw_output_sha256 is required for executed evidence",
        ),
        (
            "normalized_output_sha256",
            "normalized_output_sha256 is required for executed evidence",
        ),
    ],
)
def test_executed_evidence_requires_raw_and_normalized_output_hashes(
    field_name: str,
    message: str,
) -> None:
    """Executed evidence remains fail-closed when either output lineage is absent."""
    provenance = replace(_provenance(), **{field_name: None})

    with pytest.raises(ValueError, match=message):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(_capability(ConformanceExecutionStatus.PASSED),),
            run_provenance=provenance,
        )


def test_mutated_output_hash_revalidates_before_execution_consistency_gate() -> None:
    """Rebound output text is rejected before caller normalization callbacks."""
    _HostileText.callbacks = 0
    provenance = _provenance()
    object.__setattr__(
        provenance,
        "raw_output_sha256",
        _HostileText(_SHA_B),
    )

    with pytest.raises(ValueError, match="raw_output_sha256 must be a string"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(_capability(ConformanceExecutionStatus.PASSED),),
            run_provenance=provenance,
        )

    assert _HostileText.callbacks == 0


def test_nonexecuted_plan_keeps_optional_output_hashes() -> None:
    """A planned, nonexecuted comparison may retain source-free run metadata only."""
    provenance = replace(
        _provenance(),
        raw_output_sha256=None,
        normalized_output_sha256=None,
    )
    inventory = ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(_capability(ConformanceExecutionStatus.NOT_EXECUTED),),
        run_provenance=provenance,
    )

    assert inventory.to_manifest()["run_provenance"]["raw_output_sha256"] is None
    assert (
        inventory.to_manifest()["run_provenance"]["normalized_output_sha256"]
        is None
    )


def test_executed_evidence_accepts_complete_output_provenance() -> None:
    """A fully content-addressed executed run remains accepted and serializable."""
    inventory = ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(_capability(ConformanceExecutionStatus.PASSED),),
        run_provenance=_provenance(),
    )

    manifest = inventory.to_manifest()
    assert manifest["run_provenance"]["raw_output_sha256"] == _SHA_B
    assert manifest["run_provenance"]["normalized_output_sha256"] == _SHA_C
