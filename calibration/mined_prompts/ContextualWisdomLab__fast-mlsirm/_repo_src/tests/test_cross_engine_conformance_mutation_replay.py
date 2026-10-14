"""Replay-integrity regressions for exact conformance records mutated after init."""

from __future__ import annotations

import pytest

from fast_mlsirm.cross_engine_conformance import (
    ComparisonEngine,
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceEnvironmentKind,
    ConformanceEvidence,
    ConformanceExecutionStatus,
    ConformanceInventory,
    ConformanceLayer,
    ConformanceRedistributionStatus,
    ConformanceRunProvenance,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SOURCE_COMMIT = "0" * 40


class _HostileStatus:
    """Record whether replay dereferences an unvalidated semantic control."""

    callbacks = 0

    def __getattribute__(self, name: str):
        if name == "value":
            type(self).callbacks += 1
            raise AssertionError("unvalidated status callback executed")
        return object.__getattribute__(self, name)


class _HostileTuple(tuple):
    """Record whether replay iterates an unvalidated container."""

    callbacks = 0

    def __iter__(self):
        type(self).callbacks += 1
        raise AssertionError("unvalidated capabilities iteration executed")


def _engine() -> ComparisonEngine:
    """Build the minimal engine record used by replay-integrity fixtures."""
    return ComparisonEngine(
        engine_id="mirt_engine",
        engine_version="1.44.0",
        source_reference="doi:10.18637/jss.v048.i06",
        license_classification="gpl_3_reviewed",
    )


def _evidence() -> ConformanceEvidence:
    """Build one executed evidence record for mutation-replay tests."""
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
    """Build one executed capability record for mutation-replay tests."""
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


def _run_provenance() -> ConformanceRunProvenance:
    """Build execution provenance required by the executed fixture evidence."""
    return ConformanceRunProvenance(
        harness_commit=_SOURCE_COMMIT,
        environment_sha256=_SHA_C,
        environment_kind=ConformanceEnvironmentKind.ENVIRONMENT_LOCK,
        operating_system="linux",
        architecture="x86_64",
        rng_algorithm="pcg64_dxsm",
        rng_seeds=(17,),
        mapping_schema_version="1.0.0",
        mapping_sha256=_SHA_D,
        model_configuration_sha256=_SHA_A,
        convergence_controls_sha256=_SHA_B,
        tolerance_sha256=_SHA_C,
        tolerance_rationale="fixed-parameter mutation replay",
        raw_output_sha256=_SHA_A,
        normalized_output_sha256=_SHA_B,
        license_classification="synthetic_or_open",
        redistribution_status=ConformanceRedistributionStatus.METADATA_ONLY,
    )


def test_engine_manifest_revalidates_exact_record_after_post_init_mutation() -> None:
    """A forged exact engine cannot emit a noncanonical identity after creation."""
    engine = _engine()
    object.__setattr__(engine, "engine_id", "not-canonical")

    with pytest.raises(ValueError, match="engine_id"):
        engine.to_manifest()


def test_evidence_manifest_rejects_mutated_status_before_attribute_callback() -> None:
    """Evidence replay seals semantic status again before reading enum fields."""
    evidence = _evidence()
    _HostileStatus.callbacks = 0
    object.__setattr__(evidence, "execution_status", _HostileStatus())

    with pytest.raises(ValueError, match="execution_status"):
        evidence.to_manifest()

    assert _HostileStatus.callbacks == 0


def test_capability_manifest_rejects_mutated_status_before_attribute_callback() -> None:
    """Capability replay seals coverage status again before reading enum fields."""
    capability = _capability()
    _HostileStatus.callbacks = 0
    object.__setattr__(capability, "coverage_status", _HostileStatus())

    with pytest.raises(ValueError, match="coverage_status"):
        capability.to_manifest()

    assert _HostileStatus.callbacks == 0


def test_inventory_manifest_rejects_mutated_container_before_iteration_callback() -> None:
    """Inventory replay refuses a rebound capabilities container before iteration."""
    inventory = ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(_capability(),),
        run_provenance=_run_provenance(),
    )
    _HostileTuple.callbacks = 0
    object.__setattr__(inventory, "capabilities", _HostileTuple((_capability(),)))

    with pytest.raises(ValueError, match="capabilities must be a list or tuple"):
        inventory.to_manifest()

    assert _HostileTuple.callbacks == 0
