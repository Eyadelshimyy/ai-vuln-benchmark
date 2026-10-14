"""Runtime-environment provenance regressions for cross-engine conformance."""

from __future__ import annotations

from dataclasses import replace

import pytest

from fast_mlsirm.cross_engine_conformance import (
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceInventory,
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

    def __hash__(self) -> int:
        """Reject enum lookup before exact built-in text admission."""
        type(self).callbacks += 1
        raise AssertionError("hostile hash callback executed")


def _provenance() -> ConformanceRunProvenance:
    """Return one source-free run record with the release-grade identities."""
    return ConformanceRunProvenance(
        harness_commit=_SOURCE_COMMIT,
        environment_sha256=_SHA_A,
        environment_kind="environment_lock",
        operating_system="linux",
        architecture="x86_64",
        rng_algorithm="pcg64_dxsm",
        rng_seeds=(17, 23),
        mapping_schema_version="1.0.0",
        mapping_sha256=_SHA_B,
        model_configuration_sha256=_SHA_C,
        convergence_controls_sha256=_SHA_D,
        tolerance_sha256=_SHA_A,
        tolerance_rationale="fixed-parameter double-precision comparison",
        raw_output_sha256=None,
        normalized_output_sha256=None,
        license_classification="synthetic_or_open",
        redistribution_status="redistributable",
    )


def _planned_inventory(provenance: ConformanceRunProvenance) -> ConformanceInventory:
    """Return one nonexecuted inventory whose identity includes run provenance."""
    capability = ConformanceCapability(
        capability_id="rasch_probability",
        public_entrypoint="fast_mlsirm.rasch.probability",
        estimand="Dichotomous Rasch response probability",
        likelihood_family="bernoulli_logit",
        parameterization="difficulty with unit discrimination",
        identification="latent location fixed by the compared fixture",
        comparison_scope="fixed-parameter equation conformance",
        coverage_status=ConformanceCoverageStatus.PLANNED,
        evidence=(),
    )
    return ConformanceInventory(
        package_version="0.8.0",
        source_commit=_SOURCE_COMMIT,
        capabilities=(capability,),
        run_provenance=provenance,
    )


def test_run_manifest_captures_runtime_configuration_and_redistribution_identity() -> None:
    """Release-grade manifests expose every required source-free runtime identity."""
    manifest = _provenance().to_manifest()

    assert manifest["environment_kind"] == "environment_lock"
    assert manifest["operating_system"] == "linux"
    assert manifest["architecture"] == "x86_64"
    assert manifest["model_configuration_sha256"] == _SHA_C
    assert manifest["convergence_controls_sha256"] == _SHA_D
    assert manifest["redistribution_status"] == "redistributable"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("environment_kind", "container_image"),
        ("operating_system", "freebsd"),
        ("architecture", "aarch64"),
        ("model_configuration_sha256", _SHA_D),
        ("convergence_controls_sha256", _SHA_C),
        ("redistribution_status", "metadata_only"),
    ],
)
def test_inventory_fingerprint_binds_each_runtime_provenance_identity(
    field_name: str,
    replacement: str,
) -> None:
    """Changing any release-grade runtime identity changes inventory provenance."""
    baseline = _planned_inventory(_provenance())
    changed = _planned_inventory(replace(_provenance(), **{field_name: replacement}))

    assert baseline.inventory_fingerprint != changed.inventory_fingerprint


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("environment_kind", "docker_image", "environment_kind must be one of"),
        ("operating_system", " ", "operating_system must not be empty"),
        ("architecture", " ", "architecture must not be empty"),
        (
            "model_configuration_sha256",
            "bad",
            "model_configuration_sha256 must be a lowercase SHA-256 hex digest",
        ),
        (
            "convergence_controls_sha256",
            "bad",
            "convergence_controls_sha256 must be a lowercase SHA-256 hex digest",
        ),
        (
            "redistribution_status",
            "open",
            "redistribution_status must be one of",
        ),
    ],
)
def test_runtime_provenance_rejects_malformed_release_controls(
    field_name: str,
    replacement: str,
    message: str,
) -> None:
    """Runtime provenance fails closed on malformed release-identifying controls."""
    with pytest.raises(ValueError, match=message):
        replace(_provenance(), **{field_name: replacement})


def test_mutated_runtime_text_revalidates_before_caller_callbacks() -> None:
    """Post-construction runtime mutation is sealed before text normalization."""
    _HostileText.callbacks = 0
    provenance = _provenance()
    object.__setattr__(provenance, "operating_system", _HostileText("linux"))

    with pytest.raises(ValueError, match="operating_system must be a string"):
        provenance.to_manifest()

    assert _HostileText.callbacks == 0


def test_runtime_enums_reject_string_subclasses_before_lookup_callbacks() -> None:
    """Serialized environment and redistribution enums admit built-in text only."""
    _HostileText.callbacks = 0

    with pytest.raises(ValueError, match="environment_kind must be a supported"):
        replace(_provenance(), environment_kind=_HostileText("container_image"))
    with pytest.raises(ValueError, match="redistribution_status must be a supported"):
        replace(_provenance(), redistribution_status=_HostileText("metadata_only"))

    assert _HostileText.callbacks == 0
