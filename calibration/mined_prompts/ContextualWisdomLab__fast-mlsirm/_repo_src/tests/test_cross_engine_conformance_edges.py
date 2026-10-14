"""Fail-closed edge coverage for cross-engine conformance contracts."""

from __future__ import annotations

import pytest

import fast_mlsirm.cross_engine_conformance as conformance
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


class _HostileText(str):
    """Text subclass that records callback dispatch."""

    callbacks = 0

    def strip(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Fail if package code dispatches caller-defined normalization."""
        type(self).callbacks += 1
        raise AssertionError("hostile strip callback executed")


def _engine() -> ComparisonEngine:
    """Return one valid comparison-engine identity."""
    return ComparisonEngine(
        engine_id="mirt_engine",
        engine_version="1.44.0",
        source_reference="doi:10.18637/jss.v048.i06",
        license_classification="gpl_3_reviewed",
    )


def _evidence(
    *,
    evidence_id: str = "rasch_probability_check",
    status: ConformanceExecutionStatus = ConformanceExecutionStatus.PASSED,
    artifact_sha256: str | None = _SHA_D,
    limitation: str | None = None,
) -> ConformanceEvidence:
    """Return one valid evidence record with adjustable verdict state."""
    return ConformanceEvidence(
        evidence_id=evidence_id,
        engine=_engine(),
        layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
        execution_status=status,
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256=_SHA_A,
        fixture_sha256=_SHA_B,
        environment_sha256=_SHA_C,
        artifact_sha256=artifact_sha256,
        limitation=limitation,
    )


def _capability(
    *,
    capability_id: str = "rasch_probability",
    coverage_status: ConformanceCoverageStatus = ConformanceCoverageStatus.COVERED,
    evidence: object | None = None,
    schema_version: str = "1.0",
) -> ConformanceCapability:
    """Return one valid capability unless a caller supplies an edge value."""
    rows: object = (_evidence(),) if evidence is None else evidence
    return ConformanceCapability(
        capability_id=capability_id,
        public_entrypoint="fast_mlsirm.rasch.probability",
        estimand="Dichotomous Rasch response probability",
        likelihood_family="bernoulli_logit",
        parameterization="difficulty with unit discrimination",
        identification="latent location fixed by the compared fixture",
        comparison_scope="fixed-parameter equation conformance",
        coverage_status=coverage_status,
        evidence=rows,  # type: ignore[arg-type]
        schema_version=schema_version,
    )


def test_text_helper_rejects_callbacks_empty_and_oversize_values() -> None:
    """Bounded exact strings cover every scalar admission branch."""
    _HostileText.callbacks = 0
    with pytest.raises(ValueError, match="field must be a string"):
        conformance._text(_HostileText("value"), "field")
    assert _HostileText.callbacks == 0

    with pytest.raises(ValueError, match="field must not be empty"):
        conformance._text("   ", "field")
    with pytest.raises(ValueError, match="field must contain at most 3 characters"):
        conformance._text("abcd", "field", maximum=3)

    assert conformance._text("  value  ", "field") == "value"


def test_identifier_and_digest_helpers_cover_valid_and_invalid_shapes() -> None:
    """Identifiers and immutable hashes fail closed on malformed scalar text."""
    assert conformance._identifier("valid_name", "field") == "valid_name"
    with pytest.raises(ValueError, match="two-or-more-token lower snake_case"):
        conformance._identifier("invalid", "field")

    assert conformance._fingerprint(_SHA_A, "digest") == _SHA_A
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        conformance._fingerprint("A" * 64, "digest")
    assert conformance._optional_fingerprint(None, "digest") is None
    assert conformance._optional_fingerprint(_SHA_B, "digest") == _SHA_B

    assert conformance._git_sha(_SOURCE_COMMIT, "commit") == _SOURCE_COMMIT
    with pytest.raises(ValueError, match="full lowercase 40- or 64-character Git SHA"):
        conformance._git_sha("0" * 39, "commit")
    assert conformance._git_sha("0" * 64, "commit") == "0" * 64


def test_version_and_schema_helpers_fail_closed() -> None:
    """Only canonical numeric versions and the implemented schema are admitted."""
    assert conformance._semantic_version("1.2.3", "version") == "1.2.3"
    with pytest.raises(ValueError, match="canonical semantic version"):
        conformance._semantic_version("01.2.3", "version")

    assert conformance._schema_version("1.0") == "1.0"
    with pytest.raises(ValueError, match=r"schema_version must be '1\.0'"):
        conformance._schema_version("2.0")


def test_enum_helper_admits_exact_members_and_serialized_values() -> None:
    """Enums reject caller objects and unsupported serialized states."""
    member = ConformanceLayer.FITTED_RESULT
    assert conformance._enum_value(member, ConformanceLayer, "layer") is member
    assert (
        conformance._enum_value("fitted_result", ConformanceLayer, "layer")
        is ConformanceLayer.FITTED_RESULT
    )

    with pytest.raises(ValueError, match="supported ConformanceLayer value"):
        conformance._enum_value(7, ConformanceLayer, "layer")
    with pytest.raises(ValueError, match="layer must be one of"):
        conformance._enum_value("unknown", ConformanceLayer, "layer")


def test_engine_normalizes_manifest_and_rejects_bad_identifiers() -> None:
    """External-engine provenance stays bounded and machine-readable."""
    engine = ComparisonEngine(
        engine_id=" mirt_engine ",
        engine_version=" 1.44.0 ",
        source_reference=" doi:10.18637/jss.v048.i06 ",
        license_classification=" gpl_3_reviewed ",
    )
    assert engine.to_manifest() == {
        "engine_id": "mirt_engine",
        "engine_version": "1.44.0",
        "license_classification": "gpl_3_reviewed",
        "source_reference": "doi:10.18637/jss.v048.i06",
    }

    with pytest.raises(ValueError, match="engine_id must use"):
        ComparisonEngine(
            engine_id="mirt",
            engine_version="1.44.0",
            source_reference="doi:10.18637/jss.v048.i06",
            license_classification="gpl_3_reviewed",
        )
    with pytest.raises(ValueError, match="license_classification must use"):
        ComparisonEngine(
            engine_id="mirt_engine",
            engine_version="1.44.0",
            source_reference="doi:10.18637/jss.v048.i06",
            license_classification="GPL-3.0",
        )


def test_nested_engine_requires_exact_package_record() -> None:
    """Evidence cannot delegate validation to arbitrary engine-like objects."""
    with pytest.raises(ValueError, match="engine must be a ComparisonEngine"):
        ConformanceEvidence(
            evidence_id="rasch_probability_check",
            engine=object(),  # type: ignore[arg-type]
            layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
            execution_status=ConformanceExecutionStatus.PASSED,
            parameter_mapping_version="1.0.0",
            parameter_mapping_sha256=_SHA_A,
            fixture_sha256=_SHA_B,
            environment_sha256=_SHA_C,
            artifact_sha256=_SHA_D,
        )


def test_evidence_accepts_serialized_enums_and_normalizes_limitation() -> None:
    """Evidence manifests preserve explicit verdict and provenance fields."""
    evidence = ConformanceEvidence(
        evidence_id="rasch_probability_check",
        engine=_engine(),
        layer="fitted_result",  # type: ignore[arg-type]
        execution_status="failed",  # type: ignore[arg-type]
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256=_SHA_A,
        fixture_sha256=_SHA_B,
        environment_sha256=_SHA_C,
        artifact_sha256=_SHA_D,
        limitation=" optimizer stopped at the preregistered iteration cap ",
    )
    manifest = evidence.to_manifest()
    assert evidence.layer is ConformanceLayer.FITTED_RESULT
    assert evidence.execution_status is ConformanceExecutionStatus.FAILED
    assert evidence.limitation == "optimizer stopped at the preregistered iteration cap"
    assert manifest["execution_status"] == "failed"
    assert manifest["layer"] == "fitted_result"
    assert manifest["artifact_sha256"] == _SHA_D


def test_evidence_rejects_invalid_versions_digests_and_enums() -> None:
    """Evidence provenance cannot be malformed or use undeclared states."""
    kwargs = {
        "evidence_id": "rasch_probability_check",
        "engine": _engine(),
        "layer": ConformanceLayer.FIXED_PARAMETER_EQUATION,
        "execution_status": ConformanceExecutionStatus.PASSED,
        "parameter_mapping_version": "1.0.0",
        "parameter_mapping_sha256": _SHA_A,
        "fixture_sha256": _SHA_B,
        "environment_sha256": _SHA_C,
        "artifact_sha256": _SHA_D,
    }

    with pytest.raises(ValueError, match="canonical semantic version"):
        ConformanceEvidence(**{**kwargs, "parameter_mapping_version": "v1"})
    with pytest.raises(ValueError, match="parameter_mapping_sha256 must be"):
        ConformanceEvidence(**{**kwargs, "parameter_mapping_sha256": "bad"})
    with pytest.raises(ValueError, match="execution_status must be one of"):
        ConformanceEvidence(**{**kwargs, "execution_status": "unknown"})
    with pytest.raises(ValueError, match="layer must be one of"):
        ConformanceEvidence(**{**kwargs, "layer": "unknown"})


def test_evidence_collection_rejects_container_size_type_and_duplicates() -> None:
    """Evidence collections are bounded exact containers with unique identities."""
    with pytest.raises(ValueError, match="evidence must be a list or tuple"):
        _capability(evidence={_evidence()})

    too_many = [_evidence(evidence_id=f"engine_check_{index}") for index in range(129)]
    with pytest.raises(ValueError, match="evidence must contain at most 128 values"):
        _capability(evidence=too_many)

    with pytest.raises(ValueError, match=r"evidence\[0\] must be a ConformanceEvidence"):
        _capability(evidence=(object(),))

    duplicate = (_evidence(), _evidence())
    with pytest.raises(ValueError, match="evidence_id values must be unique"):
        _capability(evidence=duplicate)


def test_evidence_collection_sorts_unique_records() -> None:
    """Evidence ordering cannot change canonical inventory identity."""
    second = _evidence(evidence_id="second_engine_check")
    first = _evidence(evidence_id="first_engine_check")
    capability = _capability(evidence=[second, first])
    assert [row.evidence_id for row in capability.evidence] == [
        "first_engine_check",
        "second_engine_check",
    ]


@pytest.mark.parametrize(
    "status",
    [
        ConformanceCoverageStatus.COVERED,
        ConformanceCoverageStatus.PARTIALLY_COVERED,
    ],
)
def test_supported_coverage_requires_evidence(
    status: ConformanceCoverageStatus,
) -> None:
    """Covered states cannot be asserted with an empty comparison set."""
    with pytest.raises(ValueError, match="covered capability requires evidence"):
        _capability(coverage_status=status, evidence=())


@pytest.mark.parametrize(
    "status",
    [
        ConformanceCoverageStatus.NO_INDEPENDENT_ENGINE,
        ConformanceCoverageStatus.NOT_COMPARABLE,
    ],
)
def test_uncovered_states_are_explicit_and_reject_comparison_evidence(
    status: ConformanceCoverageStatus,
) -> None:
    """Explicit uncovered states cannot carry contradictory comparison rows."""
    capability = _capability(coverage_status=status, evidence=())
    assert capability.coverage_status is status

    with pytest.raises(ValueError, match="must not contain comparison evidence"):
        _capability(coverage_status=status, evidence=(_evidence(),))


def test_partial_coverage_with_evidence_is_valid() -> None:
    """Partial coverage can record the independent slice that exists."""
    capability = _capability(
        coverage_status=ConformanceCoverageStatus.PARTIALLY_COVERED,
        evidence=(_evidence(),),
    )
    assert capability.coverage_status is ConformanceCoverageStatus.PARTIALLY_COVERED


def test_planned_coverage_accepts_only_nonexecuted_rows() -> None:
    """A planned capability cannot smuggle a completed conformance verdict."""
    planned = _capability(
        coverage_status=ConformanceCoverageStatus.PLANNED,
        evidence=(
            _evidence(
                status=ConformanceExecutionStatus.NOT_EXECUTED,
                artifact_sha256=None,
            ),
        ),
    )
    assert planned.evidence[0].execution_status is ConformanceExecutionStatus.NOT_EXECUTED

    with pytest.raises(ValueError, match="planned coverage may contain only"):
        _capability(
            coverage_status=ConformanceCoverageStatus.PLANNED,
            evidence=(_evidence(),),
        )


def test_capability_normalizes_serialized_status_and_manifest() -> None:
    """Capability metadata remains deterministic and JSON-compatible."""
    capability = ConformanceCapability(
        capability_id=" rasch_probability ",
        public_entrypoint=" fast_mlsirm.rasch.probability ",
        estimand=" Dichotomous Rasch response probability ",
        likelihood_family=" bernoulli_logit ",
        parameterization=" difficulty with unit discrimination ",
        identification=" latent location fixed by the compared fixture ",
        comparison_scope=" fixed-parameter equation conformance ",
        coverage_status="covered",  # type: ignore[arg-type]
        evidence=[_evidence()],  # type: ignore[arg-type]
    )
    manifest = capability.to_manifest()
    assert capability.capability_id == "rasch_probability"
    assert capability.coverage_status is ConformanceCoverageStatus.COVERED
    assert manifest["coverage_status"] == "covered"
    assert manifest["schema_version"] == "1.0"


def test_capability_rejects_bad_schema_and_coverage_state() -> None:
    """Capability schema and coverage enums fail closed."""
    with pytest.raises(ValueError, match=r"schema_version must be '1\.0'"):
        _capability(schema_version="2.0")

    with pytest.raises(ValueError, match="coverage_status must be one of"):
        ConformanceCapability(
            capability_id="rasch_probability",
            public_entrypoint="fast_mlsirm.rasch.probability",
            estimand="Rasch probability",
            likelihood_family="bernoulli_logit",
            parameterization="difficulty with unit discrimination",
            identification="latent location fixed",
            comparison_scope="fixed parameter equation",
            coverage_status="unknown",  # type: ignore[arg-type]
            evidence=(_evidence(),),
        )


def test_capability_collection_rejects_container_size_type_and_duplicates() -> None:
    """Inventory capability collections are non-empty, bounded, and package-owned."""
    with pytest.raises(ValueError, match="capabilities must be a list or tuple"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities={_capability()},  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="capabilities must contain between"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(),
        )

    capability = _capability()
    with pytest.raises(ValueError, match="capabilities must contain between"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=[capability] * 129,
        )

    with pytest.raises(
        ValueError,
        match=r"capabilities\[0\] must be a ConformanceCapability",
    ):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(object(),),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="capability_id values must be unique"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(capability, capability),
        )


def test_inventory_rejects_bad_package_commit_and_schema_versions() -> None:
    """Top-level provenance is bound to canonical package and Git identities."""
    capability = _capability()
    with pytest.raises(ValueError, match="package_version must be a canonical"):
        ConformanceInventory(
            package_version="v0.8",
            source_commit=_SOURCE_COMMIT,
            capabilities=(capability,),
        )
    with pytest.raises(ValueError, match="source_commit must be a full lowercase"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit="f" * 39,
            capabilities=(capability,),
        )
    with pytest.raises(ValueError, match=r"schema_version must be '1\.0'"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(capability,),
            schema_version="2.0",
        )


def test_inventory_revalidates_nested_capability_before_text_callbacks() -> None:
    """Mutated package records cannot smuggle callback-bearing text into hashing."""
    _HostileText.callbacks = 0
    capability = _capability()
    object.__setattr__(
        capability,
        "public_entrypoint",
        _HostileText("fast_mlsirm.rasch.probability"),
    )

    with pytest.raises(ValueError, match="public_entrypoint must be a string"):
        ConformanceInventory(
            package_version="0.8.0",
            source_commit=_SOURCE_COMMIT,
            capabilities=(capability,),
        )

    assert _HostileText.callbacks == 0
