"""Buyer-visible regressions for cross-engine conformance evidence rendering."""

from __future__ import annotations

import json

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
from fast_mlsirm.cross_engine_report import (
    render_conformance_long_form_json,
    render_conformance_report,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64


def _canonical_json(inventory: ConformanceInventory) -> str:
    """Return deterministic compact JSON accepted by strict replay."""
    return json.dumps(
        inventory.to_manifest(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _executed_inventory() -> ConformanceInventory:
    """Return one executed capability with complete source-free provenance."""
    engine = ComparisonEngine(
        engine_id="r_mirt",
        engine_version="1.45.1",
        source_reference="CRAN mirt 1.45.1",
        license_classification="open_source",
    )
    evidence = ConformanceEvidence(
        evidence_id="mirt_fixed_equation",
        engine=engine,
        layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
        execution_status=ConformanceExecutionStatus.PASSED,
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256=_SHA_A,
        fixture_sha256=_SHA_B,
        environment_sha256=_SHA_C,
        artifact_sha256=_SHA_D,
        limitation="<script>alert('x')</script> equation-only evidence",
    )
    capability = ConformanceCapability(
        capability_id="dichotomous_probability",
        public_entrypoint="fast_mlsirm.predict_proba",
        estimand="P(Y=1 | theta)",
        likelihood_family="binary_logistic",
        parameterization="<script>unsafe</script> simple structure",
        identification="reference latent scale fixed",
        comparison_scope="fixed parameters on one deterministic fixture",
        coverage_status=ConformanceCoverageStatus.PARTIALLY_COVERED,
        evidence=(evidence,),
    )
    provenance = ConformanceRunProvenance(
        harness_commit="1" * 40,
        environment_sha256=_SHA_C,
        environment_kind=ConformanceEnvironmentKind.ENVIRONMENT_LOCK,
        operating_system="linux",
        architecture="x86_64",
        rng_algorithm="pcg64",
        rng_seeds=(11, 29),
        mapping_schema_version="1.0.0",
        mapping_sha256=_SHA_A,
        model_configuration_sha256=_SHA_E,
        convergence_controls_sha256=_SHA_F,
        tolerance_sha256="0" * 64,
        tolerance_rationale="absolute and relative equation tolerance",
        raw_output_sha256="1" * 64,
        normalized_output_sha256="2" * 64,
        license_classification="open_source",
        redistribution_status=ConformanceRedistributionStatus.METADATA_ONLY,
    )
    return ConformanceInventory(
        package_version="0.1.0",
        source_commit="3" * 40,
        capabilities=(capability,),
        run_provenance=provenance,
    )


def _no_engine_inventory() -> ConformanceInventory:
    """Return one explicit no-independent-engine capability."""
    capability = ConformanceCapability(
        capability_id="high_stakes_decision",
        public_entrypoint="fast_mlsirm.fit",
        estimand="construct-valid high-stakes decision",
        likelihood_family="not_applicable",
        parameterization="not applicable",
        identification="not established by numerical conformance",
        comparison_scope="no independent engine establishes this claim",
        coverage_status=ConformanceCoverageStatus.NO_INDEPENDENT_ENGINE,
        evidence=(),
    )
    return ConformanceInventory(
        package_version="0.1.0",
        source_commit="4" * 40,
        capabilities=(capability,),
    )


def test_report_is_deterministic_and_returns_canonical_json_copy() -> None:
    """One canonical manifest must render byte-identically on every call."""
    inventory = _executed_inventory()
    payload = _canonical_json(inventory)

    first = render_conformance_report(payload)
    second = render_conformance_report(payload)

    assert first == second
    html_text, json_text = first
    assert html_text.startswith("<!doctype html>")
    assert json_text == json.dumps(
        inventory.to_manifest(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def test_report_delegates_tamper_detection_to_strict_inventory_replay() -> None:
    """A report must never render a manifest whose canonical fingerprint is wrong."""
    manifest = _executed_inventory().to_manifest()
    manifest["inventory_fingerprint"] = "9" * 64

    with pytest.raises(ValueError, match="inventory_fingerprint"):
        render_conformance_report(json.dumps(manifest))


def test_report_escapes_untrusted_text_and_exposes_accessible_table_semantics() -> None:
    """Buyer evidence remains text-visible, escaped, and table-accessible."""
    html_text, _ = render_conformance_report(_canonical_json(_executed_inventory()))

    assert "<script>unsafe</script>" not in html_text
    assert "<script>alert('x')</script>" not in html_text
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in html_text
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in html_text
    assert '<meta http-equiv="Content-Security-Policy"' in html_text
    assert "<script" not in html_text.lower()
    assert "<caption>Capability × engine conformance evidence</caption>" in html_text
    assert '<th scope="col">Capability</th>' in html_text
    assert '<th scope="col">Execution status</th>' in html_text
    assert "Exact values are shown in text; this report has no hover-only evidence." in html_text


def test_report_exposes_exact_inventory_and_run_provenance() -> None:
    """A buyer must be able to reconstruct the evidence identity from visible text."""
    inventory = _executed_inventory()
    html_text, _ = render_conformance_report(_canonical_json(inventory))

    assert inventory.inventory_fingerprint in html_text
    assert inventory.package_version in html_text
    assert inventory.source_commit in html_text
    assert inventory.schema_version in html_text
    assert "1" * 40 in html_text
    assert _SHA_A in html_text
    assert _SHA_C in html_text
    assert _SHA_D in html_text
    assert "metadata_only" in html_text
    assert "11, 29" in html_text
    assert "not construct validity, fairness, or high-stakes approval" in html_text


def test_report_renders_explicit_no_engine_and_no_run_states() -> None:
    """Missing independent evidence must remain an explicit non-green state."""
    html_text, _ = render_conformance_report(_canonical_json(_no_engine_inventory()))

    assert "no_independent_engine" in html_text
    assert "No independent engine evidence rows are recorded for this inventory." in html_text
    assert "No run provenance is recorded because this inventory contains no executed evidence." in html_text
    assert "Not recorded" in html_text


def test_long_form_json_is_deterministic_and_provenance_bound() -> None:
    """Downloadable rows retain immutable inventory and engine-evidence identity."""
    inventory = _executed_inventory()
    payload = _canonical_json(inventory)

    first = render_conformance_long_form_json(payload)
    second = render_conformance_long_form_json(payload)

    assert first == second
    rows = json.loads(first)
    assert len(rows) == 1
    row = rows[0]
    assert row["inventory_fingerprint"] == inventory.inventory_fingerprint
    assert row["package_version"] == inventory.package_version
    assert row["source_commit"] == inventory.source_commit
    assert row["schema_version"] == inventory.schema_version
    assert row["capability_id"] == "dichotomous_probability"
    assert row["coverage_status"] == "partially_covered"
    assert row["evidence_id"] == "mirt_fixed_equation"
    assert row["engine_id"] == "r_mirt"
    assert row["execution_status"] == "passed"
    assert row["parameter_mapping_sha256"] == _SHA_A
    assert row["fixture_sha256"] == _SHA_B
    assert row["environment_sha256"] == _SHA_C
    assert row["artifact_sha256"] == _SHA_D
    assert row["limitation"] == "<script>alert('x')</script> equation-only evidence"


def test_long_form_json_preserves_explicit_no_engine_state() -> None:
    """A long-form table must not drop or greenwash capabilities without evidence."""
    inventory = _no_engine_inventory()

    rows = json.loads(render_conformance_long_form_json(_canonical_json(inventory)))

    assert len(rows) == 1
    row = rows[0]
    assert row["inventory_fingerprint"] == inventory.inventory_fingerprint
    assert row["capability_id"] == "high_stakes_decision"
    assert row["coverage_status"] == "no_independent_engine"
    assert row["execution_status"] == "not_executed"
    assert row["evidence_id"] is None
    assert row["engine_id"] is None
    assert row["limitation"] == "No independent engine evidence row is recorded for this capability."


def test_long_form_json_rejects_tampered_manifest_before_projection() -> None:
    """Flattening must use the same strict replay boundary as the human report."""
    manifest = _executed_inventory().to_manifest()
    manifest["inventory_fingerprint"] = "9" * 64

    with pytest.raises(ValueError, match="inventory_fingerprint"):
        render_conformance_long_form_json(json.dumps(manifest))
