"""Regression for execution-backed conformance coverage claims."""

from __future__ import annotations

import unicodedata

import pytest

import fast_mlsirm.cross_engine_conformance as conformance
from fast_mlsirm.cross_engine_conformance import (
    ComparisonEngine,
    ConformanceCapability,
    ConformanceCoverageStatus,
    ConformanceEvidence,
    ConformanceExecutionStatus,
    ConformanceLayer,
)


def test_supported_coverage_rejects_only_nonexecuted_evidence() -> None:
    """Covered claims require at least one comparison that actually executed."""
    engine = ComparisonEngine(
        engine_id="mirt_engine",
        engine_version="1.44.0",
        source_reference="doi:10.18637/jss.v048.i06",
        license_classification="gpl_3_reviewed",
    )
    evidence = ConformanceEvidence(
        evidence_id="rasch_probability_check",
        engine=engine,
        layer=ConformanceLayer.FIXED_PARAMETER_EQUATION,
        execution_status=ConformanceExecutionStatus.NOT_EXECUTED,
        parameter_mapping_version="1.0.0",
        parameter_mapping_sha256="a" * 64,
        fixture_sha256="b" * 64,
        environment_sha256="c" * 64,
        artifact_sha256=None,
    )

    for coverage_status in (
        ConformanceCoverageStatus.COVERED,
        ConformanceCoverageStatus.PARTIALLY_COVERED,
    ):
        with pytest.raises(
            ValueError,
            match="covered capability requires executed evidence",
        ):
            ConformanceCapability(
                capability_id="rasch_probability",
                public_entrypoint="fast_mlsirm.rasch.probability",
                estimand="Dichotomous Rasch response probability",
                likelihood_family="bernoulli_logit",
                parameterization="difficulty with unit discrimination",
                identification="latent location fixed by the compared fixture",
                comparison_scope="fixed-parameter equation conformance",
                coverage_status=coverage_status,
                evidence=(evidence,),
            )


def test_text_normalizes_unicode_to_nfc_before_manifest_identity() -> None:
    """Equivalent Unicode forms must canonicalize before content hashing."""
    decomposed = unicodedata.normalize("NFD", "café")

    normalized = conformance._text(decomposed, "estimand")

    assert normalized == "café"
    assert unicodedata.is_normalized("NFC", normalized)
